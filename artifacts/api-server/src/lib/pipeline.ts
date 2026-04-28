import Anthropic from "@anthropic-ai/sdk";
import { db } from "@workspace/db";
import { articlesTable, pipelineLogsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { logger } from "./logger";
import { publishToGoogleDocs, isGoogleDocsConfigured } from "./google-docs";
import {
  humanizeText,
  scoreAiContent,
  isZeroGptConfigured,
  ZeroGptError,
  getTextStats,
  stripIntrusionLines,
} from "./zerogpt";
import {
  gatherVerifiedSources,
  extractCitations,
  verifyCitations,
  stripUnverifiedCitations,
  type VerifiedSource,
} from "./web-search";

type ArticleStatus =
  | "queued"
  | "researching"
  | "writing"
  | "humanizing"
  | "formatting"
  | "completed"
  | "failed"
  | "flagged";

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const PRIMARY_DENSITY_TARGET_MIN = 1.0;
const PRIMARY_DENSITY_TARGET_MAX = 2.5;
const ENABLE_POST_HUMANIZATION_DENSITY_REBALANCE =
  process.env.ENABLE_POST_HUMANIZATION_DENSITY_REBALANCE === "true";
const DENSITY_REBALANCE_MAX_WORD_DRIFT = 0.08;
const DENSITY_REBALANCE_MIN_IMPROVEMENT = 0.25;
const DRAFT_DENSITY_RETRY_MAX_DISTANCE = 0.2;
const WORD_COUNT_WARNING_TOLERANCE = 200;

// ─── DB helpers ──────────────────────────────────────────────────────────────

async function updateArticleStatus(
  id: number,
  status: ArticleStatus,
  extra?: Partial<typeof articlesTable.$inferSelect>,
) {
  await db.update(articlesTable).set({ status, ...extra }).where(eq(articlesTable.id, id));
}

async function logStep(
  articleId: number,
  stepName: string,
  status: "running" | "completed" | "failed",
  details?: string,
) {
  await db.insert(pipelineLogsTable).values({
    articleId,
    stepName,
    status,
    details: details ?? null,
  });
}

// ─── Text utilities ──────────────────────────────────────────────────────────

function countWords(text: string): number {
  return text.split(/\s+/).filter((w) => w.length > 0).length;
}

function truncateReferenceInput(input: string, maxChars = 12000): string {
  if (input.length <= maxChars) return input;
  const truncated = input.slice(0, maxChars);
  const lastParagraphBreak = truncated.lastIndexOf("\n\n");
  if (lastParagraphBreak > maxChars * 0.8) return truncated.slice(0, lastParagraphBreak);
  return truncated;
}

function calculateKeywordDensity(text: string, keyword: string): number {
  if (!keyword || !text) return 0;
  const normalizedText = text
    .toLowerCase()
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/\[[^\]]+\]\([^)]+\)/g, " ")
    .replace(/https?:\/\/[^\s)\]}>,]+/g, " ")
    .replace(/[#>*_|~]/g, " ")
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const words = normalizedText.split(/\s+/).filter((w) => w.length > 0);
  const totalWords = words.length;
  if (totalWords === 0) return 0;
  const kw = keyword
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!kw) return 0;
  const kwPattern = kw
    .split(/\s+/)
    .map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("\\s+");
  const regex = new RegExp(`\\b${kwPattern}\\b`, "g");
  const matches = normalizedText.match(regex);
  const count = matches ? matches.length : 0;
  return parseFloat(((count / totalWords) * 100).toFixed(2));
}

function densityDistanceFromBand(value: number, min: number, max: number): number {
  if (value < min) return min - value;
  if (value > max) return value - max;
  return 0;
}

function extractFAQs(text: string): string[] {
  const lines = text.split("\n");
  const faqStart = lines.findIndex((line) =>
    /^#{1,3}\s*(FAQ|Frequently Asked Questions|Common Questions)\b/i.test(line.trim()),
  );
  if (faqStart === -1) return [];
  return lines
    .slice(faqStart + 1)
    .map((l) => l.trim())
    .filter(
      (l) => l.length > 0 && !l.startsWith("#") && /^(?:\*\*)?\s*Q(?:uestion)?\s*\d+\s*[:.]/i.test(l),
    );
}

const FAQ_OVERLAP_STOPWORDS = new Set([
  "the","a","an","is","are","was","were","be","been","being","and","or","but","if","then","so",
  "of","to","in","on","at","for","with","by","from","as","it","its","this","that","these","those",
  "i","you","we","they","he","she","what","why","how","when","where","who","which","does","do","did",
  "can","could","should","would","will","has","have","had","not","no","yes","than","about","into",
  "out","your","my","our","their","there","here","more","most",
]);

function extractContentWords(text: string): Set<string> {
  return new Set(
    text.toLowerCase()
      .replace(/[^a-z0-9\s-]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length >= 4 && !FAQ_OVERLAP_STOPWORDS.has(w))
  );
}

/**
 * Soft check: identify FAQ entries whose content significantly overlaps with
 * the article body. Returns the indices of overlapping FAQs (≥70% of meaningful
 * content words appear in the body) so the safety-net check can warn — not
 * fail. The model is asked to avoid this in the writing prompt; this is the
 * post-hoc verification.
 */
/**
 * Extract every H2 and H3 heading from the article, excluding the FAQ section
 * and any sub-headings within it. Once we hit the FAQ heading, we stop
 * collecting (FAQ Q-numbers shouldn't count toward heading-keyword stats).
 */
function extractHeadings(article: string): { level: 2 | 3; text: string }[] {
  const headings: { level: 2 | 3; text: string }[] = [];
  let inFaqSection = false;
  for (const line of article.split("\n")) {
    const m = /^(#{2,3})\s+(.+)$/.exec(line.trim());
    if (!m) continue;
    const level = m[1].length === 2 ? 2 : 3;
    const text = m[2].trim();
    if (/^(FAQ|Frequently Asked Questions|Common Questions)\b/i.test(text)) {
      inFaqSection = true;
      continue;
    }
    if (inFaqSection && level === 2) {
      // A new H2 after the FAQ section means we're back in body content.
      // (Articles don't typically have content after FAQs, but defensively
      //  re-enable counting.)
      inFaqSection = false;
    }
    if (inFaqSection) continue;
    headings.push({ level: level as 2 | 3, text });
  }
  return headings;
}

/**
 * Check primary/secondary keyword presence in headings, per the three rules:
 *   1. At least 30% of H2/H3 headings must include the primary keyword.
 *   2. Every H2/H3 must include at least one primary OR secondary keyword.
 *   3. When secondary keywords are provided, at least 25% of H2/H3 must
 *      include at least one secondary keyword.
 *
 * Matching is case-insensitive and uses whole-word boundaries so "tone" in
 * a heading isn't matched by the keyword "stone." Multi-word keywords match
 * if all their words appear adjacent (any whitespace between).
 *
 * Returns the violation set so the pipeline can decide whether to auto-fix.
 */
type HeadingCheckResult = {
  totalHeadings: number;
  primaryHeadings: number;
  secondaryHeadings: number;
  primaryRatio: number;
  secondaryRatio: number;
  noKeywordHeadings: { level: 2 | 3; text: string }[];
  rule1Pass: boolean; // primary >= 30%
  rule2Pass: boolean; // every heading has primary OR secondary
  rule3Pass: boolean; // secondary >= 25% (only meaningful when secondaries provided)
  hasSecondaryKeywords: boolean;
};

function headingMatchesKeyword(headingText: string, keyword: string): boolean {
  const cleaned = keyword.trim().toLowerCase();
  if (!cleaned) return false;
  const escaped = cleaned.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // For multi-word keywords, allow any whitespace between words.
  const pattern = escaped.split(/\s+/).join("\\s+");
  return new RegExp(`\\b${pattern}\\b`, "i").test(headingText);
}

function checkHeadingKeywords(
  article: string,
  primaryKeyword: string,
  secondaryKeywords: string | null,
): HeadingCheckResult {
  const headings = extractHeadings(article);
  const totalHeadings = headings.length;
  if (totalHeadings === 0) {
    return {
      totalHeadings: 0, primaryHeadings: 0, secondaryHeadings: 0,
      primaryRatio: 0, secondaryRatio: 0, noKeywordHeadings: [],
      rule1Pass: false, rule2Pass: false, rule3Pass: false,
      hasSecondaryKeywords: false,
    };
  }

  const secondaryList = secondaryKeywords
    ? secondaryKeywords.split(",").map((s) => s.trim()).filter((s) => s.length > 0)
    : [];

  let primaryHeadings = 0;
  let secondaryHeadings = 0;
  const noKeywordHeadings: { level: 2 | 3; text: string }[] = [];

  for (const h of headings) {
    const hasPrimary = headingMatchesKeyword(h.text, primaryKeyword);
    const hasSecondary = secondaryList.some((kw) => headingMatchesKeyword(h.text, kw));
    if (hasPrimary) primaryHeadings++;
    if (hasSecondary) secondaryHeadings++;
    if (!hasPrimary && !hasSecondary) noKeywordHeadings.push(h);
  }

  const primaryRatio = primaryHeadings / totalHeadings;
  const secondaryRatio = secondaryList.length > 0 ? secondaryHeadings / totalHeadings : 0;
  const hasSecondaryKeywords = secondaryList.length > 0;

  return {
    totalHeadings,
    primaryHeadings,
    secondaryHeadings,
    primaryRatio,
    secondaryRatio,
    noKeywordHeadings,
    rule1Pass: primaryRatio >= 0.30,
    rule2Pass: noKeywordHeadings.length === 0,
    rule3Pass: !hasSecondaryKeywords || secondaryRatio >= 0.25,
    hasSecondaryKeywords,
  };
}

function summarizeHeadingViolations(check: HeadingCheckResult): string[] {
  const issues: string[] = [];
  if (!check.rule1Pass) {
    issues.push(
      `${check.primaryHeadings}/${check.totalHeadings} headings include primary keyword (target ≥30%, got ${(check.primaryRatio * 100).toFixed(0)}%)`,
    );
  }
  if (!check.rule2Pass) {
    issues.push(
      `${check.noKeywordHeadings.length} heading(s) missing both primary and secondary keywords`,
    );
  }
  if (!check.rule3Pass) {
    issues.push(
      `${check.secondaryHeadings}/${check.totalHeadings} headings include a secondary keyword (target ≥25%, got ${(check.secondaryRatio * 100).toFixed(0)}%)`,
    );
  }
  return issues;
}

function detectFaqBodyOverlap(text: string): { duplicateIndices: number[]; total: number } {
  const lines = text.split("\n");
  const faqStart = lines.findIndex((line) =>
    /^#{1,3}\s*(FAQ|Frequently Asked Questions|Common Questions)\b/i.test(line.trim()),
  );
  if (faqStart === -1) return { duplicateIndices: [], total: 0 };

  const body = lines.slice(0, faqStart).join("\n");
  const bodyWords = extractContentWords(body);

  // Walk the FAQ section, splitting into Q1./Q2./... blocks.
  const faqLines = lines.slice(faqStart + 1);
  const blocks: string[] = [];
  let current = "";
  for (const l of faqLines) {
    const trimmed = l.trim();
    if (/^#{1,3}\s/.test(trimmed)) break; // hit next section
    if (/^(?:\*\*)?\s*Q\d+\s*[:.]/i.test(trimmed)) {
      if (current) blocks.push(current);
      current = l + "\n";
    } else if (current) {
      current += l + "\n";
    }
  }
  if (current) blocks.push(current);

  const duplicateIndices: number[] = [];
  blocks.forEach((block, i) => {
    const words = [...extractContentWords(block)];
    if (words.length < 4) return;
    const overlap = words.filter((w) => bodyWords.has(w));
    const ratio = overlap.length / words.length;
    if (ratio >= 0.70) duplicateIndices.push(i);
  });

  return { duplicateIndices, total: blocks.length };
}

function generateSeoSlug(title: string, keyword: string): string {
  const source = title || keyword || "blog-post";
  return source
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .split(/\s+/)
    .filter((w) => !["a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for"].includes(w))
    .slice(0, 8)
    .join("-")
    .slice(0, 80);
}

function stripChatbotIntrusions(text: string): { text: string; removedCount: number } {
  return stripIntrusionLines(text);
}

// ─── Strict word-comparison verifier ────────────────────────────────────────
// Used to gate the post-humanization Claude structure-restore step. The job:
// answer "did Claude add only markdown markers, or did it actually change
// words?". If words changed, we reject Claude's output and keep ZeroGPT's.
//
// Tolerant of:
//   - Whitespace and newline differences
//   - Markdown structural markers (#, ##, ###, **, |, table separator rows,
//     leading -/+/*/digit. for list bullets)
//   - Smart vs straight quotes (curly → straight)
//   - Case differences caused by a word now starting a heading
// Strict on:
//   - Word substitutions, insertions, deletions
//   - Reordering

function tokenizeForCompare(text: string): string[] {
  if (!text) return [];
  let s = text;
  // Drop fenced code blocks (rare in our pipeline, but defensive).
  s = s.replace(/```[\s\S]*?```/g, " ");
  // Drop table separator rows like "|---|---|".
  s = s.replace(/^\|[\s|:\-]+\|?\s*$/gm, " ");
  // Drop leading bullet markers, list numbers, and blockquote >.
  s = s.replace(/^\s*(?:[-+*]|\d+\.)\s+/gm, "");
  s = s.replace(/^\s*>\s*/gm, "");
  // Drop heading hashes.
  s = s.replace(/^\s*#{1,6}\s+/gm, "");
  // Drop pipes (table cell separators).
  s = s.replace(/\|/g, " ");
  // Drop bold/italic emphasis markers.
  s = s.replace(/\*\*/g, " ").replace(/__/g, " ");
  // Single * and _ used for italic — only when paired around a word.
  s = s.replace(/(^|[^*])\*([^*\s][^*]*?[^*\s]|[^*\s])\*(?!\*)/g, "$1$2");
  s = s.replace(/(^|[^_])_([^_\s][^_]*?[^_\s]|[^_\s])_(?!_)/g, "$1$2");
  // Markdown links → just the text.
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1");
  // Normalize curly quotes.
  s = s
    .replace(/[\u2018\u2019\u201A\u201B]/g, "'")
    .replace(/[\u201C\u201D\u201E\u201F]/g, '"');
  // Collapse all punctuation to spaces (we compare words only).
  s = s.replace(/[^A-Za-z0-9'\-\s]/g, " ");
  s = s.toLowerCase();
  return s.split(/\s+/).filter((w) => w.length > 0);
}

type WordComparisonResult = {
  match: boolean;
  sourceWordCount: number;
  candidateWordCount: number;
  firstDiffs: { index: number; expected: string | undefined; got: string | undefined }[];
};

function compareWordSequences(source: string, candidate: string): WordComparisonResult {
  const a = tokenizeForCompare(source);
  const b = tokenizeForCompare(candidate);
  const firstDiffs: WordComparisonResult["firstDiffs"] = [];
  const max = Math.max(a.length, b.length);
  for (let i = 0; i < max; i++) {
    if (a[i] !== b[i]) {
      firstDiffs.push({ index: i, expected: a[i], got: b[i] });
      if (firstDiffs.length >= 5) break;
    }
  }
  return {
    match: a.length === b.length && firstDiffs.length === 0,
    sourceWordCount: a.length,
    candidateWordCount: b.length,
    firstDiffs,
  };
}

// ─── Claude API helper ───────────────────────────────────────────────────────

type ClaudeGenerationOverrides = {
  temperature?: number;
  system?: string;
  prefill?: string;
  includePrefillInReturn?: boolean;
};

function getAnthropicClient(): Anthropic {
  if (!ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY is not configured");
  return new Anthropic({ apiKey: ANTHROPIC_API_KEY });
}

async function callClaude(
  client: Anthropic,
  prompt: string,
  maxTokens = 8192,
  overrides: ClaudeGenerationOverrides = {},
): Promise<string> {
  const temperature = overrides.temperature ?? 0.85;
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: prompt }];
  const prefill = overrides.prefill ? overrides.prefill.replace(/\s+$/, "") : "";
  if (prefill) messages.push({ role: "assistant", content: prefill });

  const message = await client.messages.create({
    model: "claude-opus-4-5",
    max_tokens: maxTokens,
    temperature,
    ...(overrides.system ? { system: overrides.system } : {}),
    messages,
  });
  const textContent = message.content.find((c) => c.type === "text");
  const generated = textContent ? textContent.text : "";
  const includePrefill = overrides.includePrefillInReturn ?? true;
  return prefill && includePrefill ? prefill + generated : generated;
}

// ─── Main pipeline ────────────────────────────────────────────────────────────

async function runPipeline(articleId: number): Promise<void> {
  logger.info({ articleId }, "Starting pipeline");

  let article;
  try {
    [article] = await db.select().from(articlesTable).where(eq(articlesTable.id, articleId));
    if (!article) {
      logger.error({ articleId }, "Article not found");
      return;
    }
  } catch (err) {
    logger.error({ articleId, err }, "Failed to fetch article");
    return;
  }

  const client = (() => {
    try {
      return getAnthropicClient();
    } catch {
      logger.warn({ articleId }, "No Anthropic API key configured");
      return null;
    }
  })();

  if (!client) {
    await updateArticleStatus(articleId, "failed", {
      errorMessage: "ANTHROPIC_API_KEY is not configured. Please add your API key to run the pipeline.",
    });
    await logStep(articleId, "startup", "failed", "No API key configured");
    return;
  }

  try {
    // Step 1: Input collation
    await logStep(
      articleId,
      "input_collation",
      "completed",
      `Topic: ${article.topic}; Primary keyword: ${article.primaryKeyword}; Audience: ${article.targetAudience || "not provided"}; Target words: ${article.wordCountTarget}`,
    );

    // Step 2: Research
    await updateArticleStatus(articleId, "researching");
    await logStep(articleId, "research", "running", "Building research brief");

    const referenceInput = article.referenceInput ? truncateReferenceInput(article.referenceInput) : "";

    const researchPrompt = `Produce a research brief for the following blog article. The brief will be read by a writer, so it must be self-contained.

TOPIC: ${article.topic}
PRIMARY KEYWORD: ${article.primaryKeyword}
${article.secondaryKeywords ? `SECONDARY KEYWORDS: ${article.secondaryKeywords}` : ""}
${article.targetAudience ? `TARGET AUDIENCE: ${article.targetAudience}` : ""}
${article.tone ? `TONE: ${article.tone}` : ""}
TARGET WORD COUNT: ${article.wordCountTarget}
${referenceInput ? `\nREFERENCE INPUT (prioritize when forming the outline and facts):\n<<<REFERENCE\n${referenceInput}\nREFERENCE>>>\n` : ""}

Produce these sections in order, as Markdown:

## Outline
A list of 5-7 H2 sections with 1-3 sub-bullet points each describing what the section will cover. Each H2 must cover a meaningfully different aspect of the topic.

## Key facts and data
Specific statistics, dates, named sources, or concrete facts worth including in the article. Cite sources inline where possible.

## FAQ candidates
4 to 8 candidate FAQs with short answers. Only include questions that are genuinely useful and whose answers add information NOT already covered in the main outline. Quality over quantity — if you can only justify 4 distinct questions, list 4. Never pad to reach a count.

## Recommended angle
A one-paragraph hook or angle that differentiates this article from competitors.`;

    const researchOutput = await callClaude(client, researchPrompt, 4096, {
      temperature: 0.4,
      system: `You are a research assistant. Produce concise, source-aware research briefs with the exact section headings requested. Stick to facts and concrete specifics.`,
    });
    await logStep(articleId, "research", "completed", `Research brief generated (${countWords(researchOutput)} words)`);

    // Step 2b: Source gathering — DISABLED.
    // The web-search step that constrained the writer to cite only verified
    // sources is currently turned off. The pipeline writes without an explicit
    // source list; the ZeroGPT humanizer transforms the draft as-is.
    //
    // To re-enable: uncomment the block below, restore the original logging,
    // and the writing prompt's citation rules (in buildWritingPrompt) will
    // automatically activate when sourcesBlock is non-empty.
    //
    // await logStep(articleId, "source_gathering", "running", "Searching the web for verified sources");
    // try {
    //   verifiedSources = await gatherVerifiedSources(client, article.topic, article.primaryKeyword, article.secondaryKeywords);
    //   await logStep(articleId, "source_gathering", "completed",
    //     verifiedSources.length > 0 ? `Found ${verifiedSources.length} verified source(s)` : "No sources found");
    // } catch (err) {
    //   logger.warn({ articleId, err }, "Source gathering failed");
    //   await logStep(articleId, "source_gathering", "failed", "Source gathering errored");
    // }
    const verifiedSources: VerifiedSource[] = [];
    const sourcesBlock = "";

    // Step 3: Write article with keyword-density retry.
    // Density is checked on the Claude draft (not the humanized version) because
    // the humanizer paraphrases away keyword instances; trying to enforce density
    // post-humanization would be a fight against the humanizer. We aim for
    // 1.0%–2.5% density on the draft so that some buffer remains after humanization.
    // Up to 2 retries if the draft falls outside the target range.
    await updateArticleStatus(articleId, "writing");

    const targetWords = article.wordCountTarget;
    const MAX_DENSITY_ATTEMPTS = 2; // 1 initial + 1 retry

    const buildWritingPrompt = (densityHint?: { lastDensity: number; tooLow: boolean }) => {
      const densitySection = densityHint
        ? `\n\nPREVIOUS ATTEMPT: primary keyword density was ${densityHint.lastDensity}% (target ${PRIMARY_DENSITY_TARGET_MIN}%–${PRIMARY_DENSITY_TARGET_MAX}%). ${
          densityHint.tooLow
              ? `That's TOO LOW. Slightly increase natural mentions of "${article.primaryKeyword}" where they genuinely fit the context. Keep prose varied and avoid repetitive phrasing.`
              : `That's TOO HIGH. Reduce repeated close-together uses of "${article.primaryKeyword}" by using natural references and varied sentence construction.`
        }\n`
        : "";

      return `Write a complete blog article using the research brief below.

RESEARCH BRIEF:
${researchOutput}
${sourcesBlock ? `\nVERIFIED SOURCES (the ONLY allowed citation pool):\n${sourcesBlock}\n\nCITATION RULES:\n- You may only cite, quote, or attribute claims to sources from the VERIFIED SOURCES list above.\n- When citing, use one of these forms: a markdown link to the source URL, "according to [Source Name]", or "a [year] [Org] [study/report]". The named org or domain MUST appear in the verified list.\n- Do NOT invent sources. Do NOT cite "industry reports" or "experts say" without a named source from the list.\n- If a claim isn't supported by the verified sources, either don't make it, or state it as your own observation without attribution.\n- Use sources sparingly and where they genuinely add credibility — 2 to 5 citations is typical for an article this length.\n` : ""}
ARTICLE SPECIFICATIONS:
- Topic: ${article.topic}
- Primary keyword: "${article.primaryKeyword}" — target density band: ${PRIMARY_DENSITY_TARGET_MIN}% to ${PRIMARY_DENSITY_TARGET_MAX}%. Keep usage natural and contextually relevant; never force repetitive phrasing to hit a numeric target.
${article.secondaryKeywords ? `- Secondary keywords: "${article.secondaryKeywords}" — work these in naturally.` : ""}
${article.targetAudience ? `- Target audience: ${article.targetAudience}` : ""}
${article.tone ? `- Tone: ${article.tone}. Match this tone consistently across the article.` : ""}
- Target word count: ${targetWords} words. Aim for approximately this length, but prioritize quality and natural flow over hitting an exact count.

STRUCTURE:
- H1 title, then H2 sections with optional H3 subsections.
- HEADING KEYWORD RULES: Every H2 and H3 heading must include either the primary keyword or one of the secondary keywords. Aim for the primary keyword to appear in roughly 30-50% of headings (more if the topic warrants), with at least 25% of headings including a secondary keyword when secondary keywords are provided. Weave keywords naturally into the heading's actual subject — do NOT prepend them artificially or stuff them. The FAQ section's main heading stays as "Frequently Asked Questions".
- End with a "Frequently Asked Questions" section containing 4 to 8 Q&A pairs (use "Q1.", "Q2.", ... numbering). Pick the count based on how many genuinely distinct questions the topic supports — never pad to reach a number, never repeat a question whose answer already appears in the body.
- Each FAQ answer MUST cover information not already present in the body of the article. If you cannot write an FAQ whose answer is genuinely new, drop that slot rather than pad.
- Include 1-2 tables and bullet lists where appropriate.
${referenceInput ? `\nREFERENCE INPUT:\n<<<REFERENCE\n${referenceInput}\nREFERENCE>>>\n` : ""}

${article.tone ? `Write in the tone described above.` : "Write in a formal, expert voice."} Prefer direct, concrete statements with named sources and specific numbers. Start the article with the H1 line — no preamble, no commentary.${densitySection}`;
    };

    let articleDraft = "";
    let densityAttempt = 0;
    let lastDraftDensity = 0;

    while (densityAttempt < MAX_DENSITY_ATTEMPTS) {
      densityAttempt++;
      const stepLabel = densityAttempt === 1 ? "writing" : `writing_density_retry_${densityAttempt - 1}`;
      const densityHint = densityAttempt > 1
        ? { lastDensity: lastDraftDensity, tooLow: lastDraftDensity < PRIMARY_DENSITY_TARGET_MIN }
        : undefined;

      await logStep(
        articleId,
        stepLabel,
        "running",
        densityAttempt === 1
          ? `Generating article draft (target ${targetWords} words; primary keyword density target ${PRIMARY_DENSITY_TARGET_MIN}%–${PRIMARY_DENSITY_TARGET_MAX}%)`
          : `Last attempt density was ${lastDraftDensity}% (need ${PRIMARY_DENSITY_TARGET_MIN}%–${PRIMARY_DENSITY_TARGET_MAX}%); regenerating`,
      );

      articleDraft = await callClaude(client, buildWritingPrompt(densityHint), 8192, {
        temperature: 0.85,
        system: `You are writing a long-form SEO blog post for a professional publication. Output markdown only, starting with the H1.`,
      });

      const draftWords = countWords(articleDraft);
      const draftDensity = calculateKeywordDensity(articleDraft, article.primaryKeyword);
      lastDraftDensity = draftDensity;

      await logStep(
        articleId,
        stepLabel,
        "completed",
        `Draft generated (${draftWords} words; primary keyword density ${draftDensity}%)`,
      );

      if (draftDensity >= PRIMARY_DENSITY_TARGET_MIN && draftDensity <= PRIMARY_DENSITY_TARGET_MAX) {
        break; // density in band, accept
      }

      const densityDistance = densityDistanceFromBand(
        draftDensity,
        PRIMARY_DENSITY_TARGET_MIN,
        PRIMARY_DENSITY_TARGET_MAX,
      );
      if (densityDistance <= DRAFT_DENSITY_RETRY_MAX_DISTANCE) {
        break;
      }
    }

    // Step 3b: Citation verification — DISABLED.
    // The mechanical citation extraction/verification/strip step is turned off
    // because source gathering is also disabled (no verified-source list to
    // check against). Re-enabling source gathering above will need this block
    // re-enabled too.
    //
    // const extracted = extractCitations(articleDraft);
    // const unverified = verifyCitations(extracted, verifiedSources);
    // let citationStripped = 0;
    // let articleAfterCitationCheck = articleDraft;
    // if (unverified.length > 0) {
    //   const stripResult = stripUnverifiedCitations(articleDraft, unverified);
    //   articleAfterCitationCheck = stripResult.article;
    //   citationStripped = stripResult.stripped;
    // }
    // const verifiedCitationCount = extracted.length - unverified.length;
    const articleAfterCitationCheck = articleDraft;
    const citationStripped = 0;
    const verifiedCitationCount = 0;

    // Step 3c: Heading-keyword check and one-shot auto-fix (PRE-humanization).
    // Runs on the Claude draft so the model rewriting headings has clean prose
    // to work with. ZeroGPT then humanizes the heading-corrected article.
    //
    // Rules: 30% of headings include primary, every heading has primary OR
    // secondary, 25% include a secondary (when secondaries provided).
    let articleAfterHeadingFix = articleAfterCitationCheck;
    let headingCheck = checkHeadingKeywords(articleAfterCitationCheck, article.primaryKeyword, article.secondaryKeywords);
    const headingViolationsInitial = summarizeHeadingViolations(headingCheck);

    if (headingViolationsInitial.length > 0) {
      await logStep(
        articleId,
        "heading_check",
        "running",
        `Heading violations detected (${headingViolationsInitial.join("; ")}). Attempting one model rewrite before humanization.`,
      );

      try {
        const headingFixPrompt = `The article below has heading-keyword issues that need fixing. Rewrite ONLY the H2 (##) and H3 (###) headings to satisfy these rules. Do NOT change the article body content under any heading — preserve all paragraphs, lists, tables, and FAQ content verbatim.

CURRENT VIOLATIONS:
${headingViolationsInitial.map((v) => `- ${v}`).join("\n")}

KEYWORD RULES TO SATISFY:
- Primary keyword: "${article.primaryKeyword}" — must appear in at least 30% of H2/H3 headings.
${article.secondaryKeywords ? `- Secondary keywords: "${article.secondaryKeywords}" — at least 25% of H2/H3 headings must include at least one of these.` : ""}
- Every H2 and H3 heading must include at least one keyword (primary or secondary).
- Skip the FAQ section's main heading — it stays as "Frequently Asked Questions".

CRITICAL CONSTRAINTS:
- Headings must read naturally. Do NOT keyword-stuff. Do NOT just prepend the keyword to existing headings — rewrite them so the keyword fits the heading's actual subject.
- Preserve heading levels (H2 stays H2, H3 stays H3).
- Preserve heading order.
- Preserve all body content under each heading exactly.
- Return the COMPLETE article with rewritten headings only.

ARTICLE:
${articleAfterCitationCheck}`;

        const fixed = await callClaude(client, headingFixPrompt, 8192, {
          temperature: 0.4,
          system: `You are an SEO editor rewriting article headings to include target keywords naturally. Output the complete article with only the headings changed; body content stays verbatim. Output markdown only, starting with the H1.`,
        });

        if (fixed.trim().length > 0) {
          // Re-check after the fix attempt.
          const recheck = checkHeadingKeywords(fixed, article.primaryKeyword, article.secondaryKeywords);
          const recheckViolations = summarizeHeadingViolations(recheck);
          if (recheckViolations.length < headingViolationsInitial.length) {
            articleAfterHeadingFix = fixed;
            headingCheck = recheck;
            await logStep(
              articleId,
              "heading_check",
              recheckViolations.length === 0 ? "completed" : "failed",
              recheckViolations.length === 0
                ? `Heading rewrite fixed all violations (${headingCheck.primaryHeadings}/${headingCheck.totalHeadings} primary, ${headingCheck.secondaryHeadings}/${headingCheck.totalHeadings} secondary)`
                : `Heading rewrite reduced violations to: ${recheckViolations.join("; ")}. Article continuing to humanization with remaining issues flagged.`,
            );
          } else {
            await logStep(
              articleId,
              "heading_check",
              "failed",
              `Heading rewrite did not improve violations. Article continuing to humanization with: ${headingViolationsInitial.join("; ")}`,
            );
          }
        } else {
          await logStep(articleId, "heading_check", "failed", "Heading rewrite returned empty content. Article continuing with original headings.");
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn({ articleId, err }, "Heading rewrite attempt errored");
        await logStep(articleId, "heading_check", "failed", `Heading rewrite errored: ${msg}. Article continuing with original headings.`);
      }
    } else {
      await logStep(
        articleId,
        "heading_check",
        "completed",
        `Heading keywords OK (${headingCheck.primaryHeadings}/${headingCheck.totalHeadings} include primary, ${headingCheck.secondaryHeadings}/${headingCheck.totalHeadings} include secondary)`,
      );
    }

    // Step 4: ZeroGPT humanization. Single direct call. No chunking, no
    // retries, no Claude fallback. If ZeroGPT fails, the pipeline fails — no
    // silent degradation. The article is sent to ZeroGPT in one piece.
    await updateArticleStatus(articleId, "humanizing");
    let finalArticle = articleAfterHeadingFix;
    let zeroGptScore: number | null = null;
    const humanizationFailed = false;

    if (!isZeroGptConfigured()) {
      await logStep(
        articleId,
        "zerogpt_humanize",
        "failed",
        "ZEROGPT_API_KEY is not configured. Article cannot be humanized.",
      );
      throw new Error("ZEROGPT_API_KEY is not configured");
    }

    {
      await logStep(articleId, "zerogpt_humanize", "running", "Humanizing article with ZeroGPT");
      const humanizeStart = Date.now();
      const inputStats = getTextStats(articleAfterHeadingFix);
      try {
        const humanized = await humanizeText(articleAfterHeadingFix, article.tone);
        if (!humanized || humanized.trim().length === 0) {
          throw new ZeroGptError("ZeroGPT returned empty paraphrase");
        }
        finalArticle = humanized;
        await logStep(
          articleId,
          "zerogpt_humanize",
          "completed",
          `Humanization complete in ${Date.now() - humanizeStart}ms (${inputStats.words} words/${inputStats.chars} chars input → ${countWords(humanized)} words output)`,
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        logger.warn({ articleId, err }, "ZeroGPT humanization failed");
        await logStep(
          articleId,
          "zerogpt_humanize",
          "failed",
          `ZeroGPT humanization failed after ${Date.now() - humanizeStart}ms for ${inputStats.words} words/${inputStats.chars} chars: ${errMsg}`,
        );
        throw err; // Hard fail. No fallback. The article gets status=failed.
      }
    }

    // From here onward, finalArticle is the ZeroGPT-humanized text.
    const humanizationSucceeded = true;

    // Step 4b: Claude structure restore (strict word-preservation).
    // ZeroGPT often returns text with markdown structure damaged — headings
    // glued to paragraphs, FAQ numbering inline, bullets flattened. We ask
    // Claude to restore ONLY markdown markers (#, ##, ###, **, |, -). Claude
    // is forbidden from changing any words. We then verify with a strict
    // tokenized word-comparison: if Claude touched any words, we reject the
    // restored version and keep the ZeroGPT output (with chatbot intrusions
    // already stripped at the humanizer level).
    if (humanizationSucceeded) {
      await logStep(articleId, "claude_structure_restore", "running", "Restoring markdown structure via Claude (strict word-preservation)");
      // Always strip chatbot intrusion lines first. This is regex-only, safe,
      // never changes real article words.
      const intrusionStripped = stripChatbotIntrusions(finalArticle);
      const zerogptOutput = intrusionStripped.text;
      finalArticle = zerogptOutput; // Default: keep ZeroGPT output if Claude restore fails verification.

      try {
        const restorePrompt = `You will restore markdown structure to a paraphrased article. The paraphraser has stripped most markdown markers; you must put them back without changing any words.

REFERENCE ARTICLE (shows the structure that should exist — H1/H2/H3 headings, bold text, bullet lists, tables, FAQ Q-numbering):
<<<REFERENCE
${articleAfterHeadingFix}
REFERENCE>>>

PARAPHRASED ARTICLE (correct words, broken structure — restore the structure):
<<<ARTICLE
${zerogptOutput}
ARTICLE>>>

ABSOLUTE RULES:
- Preserve EVERY word from the paraphrased article in the same order. Do NOT reword, rewrite, paraphrase, fix typos, fix grammar, change tense, change punctuation, add words, remove words, or reorder words.
- Your only allowed action is adding markdown markers: # for H1, ## for H2, ### for H3, ** for bold, | for table cells, - for bullets, line breaks between paragraphs.
- Use the reference to know WHERE each marker belongs (which lines are headings, which lines are bullets, which sections are tables, which paragraphs are FAQ Q&As).
- If the paraphrased article is missing a section that the reference has, do NOT add it. Only restructure what's there.
- If the paraphrased article has a section the reference doesn't, keep it.
- Output the restructured article as markdown. No preamble. No commentary. No explanations.`;

        const restored = await callClaude(client, restorePrompt, 8192, {
          temperature: 0.0,
          system:
            "You are a markdown structure restorer. You add markdown markers to text. You never change, add, remove, or reorder any words. Output the same words in the same order, with markdown markers added.",
        });

        if (restored.trim().length === 0) {
          await logStep(articleId, "claude_structure_restore", "failed", "Claude returned empty content. Keeping ZeroGPT output.");
        } else {
          const cmp = compareWordSequences(zerogptOutput, restored);
          if (cmp.match) {
            finalArticle = restored;
            await logStep(
              articleId,
              "claude_structure_restore",
              "completed",
              `Structure restored (${cmp.sourceWordCount} words preserved exactly; ${intrusionStripped.removedCount} intrusion line(s) stripped pre-restore)`,
            );
          } else {
            const diffSummary = cmp.firstDiffs
              .slice(0, 3)
              .map((d) => `[#${d.index}] expected="${d.expected ?? "<end>"}" got="${d.got ?? "<end>"}"`)
              .join("; ");
            await logStep(
              articleId,
              "claude_structure_restore",
              "failed",
              `Claude changed words (source ${cmp.sourceWordCount} vs candidate ${cmp.candidateWordCount} tokens; first diffs: ${diffSummary}). Keeping ZeroGPT output.`,
            );
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn({ articleId, err }, "Claude structure restore errored");
        await logStep(articleId, "claude_structure_restore", "failed", `Claude structure restore errored: ${msg}. Keeping ZeroGPT output.`);
      }
    }

    // Step 4c: Optional keyword-density rebalance on the post-humanized article.
    const finalDensityBeforeRebalance = calculateKeywordDensity(finalArticle, article.primaryKeyword);
    const densityOutOfBand =
      finalDensityBeforeRebalance < PRIMARY_DENSITY_TARGET_MIN ||
      finalDensityBeforeRebalance > PRIMARY_DENSITY_TARGET_MAX;
    if (densityOutOfBand && ENABLE_POST_HUMANIZATION_DENSITY_REBALANCE) {
      await logStep(
        articleId,
        "density_rebalance",
        "running",
        `Final density is ${finalDensityBeforeRebalance}% (target ${PRIMARY_DENSITY_TARGET_MIN}%–${PRIMARY_DENSITY_TARGET_MAX}%). Attempting one controlled rewrite.`,
      );
      try {
        const rebalancePrompt = `Adjust this article so the PRIMARY KEYWORD density lands between ${PRIMARY_DENSITY_TARGET_MIN}% and ${PRIMARY_DENSITY_TARGET_MAX}%.

PRIMARY KEYWORD: "${article.primaryKeyword}"
CURRENT DENSITY: ${finalDensityBeforeRebalance}%

STRICT RULES:
- Keep markdown structure (H1/H2/H3, lists, tables, FAQ section and Q-numbering) intact.
- Keep the same factual claims and overall meaning.
- Make the smallest possible edits needed to move keyword density into range.
- Do not add preamble or commentary; return only the full markdown article.

ARTICLE:
${finalArticle}`;

        const rebalanced = await callClaude(client, rebalancePrompt, 8192, {
          temperature: 0.3,
          system:
            "You are an SEO editor making minimal edits to tune primary keyword density while preserving structure and meaning. Output markdown only.",
        });

        if (rebalanced.trim().length > 0) {
          const rebalancedDensity = calculateKeywordDensity(rebalanced, article.primaryKeyword);
          const beforeWords = countWords(finalArticle);
          const afterWords = countWords(rebalanced);
          const wordDrift = Math.abs(afterWords - beforeWords) / Math.max(beforeWords, 1);
          const beforeDistance = densityDistanceFromBand(
            finalDensityBeforeRebalance,
            PRIMARY_DENSITY_TARGET_MIN,
            PRIMARY_DENSITY_TARGET_MAX,
          );
          const afterDistance = densityDistanceFromBand(
            rebalancedDensity,
            PRIMARY_DENSITY_TARGET_MIN,
            PRIMARY_DENSITY_TARGET_MAX,
          );
          const rebalancedInBand =
            rebalancedDensity >= PRIMARY_DENSITY_TARGET_MIN &&
            rebalancedDensity <= PRIMARY_DENSITY_TARGET_MAX;
          const materiallyImproved =
            beforeDistance - afterDistance >= DENSITY_REBALANCE_MIN_IMPROVEMENT;
          if (
            wordDrift <= DENSITY_REBALANCE_MAX_WORD_DRIFT &&
            (rebalancedInBand || (afterDistance < beforeDistance && materiallyImproved))
          ) {
            finalArticle = rebalanced;
            await logStep(
              articleId,
              "density_rebalance",
              "completed",
              rebalancedInBand
                ? `Density rebalanced into target band (${finalDensityBeforeRebalance}% → ${rebalancedDensity}%)`
                : `Density improved toward target (${finalDensityBeforeRebalance}% → ${rebalancedDensity}%)`,
            );
          } else {
            await logStep(
              articleId,
              "density_rebalance",
              "failed",
              `Rebalance not accepted (density ${finalDensityBeforeRebalance}% → ${rebalancedDensity}%, word drift ${(wordDrift * 100).toFixed(1)}%). Keeping current version.`,
            );
          }
        } else {
          await logStep(
            articleId,
            "density_rebalance",
            "failed",
            "Density rebalance returned empty content. Keeping current version.",
          );
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn({ articleId, err }, "Density rebalance attempt errored");
        await logStep(
          articleId,
          "density_rebalance",
          "failed",
          `Density rebalance errored: ${msg}. Keeping current version.`,
        );
      }
    } else {
      await logStep(
        articleId,
        "density_rebalance",
        "completed",
        densityOutOfBand
          ? `Skipped post-humanization density rebalance (current ${finalDensityBeforeRebalance}%) to preserve humanized language and reduce AI-detection risk. Set ENABLE_POST_HUMANIZATION_DENSITY_REBALANCE=true to re-enable.`
          : `Final density already in range (${finalDensityBeforeRebalance}%).`,
      );
    }

    if (isZeroGptConfigured()) {
      // Score the article we're actually going to publish
      await logStep(articleId, "zerogpt_score", "running", "Scoring article with ZeroGPT detector");
      try {
        zeroGptScore = await scoreAiContent(finalArticle);
        await logStep(
          articleId,
          "zerogpt_score",
          "completed",
          `ZeroGPT AI score: ${zeroGptScore.toFixed(1)}%`,
        );
      } catch (err) {
        const errMsg = err instanceof ZeroGptError ? err.message : String(err);
        logger.warn({ articleId, err }, "ZeroGPT scoring failed; publishing without score");
        await logStep(
          articleId,
          "zerogpt_score",
          "failed",
          `Scoring failed after retries: ${errMsg}. Article will be published without a score.`,
        );
      }
    } else {
      await logStep(
        articleId,
        "zerogpt_score",
        "failed",
        "ZEROGPT_API_KEY not configured — score skipped",
      );
    }

    // Step 5: Safety-net checks (keyword density + FAQ count + FAQ uniqueness + heading keywords)
    const finalWordCount = countWords(finalArticle);
    const wordCountOutOfBand =
      Math.abs(finalWordCount - targetWords) > WORD_COUNT_WARNING_TOLERANCE;
    const primaryDensity = calculateKeywordDensity(finalArticle, article.primaryKeyword);
    const faqCount = extractFAQs(finalArticle).length;
    const faqCountValid = faqCount >= 4 && faqCount <= 8;
    const densityValid =
      primaryDensity >= PRIMARY_DENSITY_TARGET_MIN &&
      primaryDensity <= PRIMARY_DENSITY_TARGET_MAX;
    const faqOverlap = detectFaqBodyOverlap(finalArticle);

    // Re-check headings on the final article (ZeroGPT humanization could have
    // altered headings). The auto-fix already ran before humanization in
    // step 3c — we don't run it again here. We just record current state.
    const headingCheckFinal = checkHeadingKeywords(finalArticle, article.primaryKeyword, article.secondaryKeywords);

    const issues: string[] = [];
    if (!densityValid) {
      issues.push(
        `primary density ${primaryDensity}% (target ${PRIMARY_DENSITY_TARGET_MIN}-${PRIMARY_DENSITY_TARGET_MAX}%)`,
      );
    }
    if (wordCountOutOfBand) {
      issues.push(
        `word count ${finalWordCount} (target ${targetWords} ±${WORD_COUNT_WARNING_TOLERANCE})`,
      );
    }
    if (!faqCountValid) issues.push(`FAQ count ${faqCount} (allowed: 4-8)`);
    if (faqOverlap.duplicateIndices.length > 0) {
      issues.push(
        `${faqOverlap.duplicateIndices.length} of ${faqOverlap.total} FAQ(s) duplicate body content (Q${faqOverlap.duplicateIndices.map((i) => i + 1).join(", Q")})`,
      );
    }
    const headingViolationsFinal = summarizeHeadingViolations(headingCheckFinal);
    if (headingViolationsFinal.length > 0) {
      issues.push(...headingViolationsFinal);
    }

    if (issues.length > 0) {
      await logStep(
        articleId,
        "safety_checks",
        "failed",
        `Safety-net violations: ${issues.join("; ")}. Article flagged but still delivered.`,
      );
    } else {
      await logStep(articleId, "safety_checks", "completed", `Density ${primaryDensity}%, FAQs ${faqCount}, no FAQ-body overlap, headings OK`);
    }

    // Step 6: SEO metadata
    await updateArticleStatus(articleId, "formatting");
    await logStep(articleId, "seo_metadata", "running", "Generating SEO metadata");

    const h1Match = finalArticle.match(/^#\s+(.+)$/m);
    const h1Title = h1Match ? h1Match[1].trim() : article.topic;
    const firstBodyParagraph =
      finalArticle
        .split("\n")
        .map((l) => l.trim())
        .find(
          (l) => l.length > 40 && !l.startsWith("#") && !l.startsWith("|") && !l.startsWith("-"),
        ) ?? "";

    const seoPrompt = `Produce SEO metadata for this article.

Article H1: ${h1Title}
Primary keyword: ${article.primaryKeyword}
${article.secondaryKeywords ? `Secondary keywords: ${article.secondaryKeywords}` : ""}
First paragraph: ${firstBodyParagraph.slice(0, 400)}

Requirements:
- "title": 50-60 characters, includes the primary keyword.
- "metaDescription": 140-160 characters, includes the primary keyword.
- "slug": lowercase URL-friendly slug derived from the title, hyphen-separated.
- "tags": comma-separated string of exactly 5 relevant tags.

Respond with a single JSON object and nothing else.`;

    let seoData: { title: string; metaDescription: string; slug: string; tags: string } = {
      title: article.topic,
      metaDescription: "",
      slug: "",
      tags: "",
    };
    try {
      const seoRaw = await callClaude(client, seoPrompt, 512, {
        temperature: 0.2,
        system: `You generate SEO metadata. Return a single valid JSON object with exactly the keys "title", "metaDescription", "slug", "tags".`,
        prefill: "{",
      });
      const jsonMatch = seoRaw.match(/\{[\s\S]*\}/);
      if (jsonMatch) seoData = JSON.parse(jsonMatch[0]);
    } catch {
      logger.warn({ articleId }, "SEO metadata parsing failed, using defaults");
    }
    await logStep(articleId, "seo_metadata", "completed", `Title: ${seoData.title}`);

    // Step 7: Google Docs delivery
    let googleDocUrl: string | undefined;
    let docFileName: string | undefined;

    if (isGoogleDocsConfigured()) {
      await logStep(articleId, "google_docs", "running", "Publishing to Google Docs");
      try {
        const docResult = await publishToGoogleDocs({
          title: seoData.title || article.topic,
          content: finalArticle,
        });
        googleDocUrl = docResult.docUrl;
        docFileName = docResult.fileName;
        await logStep(articleId, "google_docs", "completed", `Published: ${googleDocUrl}`);
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        logger.warn({ articleId, err }, "Google Docs publishing failed");
        await logStep(articleId, "google_docs", "failed", `Google Docs error: ${errMsg}`);
      }
    } else {
      await logStep(
        articleId,
        "google_docs",
        "completed",
        "Google Docs not configured — skipped (add GOOGLE_SERVICE_ACCOUNT_JSON to enable)",
      );
    }

    // Step 8: Complete
    await updateArticleStatus(articleId, "completed", {
      title: seoData.title || article.topic,
      articleContent: finalArticle,
      wordCountActual: finalWordCount,
      primaryKeywordDensity: primaryDensity,
      secondaryKeywordDensity: article.secondaryKeywords
        ? calculateKeywordDensity(finalArticle, article.secondaryKeywords.split(",")[0].trim())
        : undefined,
      faqCount,
      zeroGptScore: zeroGptScore ?? undefined,
      humanizationFailed,
      wordCountOutOfBand,
      verifiedSources: verifiedSources.length > 0 ? verifiedSources : undefined,
      citationCount: verifiedCitationCount,
      unverifiedCitationsRemoved: citationStripped,
      seoMetaDescription: seoData.metaDescription,
      seoSlug: seoData.slug || generateSeoSlug(seoData.title, article.primaryKeyword),
      seoTags: seoData.tags,
      completedAt: new Date(),
      googleDocFileName: docFileName ?? undefined,
      googleDocUrl: googleDocUrl ?? undefined,
    });

    logger.info({ articleId, zeroGptScore, humanizationFailed }, "Pipeline completed");
  } catch (err) {
    logger.error({ articleId, err }, "Pipeline failed");
    const errorMessage = err instanceof Error ? err.message : String(err);
    await updateArticleStatus(articleId, "failed", { errorMessage });
    await logStep(articleId, "pipeline", "failed", errorMessage);
  }
}

export { runPipeline };
