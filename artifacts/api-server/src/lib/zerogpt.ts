import { logger } from "./logger";

/**
 * Simplified ZeroGPT client.
 *
 * Two operations: paraphrase a single chunk of text, score a chunk for
 * AI-generation likelihood. No chunking, no tone mapping, no retry loop, no
 * Claude fallback. If ZeroGPT misbehaves, callers get a clear error and the
 * pipeline fails visibly.
 *
 * Config:
 *   ZEROGPT_API_KEY        (required)  - API key from api.zerogpt.com
 *   ZEROGPT_API_BASE_URL   (optional)  - defaults to https://api.zerogpt.com
 *
 * Endpoints used:
 *   POST /api/transform/paraphrase  - paraphrase text
 *     Body: { string, tone, skipRealtime: 1, gen_speed: "quick" }
 *     Response: { success, data: { message: "<paraphrased text>" } }
 *   POST /api/detect/detectText     - score text for AI generation
 *     Body: { input_text }
 *     Response: { success, data: { is_gpt_generated: <0-100> } }
 *
 * Auth header is "ApiKey: <key>" — NOT "Authorization: Bearer".
 */

const ZEROGPT_API_KEY = process.env.ZEROGPT_API_KEY;
const ZEROGPT_BASE_URL = process.env.ZEROGPT_API_BASE_URL ?? "https://api.zerogpt.com";

const PARAPHRASE_TIMEOUT_MS = 300_000; // 5 minutes
const DETECT_TIMEOUT_MS = 25_000;

const VALID_TONES = new Set([
  "Standard",
  "Academic",
  "Fluent",
  "Formal",
  "Simple",
  "Creative",
  "Engineer",
  "Doctor",
  "Lawyer",
  "Teenager",
]);

// Used by the salvage step. If ZeroGPT prefixes its real paraphrase with chatbot
// text like "Here's the rewritten version: <600 words>", strip the prefix and
// keep the body. Anchored to position 0 so it cannot match content mid-article.
const INTRUSION_PREFIX_REGEX =
  /^(?:(?:sure|certainly|absolutely|of course|okay|alright)[!,]?\s*)?(?:here(?:'s| is)[^.\n:]{0,80}(?:rewritten|paraphrased|humanized|reworded) version|please provide(?:[^.\n:]{0,80}text|[^.\n:]{0,80}content)|certainly!?[^.\n:]{0,40}provide)[^.\n:]{0,80}[:.]?\s*/i;

// If the entire response matches this anywhere, it's chatbot meta-text — not
// a paraphrase. Used to decide whether the response needs salvage or rejection.
const INTRUSION_CONTENT_REGEX =
  /please provide the text you would like me to paraphrase|please provide (the )?text to paraphrase|certainly!? please provide|here(?:'s| is)[^.\n:]{0,80}(?:rewritten|paraphrased|humanized|reworded) version|as an ai language model/i;

const MIN_SALVAGED_WORDS = 50;

export class ZeroGptError extends Error {
  constructor(message: string, public readonly cause?: unknown) {
    super(message);
    this.name = "ZeroGptError";
  }
}

export function isZeroGptConfigured(): boolean {
  return Boolean(ZEROGPT_API_KEY);
}

export function getTextStats(text: string): { words: number; chars: number } {
  const trimmed = text.trim();
  const words = trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
  return { words, chars: text.length };
}

/**
 * Paraphrase a chunk of text via ZeroGPT. Returns the paraphrased text or
 * throws ZeroGptError. No retries, no fallbacks — one attempt, real result
 * or visible failure.
 *
 * Tone is passed through if it matches a ZeroGPT preset (Standard, Academic,
 * Fluent, Formal, Simple, Creative, Engineer, Doctor, Lawyer, Teenager),
 * otherwise falls back to Standard.
 *
 * Salvage logic: if ZeroGPT returns its real paraphrase with a chatbot prefix
 * like "Here's the rewritten version: <body>", the prefix is stripped and the
 * body is returned. This is load-bearing — without it, ZeroGPT responses with
 * prefixes get thrown away entirely and the pipeline ships un-humanized text.
 */
export async function humanizeText(text: string, tone?: string | null): Promise<string> {
  if (!ZEROGPT_API_KEY) {
    throw new ZeroGptError("ZEROGPT_API_KEY is not configured");
  }
  if (!text || !text.trim()) {
    throw new ZeroGptError("paraphrase preflight: empty input");
  }

  const safeTone = tone && VALID_TONES.has(tone) ? tone : "Standard";

  const response = await fetchWithTimeout(
    `${ZEROGPT_BASE_URL}/api/transform/paraphrase`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ApiKey: ZEROGPT_API_KEY,
      },
      body: JSON.stringify({
        string: text,
        tone: safeTone,
        skipRealtime: 1,
        gen_speed: "quick",
      }),
    },
    PARAPHRASE_TIMEOUT_MS,
  );

  if (!response.ok) {
    const body = await response.text().catch(() => "<unreadable>");
    throw new ZeroGptError(
      `ZeroGPT paraphrase HTTP ${response.status}: ${body.slice(0, 300)}`,
    );
  }

  const json = (await response.json().catch(() => null)) as
    | { success?: boolean; message?: string; data?: { message?: string } }
    | null;

  if (!json || typeof json !== "object") {
    throw new ZeroGptError("ZeroGPT paraphrase response was not a JSON object");
  }

  if (json.success === false) {
    throw new ZeroGptError(
      `ZeroGPT paraphrase reported failure: ${json.message ?? "unknown"}`,
    );
  }

  const raw = json.data?.message ?? "";
  logger.info(
    { rawPreview: raw.slice(0, 200), totalChars: raw.length },
    "ZeroGPT paraphrase raw response preview",
  );

  if (!raw || !raw.trim()) {
    throw new ZeroGptError("ZeroGPT paraphrase returned empty text");
  }

  // Salvage: strip a chatbot prefix if one is at the start of the response.
  return salvageOrReject(raw);
}

/**
 * Score the AI-generation likelihood of a chunk of text. Returns a percentage
 * (0-100). One attempt, no retries.
 */
export async function scoreAiContent(text: string): Promise<number> {
  if (!ZEROGPT_API_KEY) {
    throw new ZeroGptError("ZEROGPT_API_KEY is not configured");
  }

  const response = await fetchWithTimeout(
    `${ZEROGPT_BASE_URL}/api/detect/detectText`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ApiKey: ZEROGPT_API_KEY,
      },
      body: JSON.stringify({ input_text: text }),
    },
    DETECT_TIMEOUT_MS,
  );

  if (!response.ok) {
    const body = await response.text().catch(() => "<unreadable>");
    throw new ZeroGptError(
      `ZeroGPT detect HTTP ${response.status}: ${body.slice(0, 300)}`,
    );
  }

  const json = (await response.json().catch(() => null)) as
    | { success?: boolean; message?: string; data?: { is_gpt_generated?: number } }
    | null;

  if (!json || typeof json !== "object") {
    throw new ZeroGptError("ZeroGPT detect response was not a JSON object");
  }
  if (json.success === false) {
    throw new ZeroGptError(`ZeroGPT detect reported failure: ${json.message ?? "unknown"}`);
  }

  const score = json.data?.is_gpt_generated;
  if (typeof score !== "number") {
    throw new ZeroGptError("ZeroGPT detect response missing is_gpt_generated");
  }
  return score;
}

// ─── Internals ──────────────────────────────────────────────────────────────

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new ZeroGptError(`ZeroGPT request timed out after ${timeoutMs}ms`, err);
    }
    throw new ZeroGptError(`ZeroGPT request failed: ${err instanceof Error ? err.message : String(err)}`, err);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Decide whether a paraphrase response is acceptable. Three outcomes:
 *   1. Clean response (no chatbot intrusion detected) → return as-is
 *   2. Response with a chatbot prefix that strips down to ≥50 real words → return stripped
 *   3. Pure chatbot meta-text or stripped body too short → throw
 */
function salvageOrReject(raw: string): string {
  if (!INTRUSION_CONTENT_REGEX.test(raw)) {
    return raw; // clean — no salvage needed
  }

  // Try the prefix strip first. If a prefix is found at position 0 and the
  // remaining body is ≥50 words, that's a real paraphrase with chat boilerplate.
  const prefixMatch = raw.match(INTRUSION_PREFIX_REGEX);
  if (prefixMatch && prefixMatch.index === 0) {
    const body = raw.slice(prefixMatch[0].length).trim();
    if (getTextStats(body).words >= MIN_SALVAGED_WORDS) {
      logger.warn(
        { strippedPrefix: prefixMatch[0].slice(0, 80), bodyWords: getTextStats(body).words },
        "ZeroGPT paraphrase had chatbot prefix; stripped and kept the body",
      );
      return body;
    }
  }

  // Not salvageable. Either pure meta-text, or the body after the prefix is
  // too short to be a real paraphrase (a few words is more likely meta-text
  // than a real article chunk).
  throw new ZeroGptError(
    `ZeroGPT paraphrase output was meta-text, not a real paraphrase. First 200 chars: ${raw.slice(0, 200)}`,
  );
}

/**
 * Re-export for callers that need a regex-only intrusion stripper. The
 * structure-restore step in the pipeline uses this to clean up any meta-text
 * that may have leaked through prior pipeline steps.
 */
export function stripIntrusionLines(text: string): { text: string; removedCount: number } {
  const prefixMatch = text.match(INTRUSION_PREFIX_REGEX);
  let withoutPrefix = text;
  let prefixRemoved = false;
  if (prefixMatch && prefixMatch.index === 0) {
    const body = text.slice(prefixMatch[0].length);
    if (body.trim().length > 0) {
      withoutPrefix = body;
      prefixRemoved = true;
    }
  }
  const lines = withoutPrefix.split("\n");
  const kept = lines.filter((line) => !INTRUSION_CONTENT_REGEX.test(line.trim()));
  return {
    text: kept.join("\n"),
    removedCount: (prefixRemoved ? 1 : 0) + (lines.length - kept.length),
  };
}
