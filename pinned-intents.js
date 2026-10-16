/**
 * Pinned intent vectors.
 *
 * The intent parser is a hosted instruction-tuned model (Section "Intent-Parser
 * Evaluation" of the paper), so the pipeline is NOT reproducible end to end from
 * source: without the same credentials and the same provider revision the parser
 * returns different weight vectors, and the two panels that price cost move with
 * those weights. Every number in the paper is nevertheless reproducible, because
 * the parser's output enters the pipeline as a weight vector and those vectors
 * are recorded here exactly as the analysis captured them.
 *
 * `intents.frozen.json` holds the full `analysis` object the shipped parser
 * returned for each prompt on the frozen revision. The harnesses call
 * `pinnedIntent(prompt)` instead of `recommendModels()`, which removes the
 * hosted model from the dependency graph without changing any arithmetic.
 */
import { readFileSync } from 'node:fs'

const PINNED = JSON.parse(readFileSync(new URL('./intents.frozen.json', import.meta.url), 'utf8'))

export async function pinnedIntent(prompt) {
  const analysis = PINNED[prompt]
  if (!analysis) throw new Error(`no pinned intent for prompt: ${prompt}`)
  return { analysis }
}

export const PINNED_PROMPTS = Object.keys(PINNED)
