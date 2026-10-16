/**
 * Benchmark-first Dynamic Capability Evaluator
 *
 * Quality is derived from REAL third-party benchmark observations when they are
 * present (Artificial Analysis intelligence / coding / agentic indices), and
 * falls back to a structural estimate (parameter scale + architectural tier)
 * only when a model has no benchmark signal at all.
 *
 * Benchmark indices are normalized against the live cohort (a robust 5th–95th
 * percentile band, with a soft-knee tail that keeps models above the upper
 * bound strictly ordered) so the curve auto-adapts as leaderboards move,
 * instead of hard-coding magic score constants. The frontier maps to ~0.98 by
 * construction, and any structural (non-benchmarked) model is capped strictly
 * below the frontier so real benchmark leaders always dominate.
 *
 * Decouples quality evaluation from commercial pricing markups to eliminate
 * proprietary bias.
 */

function clamp(value, min = 0.01, max = 1.0) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return min
  return Math.max(min, Math.min(max, numeric))
}

// Structural fallback is capped here so a benchmarked frontier model (≈0.98)
// always outranks a model we could only score by name/parameter heuristics.
// The cap is kept strictly BELOW the scoring engine's frontier threshold (0.92)
// so an unbenchmarked model is always treated as sub-frontier (penalized, never
// given the frontier ceiling bonus) — benchmark-proven quality wins.
const STRUCTURAL_QUALITY_CAP = 0.90

// Sensible default cohort ranges for single-model (live-sync) evaluation, when
// full-registry bounds have not been computed. Offline recalibration
// (update-registry.js) recomputes real percentile bounds and passes them in.
export const DEFAULT_BENCHMARK_BOUNDS = {
  intelligence: { min: 18, max: 72 },
  coding: { min: 14, max: 84 },
  agentic: { min: 12, max: 80 },
  math: { min: 18, max: 92 },
}

function combineText(model) {
  // Registry records expose `bestFor`; raw API records expose `description`.
  // Include both so parameter/specialization detection works on either shape.
  return `${model.id || ''} ${model.name || ''} ${model.description || ''} ${model.bestFor || ''} ${(model.tags || []).join(' ')}`.toLowerCase()
}

/**
 * Extract active and total parameter scale from description, name, id, and tags.
 */
export function extractParameterScale(model) {
  const text = combineText(model)

  // 1. MoE pattern e.g. "49b active parameters out of 770b total"
  const moeOutMatch = text.match(/(\d+(?:\.\d+)?)\s*b\s+active(?:\s+parameters)?\s+out\s+of\s+(\d+(?:\.\d+)?)\s*b\s+total/i)
  if (moeOutMatch) {
    return { activeParams: Number(moeOutMatch[1]), totalParams: Number(moeOutMatch[2]), isMoE: true }
  }

  // 2. MoE multiplier pattern e.g. "8x22b", "8x7b", "16x12b"
  const moeMultiMatch = text.match(/\b(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b/i)
  if (moeMultiMatch) {
    const experts = Number(moeMultiMatch[1])
    const size = Number(moeMultiMatch[2])
    return { activeParams: size * 2, totalParams: experts * size, isMoE: true }
  }

  // 3. Dense parameter pattern e.g. "405b", "235b", "70b", "32b", "14b", "8b", "1.5b"
  const denseMatch = text.match(/\b(\d+(?:\.\d+)?)\s*b\b/i)
  if (denseMatch) {
    const params = Number(denseMatch[1])
    return { activeParams: params, totalParams: params, isMoE: false }
  }

  return { activeParams: null, totalParams: null, isMoE: false }
}

/**
 * Parameter-to-capability continuous curve. For MoE models we use the geometric
 * mean of active and total parameters (√(active·total)) so a sparsely-activated
 * model is not credited as if all experts ran on every token.
 */
function parameterScore(params) {
  if (!params || !params.totalParams) return null

  const total = params.isMoE && params.activeParams
    ? Math.sqrt(params.activeParams * params.totalParams)
    : params.totalParams

  if (total >= 400) return 0.95       // 405B, 480B, 671B MoE, 770B MoE
  if (total >= 200) return 0.92       // 235B
  if (total >= 100) return 0.89       // 120B-140B
  if (total >= 65)  return 0.86       // 70B, 72B
  if (total >= 28)  return 0.81       // 32B, 34B
  if (total >= 12)  return 0.75       // 14B
  if (total >= 6)   return 0.69       // 7B-8B
  if (total >= 2.5) return 0.61       // 3B
  if (total >= 1)   return 0.53       // 1B-1.5B
  return 0.45                         // Sub-1B
}

/**
 * Infer architectural tier from model name/tags when parameter count is
 * undisclosed. Only GENERIC size/architecture words are used — brand/model
 * family names (opus, sonnet, gpt-4o, o3, r1, …) are deliberately excluded so
 * that a non-benchmarked alias cannot fake the frontier by its name alone.
 */
function inferUndisclosedModelTier(model) {
  const text = `${model.id || ''} ${model.name || ''}`.toLowerCase()

  // A name is not evidence. Every tier below is deliberately modest and below
  // the benchmark curve's mid-range: the previous values let a size word alone
  // place an unrunnable research fixture above real open-weight models
  // (google-t5/t5-small read "small" and scored 0.78, ranking it seventh for a
  // self-hosted coding request). Structural estimates are a last resort and are
  // allowed to be pessimistic; benchmark-observed models are unaffected.

  // Smoke-test / fixture checkpoints are not models at all.
  if (/\b(tiny|test|testing|dummy|fixture|smoke)\b/i.test(text)) return 0.45

  // Encoder-only / seq2seq research families: real models, but not generative
  // assistants, and never a coding or reasoning recommendation.
  if (/(^|[/\s])(t5|bart|bert|roberta|distilbert|albert|electra)([-\s/]|$)/i.test(text)) return 0.48

  // Reasoning / thinking architecture (generic capability signal).
  if (/\b(reasoning|thinking|chain-of-thought)\b/i.test(text)) {
    return 0.74
  }

  // Flagship / frontier / large tier (generic size words)
  if (/\b(ultra|flagship|frontier|large|heavy|max|xl|pro)\b/i.test(text)) {
    return 0.72
  }

  // Efficient / small tier — including "small", which says nothing about
  // absolute scale once the model is placed in a family (GPT-5.6-Small vs
  // t5-small).
  if (/\b(flash|mini|nano|small|lite|micro|fast|turbo|edge)\b/i.test(text)) {
    return 0.64
  }

  // Baseline standard model
  return 0.68
}

/**
 * Detect native reasoning / thinking architecture. The provider prefix of the
 * id (e.g. "thinking-machines/…") is stripped first so a provider name cannot
 * masquerade as a reasoning capability.
 */
function detectReasoningSupport(model) {
  if (model.reasoning && (model.reasoning.default_enabled || model.reasoning.mandatory || model.reasoning.supported_efforts?.length)) {
    return true
  }

  const supported = Array.isArray(model.supported_parameters)
    ? model.supported_parameters
    : (Array.isArray(model.supportedParameters) ? model.supportedParameters : [])
  if (supported.some(p => ['reasoning', 'include_reasoning', 'reasoning_effort'].includes(String(p).toLowerCase()))) {
    return true
  }

  const idNoPrefix = String(model.id || '').split('/').slice(1).join('/') || String(model.id || '')
  const text = `${idNoPrefix} ${model.name || ''} ${model.description || ''} ${model.bestFor || ''} ${(model.tags || []).join(' ')}`.toLowerCase()
  return /\b(reasoning|thinking|chain-of-thought|cot|deepseek-r1|\br1\b|\bo1\b|\bo3\b|\bo4\b)\b/i.test(text)
}

/**
 * Detect tool use & function calling capabilities.
 */
function detectToolSupport(model) {
  const supported = Array.isArray(model.supported_parameters)
    ? model.supported_parameters
    : (Array.isArray(model.supportedParameters) ? model.supportedParameters : [])
  if (supported.some(p => ['tools', 'tool_choice', 'functions', 'structured_outputs'].includes(String(p).toLowerCase()))) {
    return true
  }

  const idNoPrefix = String(model.id || '').split('/').slice(1).join('/') || String(model.id || '')
  const text = `${idNoPrefix} ${model.name || ''} ${model.description || ''} ${model.bestFor || ''}`.toLowerCase()
  return /\b(tool_use|function_call|agent|tools|actions)\b/i.test(text)
}

/**
 * Detect specialized coding capabilities.
 */
function detectCodingSpecialization(model) {
  const text = combineText(model)
  return /\b(code|coder|coding|python|typescript|javascript|rust|c\+\+|sql|debug|developer|software|programming|compiler|swe-bench|humaneval|codestral)\b/i.test(text)
}

/** Read the Artificial Analysis observation block from either field shape. */
export function readArtificialAnalysis(model) {
  if (!model || typeof model !== 'object') return null
  const obs = model.benchmarkObservations || model.benchmarks || null
  const aa = obs && obs.artificial_analysis
  return aa && typeof aa === 'object' ? aa : null
}

function finiteIndex(value) {
  const v = Number(value)
  // Treat null / undefined / non-finite / non-positive as "not evaluated".
  return Number.isFinite(v) && v > 0 ? v : null
}

// Cohort normalization uses a "soft knee". The robust percentile band
// [min, max] maps linearly onto [0, NORM_KNEE]; anything above the upper bound
// maps into the remaining (NORM_KNEE, 1] with exponentially diminishing
// returns.
//
// Why: a hard clip at the upper percentile assigned *every* model at or above
// it the identical value 1.0. On the live cohort that silently collapsed the
// top 8 coding models — genuinely spanning 76.3 to 81.6 — onto one score, so
// "best coding model" returned a tie. Extending the bound to the true maximum
// instead let one exceptional model compress the whole field below it. The knee
// keeps both properties: outlier-robust scaling for the bulk of the
// distribution, and strict ordering across the frontier.
const NORM_KNEE = 0.92
// Tail width as a fraction of the percentile band: the distance above `max` at
// which a model reaches ~63% of the remaining headroom.
const NORM_TAIL_SCALE = 0.30

function normalizeIndex(value, bound) {
  const v = finiteIndex(value)
  if (v === null || !bound) return null
  const min = Number(bound.min)
  const max = Number(bound.max)
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null
  const span = max - min
  if (v <= max) return clamp(((v - min) / span) * NORM_KNEE, 0, 1)
  const excess = (v - max) / (NORM_TAIL_SCALE * span)
  return clamp(NORM_KNEE + (1 - NORM_KNEE) * (1 - Math.exp(-excess)), 0, 1)
}

// Concave curve mapping a cohort-normalized signal [0,1] onto a quality score.
// A 95th-percentile (≈1.0) signal maps to ~0.98; the curve rewards the top end
// while keeping mid-tier models meaningfully separated.
function benchmarkCurve(norm, ceiling = 0.98) {
  return clamp(0.34 + 0.65 * Math.sqrt(clamp(norm, 0, 1)), 0.40, ceiling)
}

function normalizeSpeed(tokensPerSecond) {
  const tps = Number(tokensPerSecond)
  if (!Number.isFinite(tps) || tps <= 0) return null
  // ~40 t/s → 0.4, ~120 t/s → ~0.7, ~300+ t/s → ~0.98
  return clamp(0.25 + (tps / 300) * 0.73, 0.25, 0.98)
}

// NOTE — median time-to-first-token (`ttft_seconds`) is ingested and persisted
// but deliberately NOT folded into the speed coordinate yet.
//
// Throughput alone cannot distinguish a model that streams quickly from one
// that stalls before the first token, so TTFT is the more faithful latency
// signal and the obvious candidate to add. It was implemented and measured
// against the Agent Arena cohort study rather than shipped on plausibility:
// combining throughput and TTFT with the aggregator's own rho = 0.5 power mean
// moved the budget intent's mean regret from 5.10 to 10.55 and its
// net-per-dollar from +29.8 (0/6 tiers dominated) to -20.67 (6/6), because the
// two maps have different calibration and their mean silently shifts the level
// of the dimension rather than only its ordering. An uncalibrated level shift
// is exactly the failure this methodology exists to avoid, so the combination
// was reverted. TTFT should be added when its normalisation can be calibrated
// against a latency-priced reference, not against a cost-blind one.

/**
 * Dynamically evaluate all capabilities of a model.
 *
 * @param {object} model
 * @param {object} [bounds] cohort percentile bounds for benchmark normalization
 *        (defaults to DEFAULT_BENCHMARK_BOUNDS for single-model evaluation).
 */
export function evaluateModelCapabilities(model, bounds = DEFAULT_BENCHMARK_BOUNDS) {
  const params = extractParameterScale(model)
  const pScore = parameterScore(params)

  const isReasoning = detectReasoningSupport(model)
  const hasTools = detectToolSupport(model)
  const isCodeSpecialist = detectCodingSpecialization(model)

  const aa = readArtificialAnalysis(model)
  const b = bounds || DEFAULT_BENCHMARK_BOUNDS

  const iNorm = aa ? normalizeIndex(aa.intelligence_index, b.intelligence) : null
  const cNorm = aa ? normalizeIndex(aa.coding_index, b.coding) : null
  const aRawNorm = aa ? normalizeIndex(aa.agentic_index, b.agentic) : null

  // Agentic: prefer a real agentic index (τ-bench / terminal-bench, mapped in
  // source-sync); otherwise approximate from coding + intelligence + tool use.
  let agenticNorm = aRawNorm
  if (agenticNorm === null && (cNorm !== null || iNorm !== null)) {
    const c = cNorm ?? iNorm ?? 0
    const i = iNorm ?? cNorm ?? 0
    agenticNorm = clamp(0.55 * c + 0.30 * i + 0.15 * (hasTools ? 1 : 0), 0, 1)
  }

  let quality
  let coding
  let intelligence
  let agentic
  let benchDerived = false

  const signals = []
  if (iNorm !== null) signals.push([iNorm, 0.55])
  if (cNorm !== null) signals.push([cNorm, 0.35])
  if (agenticNorm !== null) signals.push([agenticNorm, 0.10])

  if (signals.length) {
    // ── Benchmark-first quality ──
    const wsum = signals.reduce((s, [, w]) => s + w, 0)
    const qBench = signals.reduce((s, [v, w]) => s + v * w, 0) / wsum
    quality = benchmarkCurve(qBench)
    intelligence = iNorm !== null ? benchmarkCurve(iNorm) : quality

    // Real-world software engineering and coding at the frontier require both domain
    // coding proficiency and high-level reasoning/intelligence to architect, debug,
    // and solve algorithmic problems.
    let codingNorm = cNorm
    if (cNorm !== null && iNorm !== null) {
      codingNorm = clamp(0.60 * cNorm + 0.40 * iNorm, 0, 1)
    } else if (cNorm === null && iNorm !== null) {
      codingNorm = iNorm
    }

    coding = codingNorm !== null ? benchmarkCurve(codingNorm, 0.99) : quality
    agentic = agenticNorm !== null ? benchmarkCurve(agenticNorm) : quality
    benchDerived = true
  } else {
    // ── Structural fallback (no benchmark signal) ──
    let baseQuality = pScore !== null ? pScore : inferUndisclosedModelTier(model)
    if (isReasoning) baseQuality += 0.045
    if (hasTools) baseQuality += 0.015
    if (Number(model.context_length || model.contextLength || 0) >= 1000000) baseQuality += 0.015

    quality = clamp(baseQuality, 0.40, STRUCTURAL_QUALITY_CAP)
    intelligence = quality

    coding = quality
    if (isCodeSpecialist) coding = clamp(coding * 1.05, 0.40, STRUCTURAL_QUALITY_CAP)
    if (isReasoning) coding = clamp(coding * 1.04, 0.40, STRUCTURAL_QUALITY_CAP)
    if (hasTools) coding = clamp(coding * 1.02, 0.40, STRUCTURAL_QUALITY_CAP)

    agentic = clamp(0.55 * coding + 0.30 * quality + 0.15 * (hasTools ? 1 : 0), 0.40, STRUCTURAL_QUALITY_CAP)
  }

  // ── Speed ──
  // Prefer observed throughput (Artificial Analysis median output tokens/sec);
  // otherwise estimate inversely from model scale and reasoning overhead.
  const observedSpeed = aa ? normalizeSpeed(aa.tokens_per_second) : null
  let speed
  if (observedSpeed !== null) {
    speed = observedSpeed
  } else {
    const text = `${model.id || ''} ${model.name || ''}`.toLowerCase()
    const isFlashOrMini = /\b(flash|mini|haiku|nano|small|lite|micro|fast|turbo)\b/i.test(text)
    speed = 0.70
    if (isFlashOrMini || (pScore !== null && pScore <= 0.68)) {
      speed = 0.92
    } else if (pScore !== null && pScore <= 0.80) {
      speed = 0.82
    } else if (quality >= 0.92 || isReasoning) {
      speed = 0.58
    }
  }

  // ── Privacy ──
  const access = String(model.access || '').toLowerCase()
  let privacy = 0.45
  if (access.includes('open source') || access.includes('open weight')) {
    privacy = 0.92
  } else if (access.includes('hosted open')) {
    privacy = 0.75
  }

  return {
    quality,
    coding,
    intelligence,
    agentic,
    speed,
    privacy,
    benchDerived,
    // True when `speed` came from a measured throughput observation rather than
    // the structural estimate, so callers can prefer it over a cached value.
    speedObserved: observedSpeed !== null,
    isReasoning,
    hasTools,
    isCodeSpecialist,
    params,
  }
}

/**
 * Compute cohort bounds (robust 5th–95th percentile band) for each Artificial
 * Analysis index across a set of models. Used offline to normalize the whole
 * registry against its own live distribution. Values above the upper bound are
 * ranked by `normalizeIndex`'s soft-knee tail rather than being clipped.
 */
export function computeBenchmarkBounds(models = []) {
  const cols = { intelligence: [], coding: [], agentic: [], math: [] }
  for (const model of models) {
    const aa = readArtificialAnalysis(model)
    if (!aa) continue
    const push = (key, value) => {
      const v = finiteIndex(value)
      if (v !== null) cols[key].push(v)
    }
    push('intelligence', aa.intelligence_index)
    push('coding', aa.coding_index)
    push('agentic', aa.agentic_index)
    push('math', aa.math_index)
  }

  const percentileBounds = (arr, fallback) => {
    if (arr.length < 4) return fallback
    const sorted = [...arr].sort((a, b) => a - b)
    const at = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))]
    // Lower bound stays a 5th-percentile clip: the bottom of the distribution
    // genuinely contains failed / near-zero evaluations that would otherwise
    // stretch the scale.
    const min = at(0.05)
    // Upper bound stays the 95th percentile so a single exceptional model can
    // not compress the entire field beneath it. Models ABOVE this bound are no
    // longer clipped to one identical value — `normalizeIndex` ranks them in a
    // soft-knee tail (see NORM_KNEE), which preserves frontier ordering without
    // sacrificing robustness to outliers.
    const max = at(0.95)
    return max > min ? { min, max } : fallback
  }

  return {
    intelligence: percentileBounds(cols.intelligence, DEFAULT_BENCHMARK_BOUNDS.intelligence),
    coding: percentileBounds(cols.coding, DEFAULT_BENCHMARK_BOUNDS.coding),
    agentic: percentileBounds(cols.agentic, DEFAULT_BENCHMARK_BOUNDS.agentic),
    math: percentileBounds(cols.math, DEFAULT_BENCHMARK_BOUNDS.math),
  }
}

export function getBenchmarkQuality(model, bounds) {
  if (!model || typeof model !== 'object') return null
  return evaluateModelCapabilities(model, bounds).quality
}

export function getBenchmarkSpeed(model, bounds) {
  if (!model || typeof model !== 'object') return null
  return evaluateModelCapabilities(model, bounds).speed
}

export function derivePrivacyScore(model) {
  if (!model || typeof model !== 'object') return 0.45
  return evaluateModelCapabilities(model).privacy
}

export function getFullBenchmark(model, bounds) {
  if (!model || typeof model !== 'object') return null
  const caps = evaluateModelCapabilities(model, bounds)
  const observations = model.benchmarkObservations || model.benchmarks || {}
  const designArena = Array.isArray(observations.design_arena) ? observations.design_arena : []
  const arenaScores = designArena.map((entry) => Number(entry?.elo)).filter(Number.isFinite)
  const arena = arenaScores.length ? Math.round(arenaScores.reduce((sum, score) => sum + score, 0) / arenaScores.length) : null
  const artificialAnalysis = observations.artificial_analysis || {}
  return {
    // These are estimates only when a source has not supplied an observation.
    qualityScore: model.metrics?.quality ?? caps.quality,
    speedScore: model.metrics?.speed ?? caps.speed,
    arena,
    artificialAnalysis,
    benchDerived: caps.benchDerived,
    isReasoning: caps.isReasoning,
    hasTools: caps.hasTools,
    qualitySource: model.metricSources?.quality?.source || 'heuristic estimate',
    speedSource: model.metricSources?.speed?.source || 'heuristic estimate',
  }
}
