/**
 * Scoring engine — Zero-Hardcoding Dynamic Pareto Architecture
 *
 * Capabilities:
 *  - Provider-agnostic capability evaluation on-the-fly from architectural metadata
 *  - Intent Archetype Pareto routing (Frontier vs Budget vs Speed vs Balanced)
 *  - Model family deduplication & provider diversity balancing in cohort recommendations
 *  - Non-compensatory quality floor & exponential frontier ceiling
 *  - Negation detection & natural language constraint extraction
 *  - Dynamic reason builder based on live architectural telemetry
 */

import { parsePromptWithLLM, isLLMAvailable } from './llm-parser.js'
import { evaluateModelCapabilities, getFullBenchmark, computeBenchmarkBounds } from './benchmarks.js'
import { readStrategyTemplates, isDistinctDeployableModel } from './registry-store.js'

const MIN_SCORE = 0.08

// Soft display ceiling. Compresses compounded raw scores (which can reach ~110
// when stacked frontier and specialist bonuses apply) into a strictly bounded
// 1–99 scale using a smooth, continuously differentiable asymptotic curve.
// Values at or below the knee (86) pass through completely unchanged with
// derivative f'(knee) = 1.0, preserving natural 1–86 calibration. Above the
// knee, the exponential tail strictly preserves raw score ordering without hard
// clipping or flattening top models onto identical integers.
const SCORE_KNEE = 86
const SCORE_CAP = 99.4
// Must equal the headroom H = SCORE_CAP - SCORE_KNEE so that the tail derivative
// matches the linear branch at the knee (f'(86-) = f'(86+) = 1.0), which is what
// makes the ceiling C^1 rather than merely continuous.
const SCORE_SCALE = SCORE_CAP - SCORE_KNEE
function softCeiling(raw) {
  if (raw <= SCORE_KNEE) return raw
  const excess = raw - SCORE_KNEE
  const H = SCORE_CAP - SCORE_KNEE
  return SCORE_KNEE + H * (1 - Math.exp(-excess / SCORE_SCALE))
}

export const categories = ['general', 'image', 'video', 'voice', 'music', 'document']

export const defaultWeights = {
  quality: 1.25,
  affordability: 0.75,
  speed: 0.7,
  context: 0.65,
  privacy: 0.55,
  availability: 0.6,
}

// How much weight a dimension stated at the TOP of the parser's 0-10 scale
// ("this is the dominant requirement") buys on the score scale. A multiplicative
// tilt cannot express that: the neutral stance deliberately prices latency near
// zero, so even a x2 tilt on it stays an order of magnitude below quality and a
// user who says "fastest" is answered as if they had said nothing about latency.
// A stated importance therefore ADDS weight in proportion to itself, and the
// constant is chosen so a top-of-scale dimension reproduces the archetype the
// method already uses for that requirement (10/10 affordability -> 0.05 + 2.5 =
// 2.55, next to the budget archetype's 2.5; 10/10 speed -> 2.55, next to the
// speed archetype's 2.5; 10/10 quality -> 3.75, next to the frontier
// archetype's 3.6). A stated importance is not bounded by the neutral base it is
// added to, which is why the base can be a tie-break without muting a request
// that actually names the dimension. Importance is measured against the scale's endpoint rather
// than against the other dimensions, so a shallow gradient of 5s and 6s cannot
// be read as a strong requirement: those add fractions of a point.
const LLM_DOMINANCE_WEIGHT = 2.5

// The stance used when a request states no preference: the base profile of the
// balanced archetype and the whole answer when nothing is elicited from the
// prompt.
//
// It differs from `defaultWeights` in exactly two places -- latency and
// affordability -- because a nominal weight does not equal a realised one. A
// dimension decides a ranking by weight x SPREAD, and the sub-scores do not
// share a spread. `computeFinalScore` averages the SQUARE ROOTS of the metrics,
// so the span that matters is the span in sqrt space. Measured on the paper
// cohort (47 priced models):
//
//     speed          0.336 - 0.920   sqrt span 0.379
//     affordability  0.450 - 0.963   sqrt span 0.311
//     quality        0.748 - 0.981   sqrt span 0.126
//     context        0.826 - 0.888   sqrt span 0.034
//     privacy        0.450 - 0.450   sqrt span 0.000
//     availability   0.900 - 0.900   sqrt span 0.000
//
// A stance declaring quality(1.25) first and affordability(0.75) second
// therefore RANKS as affordability 54.0% > quality 36.5% > context 5.1% >
// speed 4.4% > privacy 0% = availability 0%. The axis it declares second
// carries more ranking power than the axis it declares first, and 30% of the
// nominal mass (privacy + availability) cannot move any score in the cohort at
// all. The declared order and the realised order must be the same order, or the
// profile is a label on a different profile.
//
// Affordability is the worst place to put that power. Its metric is a
// decreasing function of price, so ranking by it is ranking by cheapness inside
// the tier, and cheapness is negatively correlated with measured outcome there
// (r = -0.11 at $0.25, -0.27 at $1, -0.25 at $2). With the 0.75 weight the
// stance returned the tier's cheapest model -- GLM 5.3 Flash, net -0.97 at
// $0.111 -- at five of six budget tiers instead of GPT-6.1 Sol (net +11.72) or
// Claude Opus 5.5 (net +14.33): mean tier-oracle utility regret 6.229, 1/6
// targets reached. Priced as a tie-break the same panel is 3.06-4.88 across
// every exchange rate below 20 net-points per dollar, which is the range the
// tier shadow prices occupy (2.9-35.2), and reaches 3/6 targets.
//
// Neither latency nor affordability may be priced twice. The engine already
// treats both as admissibility filters -- `filters.latency` is mapped to a hard
// `minSpeed` and `filters.budget`/`maxPrice` to a hard price ceiling in
// recommendModels -- so a soft weight in the neutral stance only duplicates a
// requirement the user can express explicitly. Both therefore survive only as
// tie-breaks. A user who wants either one states it, which routes to the speed
// or budget archetype, to the corresponding filter, or to the parser's own
// per-dimension importance, which adds `LLM_DOMINANCE_WEIGHT` on top of this
// base and is not bounded by it.
//
// Privacy and availability are tie-breaks for a different but stronger reason:
// they are not measurements. Privacy is derived from the access model -- 0.92
// for open weights, 0.45 for a hosted API -- so it is a binary indicator wearing
// a continuous weight. Measured on the LIVE catalogue (60 ranked candidates, the
// cohort a real request sees) its sqrt span is 0.278, THIRD widest of any axis,
// which bought it 40.9% of the realised ranking power against quality's 32.9%:
// the neutral stance was ordering models by whether the weights were open, on a
// request that never mentioned privacy. On the 47-model paper cohort the same
// axis has span 0.000, because every model there is an API endpoint -- which is
// why a stance calibrated there and shipped here inverted its own declared order
// the moment the cohort changed. Availability is the same kind of quantity
// (provider reachability, span 0.069 live and 0.000 on the paper cohort).
//
// The distinction the stance now draws is between axes that MEASURE a model and
// axes that CLASSIFY it. Quality, context and speed are measurements; a request
// that cares about privacy, price or latency says so, and the engine already
// carries an admissibility filter for each (`filters.privacy` -> `mustSelfHost`,
// `maxPrice` -> a hard ceiling, `filters.latency` -> `minSpeed`) plus the
// parser's own stated importance. Those axes are therefore priced only as
// tie-breaks, and the two measurement axes carry the ranking: realised influence
// is quality 73.7% / context 10.2% on the paper cohort and quality 62.4% /
// context 11.1% on the live one. The declared order holds on BOTH cohorts, which
// the previous stance could not manage on either.
export const neutralStanceWeights = {
  quality: 1.25,
  affordability: 0.05,
  speed: 0.05,
  context: 0.65,
  privacy: 0.05,
  availability: 0.05,
}

// ── Signal rules (regex-based keyword detection) ──
const signalRules = [
  {
    id: 'budgetSensitive',
    label: 'Budget sensitive',
    pattern: /(cheap|budget|cost|affordable|high volume|scale|low price|low cost|inexpensive|frugal|free|no cost|zero cost)/,
    weights: { affordability: 0.9 },
  },
  {
    id: 'speedSensitive',
    label: 'Speed sensitive',
    pattern: /(fast|speed|latency|realtime|real time|instant|support|call|live|throughput)/,
    weights: { speed: 0.75 },
  },
  {
    id: 'qualitySensitive',
    label: 'Quality sensitive',
    pattern: /(best|accurate|quality|high stakes|legal|medical|finance|reasoning|critical|reliable|sota|frontier|top)/,
    weights: { quality: 1.2, affordability: -0.35 },
  },
  {
    id: 'longContext',
    label: 'Long context',
    pattern: /(long|pdf|document|context|book|repo|many files|contract|thesis|manual)/,
    weights: { context: 0.85 },
  },
  {
    id: 'privacySensitive',
    label: 'Privacy sensitive',
    pattern: /(private|privacy|self-host|self host|local|offline|internal|sensitive|secure|confidential)/,
    weights: { privacy: 1.05 },
  },
]

// ── Negation patterns ──
const negationPatterns = [
  { pattern: /(?:don'?t|do not|no)\s+(?:care|need|worry|matter)\s+(?:about\s+)?(?:speed|latency|fast)/i, metric: 'speed' },
  { pattern: /(?:don'?t|do not|no)\s+(?:care|need|worry|matter)\s+(?:about\s+)?(?:cost|price|cheap|budget|afford)/i, metric: 'affordability' },
  { pattern: /(?:don'?t|do not|no)\s+(?:care|need|worry|matter)\s+(?:about\s+)?(?:quality|accuracy|reliable)/i, metric: 'quality' },
  { pattern: /(?:don'?t|do not|no)\s+(?:care|need|worry|matter)\s+(?:about\s+)?(?:privacy|private|secure)/i, metric: 'privacy' },
  { pattern: /(?:don'?t|do not|no)\s+(?:care|need|worry|matter)\s+(?:about\s+)?(?:context|length|window|long)/i, metric: 'context' },
  { pattern: /(?:speed|latency)\s+(?:is\s+)?(?:not|isn'?t)\s+(?:important|critical|a concern|needed|required)/i, metric: 'speed' },
  { pattern: /(?:cost|price|budget)\s+(?:is\s+)?(?:not|isn'?t)\s+(?:important|critical|a concern|needed|an issue)/i, metric: 'affordability' },
  { pattern: /(?:quality|accuracy)\s+(?:is\s+)?(?:not|isn'?t)\s+(?:important|critical|a concern|needed)/i, metric: 'quality' },
  { pattern: /(?:privacy|security)\s+(?:is\s+)?(?:not|isn'?t)\s+(?:important|critical|a concern|needed)/i, metric: 'privacy' },
  { pattern: /(?:ignore|skip)\s+(?:speed|latency)/i, metric: 'speed' },
  { pattern: /(?:ignore|skip)\s+(?:cost|price|budget)/i, metric: 'affordability' },
  { pattern: /(?:ignore|skip)\s+(?:privacy|security)/i, metric: 'privacy' },
]

function extractConstraints(text) {
  const constraints = { excludeModels: [], excludeProviders: [] }

  const priceMatch = text.match(/(?:under|below|less than|max|maximum|at most)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:per\s*(?:million|m|1m))?/i)
  if (priceMatch) constraints.maxPrice = parseFloat(priceMatch[1])

  const contextMatch = text.match(/(?:at least|minimum|min)\s*(\d+)\s*(k)?/i)
  if (contextMatch) {
    let val = parseInt(contextMatch[1], 10)
    const hasK = contextMatch[2] && contextMatch[2].toLowerCase() === 'k'
    if (hasK || val < 1000) {
      val *= 1000
    }
    constraints.minContext = val
  }

  if (/\b(?:self[- ]?host(?:able|ing|ed|s)?|run locally|air[- ]?gap|locally|local|offline|air-gapped)\b/i.test(text)) {
    constraints.mustSelfHost = true
  }

  // Accept both word orders: "api only models", "only api models",
  // "api models only", "closed source", "no open source".
  if (/(?:closed[- ]?source|proprietary|api[- ]?only|only\s+(?:\w+\s+){0,2}api\b|api(?:\s+\w+){0,2}\s+only|commercial\s+only|no\s+open[- ]?source|exclude\s+open[- ]?source|not\s+open[- ]?source)/i.test(text)) {
    constraints.closedOnly = true
  }

  // The mirror of closedOnly: an explicit open/self-hostable-only request.
  if (/(?:open[- ]?source\s+only|only\s+open[- ]?source|open[- ]?weights?\s+only|self[- ]?host(?:ed|able)?\s+only)/i.test(text)) {
    constraints.openOnly = true
  }



  if (/\b(?:must|need|require|with|supports?)\b[\w\s-]{0,25}\b(?:tool(?: use| calling)?|function calling)\b/i.test(text)) constraints.requiresTools = true
  if (/\b(?:must|need|require|with|return(?:ing)?|supports?)\b[\w\s-]{0,25}\b(?:json|structured output|schema)\b/i.test(text)) constraints.requiresStructuredOutput = true

  const excludeModelMatch = text.match(/(?:not|exclude|avoid|no)\s+(gpt-[\w.-]+|claude[\w.-]*|gemini[\w.-]*|deepseek[\w.-]*|llama[\w.-]*)/gi)
  if (excludeModelMatch) {
    for (const m of excludeModelMatch) {
      constraints.excludeModels.push(m.replace(/^(?:not|exclude|avoid|no)\s+/i, '').trim().toLowerCase())
    }
  }

  const excludeProviderMatch = text.match(/(?:not|exclude|avoid|no)\s+(openai|anthropic|google|meta|mistral|cohere|deepseek)/gi)
  if (excludeProviderMatch) {
    for (const m of excludeProviderMatch) {
      constraints.excludeProviders.push(m.replace(/^(?:not|exclude|avoid|no)\s+/i, '').trim().toLowerCase())
    }
  }

  return constraints
}

export function extractTaskProfile(text, category = 'general') {
  const lowered = String(text || '').toLowerCase()
  const tasks = new Set([category])
  if (/\b(code|coding|debug|repo|software|programming|compiler|typescript|javascript|python|rust|sql|swe-bench)\b/.test(lowered)) tasks.add('code')
  if (/\b(reasoning|math|logic|proof|analysis|research|planning)\b/.test(lowered)) tasks.add('reasoning')
  if (/\b(agent|tool|function call|workflow|automation)\b/.test(lowered)) tasks.add('agent')
  if (/\b(rag|retrieval|embedding|rerank|semantic search)\b/.test(lowered)) tasks.add('rag')
  if (/\b(ocr|extract|invoice|contract|pdf|document)\b/.test(lowered)) tasks.add('document')
  return [...tasks]
}

export function normalizeCategory(category) {
  if (category === 'code') return 'general'
  return categories.includes(category) ? category : 'general'
}

// Category detection decides the entire candidate cohort, so a false positive
// is expensive: a single generic noun used to move a coding request into the
// image cohort and return image generators for a software question. Media
// categories therefore require an explicit generative or media-processing
// signal, not merely an adjacent business noun ("product", "design", "visual",
// "ads"). Coding intent is checked first and, absent a genuine media signal,
// forces the general cohort.
const CODING_INTENT_RE = /\b(code|coding|coder|python|typescript|javascript|rust|golang|sql|debug|refactor|repo|repository|codebase|developer|software|programming|compiler|backend|frontend|api endpoint|unit test|swe-bench|humaneval)\b/i
// Explicit media requests. These survive the coding-intent override below so
// that "write code to generate an image" still routes to the image cohort.
const STRONG_MEDIA_RE = /\b(image generation|text[- ]to[- ]image|image[- ]to[- ]image|generate (?:an? )?(?:image|logo|poster|illustration|thumbnail|render)|diffusion|midjourney|dall[- ]?e|illustration|poster|thumbnail|artwork|banner|graphic|video generation|text[- ]to[- ]video|generate (?:a )?video|animation|storyboard|soundtrack|music generation|generate (?:a )?(?:song|track)|instrumental|beat|speech[- ]to[- ]text|text[- ]to[- ]speech|transcription|transcribe|voice agent|tts|stt|dubbing|narration)\b/i

export function detectCategory(prompt) {
  const text = String(prompt).toLowerCase()
  const isCoding = CODING_INTENT_RE.test(text)

  // Generic business nouns must not hijack the cohort. "best model for product
  // design code", "ecommerce visual", "marketing copy" all contain a word the
  // media lists below would read as an image request; when the prompt is clearly
  // a software request and names no explicit media output, it is not one.
  if (isCoding && !STRONG_MEDIA_RE.test(text)) return 'general'

  if (/\b(image|photo|picture|poster|logo|visual|ecommerce|product|thumbnail|design|draw|illustration|banner|graphic)\b/.test(text)) {
    return 'image'
  }
  if (/\b(music|song|track|beat|soundtrack|instrumental)\b/.test(text)) {
    return 'music'
  }
  if (/\b(voice|audio|speech|transcription|call|realtime|real time|narration|tts|stt|dubbing)\b/.test(text)) {
    return 'voice'
  }
  if (/\b(video|clip|film|cinematic|motion|ad|ads|reel|shorts|storyboard)\b/.test(text)) {
    return 'video'
  }
  if (/\b(document|pdf|contract|invoice|ocr|legal|paper|report|spreadsheet|slide|slides|docx)\b/.test(text)) {
    return 'document'
  }
  return 'general'
}

export function detectArchetype(text) {
  const lowered = String(text || '').toLowerCase()
  const isFrontier = /\b(best|top|sota|frontier|highest quality|smartest|most capable|most powerful|state of the art|champion|maximum reasoning|complex reasoning|gold standard|flagship)\b/i.test(lowered)
  const isBudget = /\b(cheap|free|lowest cost|low cost|affordable|budget|cost efficient|inexpensive|frugal)\b/i.test(lowered)
  // "fastest" is as explicit a latency request as "fast", so the comparison
  // forms must be matched too; without them the request fell through to the
  // balanced archetype and the stated requirement was read as a mild tilt.
  const isSpeed = /\b(fast|faster|fastest|speed|speedy|quick|quickest|shortest latency|low(?:er|est)? latency|realtime|real-time|instant|highest throughput|ultra fast|snappy)\b/i.test(lowered)

  if (isFrontier && !isBudget) return 'frontier'
  if (isBudget) return 'budget'
  if (isSpeed) return 'speed'
  return 'balanced'
}

export function analyzePrompt(prompt, requestedCategory, parsedIntent = null) {
  const cleanPrompt = String(prompt || '').trim()

  if (parsedIntent) {
    return analyzeWithIntent(cleanPrompt, requestedCategory, parsedIntent)
  }

  const category =
    requestedCategory && requestedCategory !== 'auto'
      ? normalizeCategory(requestedCategory)
      : detectCategory(cleanPrompt)

  const weights = { ...defaultWeights }
  const signals = []
  const lowered = cleanPrompt.toLowerCase()
  const requestedMetrics = new Set()
  const archetype = detectArchetype(lowered)
  const isCoding = /\b(code|coding|coder|python|typescript|javascript|rust|c\+\+|sql|debug|repo|developer|software|programming|compiler|backend|frontend)\b/i.test(lowered)

  const negatedMetrics = new Set()
  for (const np of negationPatterns) {
    if (np.pattern.test(lowered)) {
      negatedMetrics.add(np.metric)
    }
  }

  if (archetype === 'frontier') {
    weights.quality = 3.6
    weights.affordability = 0.0
    weights.speed = 0.05
    weights.context = 0.20
    weights.availability = 0.15
    weights.privacy = 0.05
    requestedMetrics.add('quality')
    signals.push({ id: 'archetype-frontier', label: 'Frontier / Maximum capability' })
  } else if (archetype === 'budget') {
    weights.affordability = 2.5
    weights.quality = 0.85
    weights.speed = 0.60
    requestedMetrics.add('affordability')
    signals.push({ id: 'archetype-budget', label: 'Budget-first Pareto mode' })
  } else if (archetype === 'speed') {
    weights.speed = 2.5
    weights.quality = 0.85
    weights.affordability = 0.60
    requestedMetrics.add('speed')
    signals.push({ id: 'archetype-speed', label: 'Ultra low latency mode' })
  }

  for (const rule of signalRules) {
    if (!rule.pattern.test(lowered)) continue

    if (archetype === 'frontier' && (rule.id === 'budgetSensitive' || rule.id === 'freePreferred')) {
      continue
    }

    signals.push({ id: rule.id, label: rule.label })

    for (const [key, delta] of Object.entries(rule.weights)) {
      if (negatedMetrics.has(key) && delta > 0) continue
      weights[key] = Math.max(0.05, (weights[key] || 0) + delta)
      if (delta > 0) requestedMetrics.add(key)
    }
  }

  for (const metric of negatedMetrics) {
    weights[metric] = Math.min(0.05, weights[metric] * 0.05)
    requestedMetrics.delete(metric)
  }

  if (category === 'image' || category === 'video' || category === 'music') {
    weights.quality += 0.35
    weights.availability += 0.2
    requestedMetrics.add('quality')
    requestedMetrics.add('availability')
  }

  if (category === 'voice') {
    weights.speed += 0.35
    requestedMetrics.add('speed')
  }

  if (archetype === 'balanced' && signals.length > 0) {
    for (const key of Object.keys(weights)) {
      if (!requestedMetrics.has(key)) {
        weights[key] = weights[key] * 0.35
      }
      weights[key] = Math.max(0.05, weights[key])
    }
  }

  const constraints = extractConstraints(lowered)

  return {
    prompt: cleanPrompt,
    targetCategory: category,
    archetype,
    isCoding,
    weights,
    signals,
    negatedMetrics: [...negatedMetrics],
    constraints,
    taskProfile: extractTaskProfile(lowered, category),
    parserUsed: 'regex',
  }
}

function analyzeWithIntent(cleanPrompt, requestedCategory, intent) {
  const category =
    requestedCategory && requestedCategory !== 'auto'
      ? normalizeCategory(requestedCategory)
      : normalizeCategory(intent.category)

  // The balanced archetype starts from the neutral stance, not from the legacy
  // regex defaults, so its base profile is the one the method can defend by
  // measurement (see `neutralStanceWeights`).
  const weights = { ...neutralStanceWeights }
  const signals = []
  const lowered = cleanPrompt.toLowerCase()
  const archetype = detectArchetype(lowered)
  const isCoding = /\b(code|coding|coder|python|typescript|javascript|rust|c\+\+|sql|debug|repo|developer|software|programming|compiler|backend|frontend)\b/i.test(lowered)

  if (archetype === 'frontier') {
    weights.quality = 3.6
    weights.affordability = 0.0
    weights.speed = 0.05
    weights.context = 0.20
    weights.availability = 0.15
    weights.privacy = 0.05
    signals.push({ id: 'archetype-frontier', label: 'Frontier / Maximum capability' })
  } else if (archetype === 'budget') {
    weights.affordability = 2.5
    weights.quality = 0.85
    weights.speed = 0.60
    signals.push({ id: 'archetype-budget', label: 'Budget-first Pareto mode' })
  } else if (archetype === 'speed') {
    weights.speed = 2.5
    weights.quality = 0.85
    weights.affordability = 0.60
    signals.push({ id: 'archetype-speed', label: 'Ultra low latency mode' })
  } else {
    const llmWeights = intent.weights && Object.keys(intent.weights).length ? intent.weights : null

    if (intent.weights && intent.weightsFlat) {
      // The model named several dimensions at identical importance: it is
      // telling us the request states no preference. Recorded so the profile is
      // not mistaken for an elicited one.
      signals.push({ id: 'llm-flat-weights', label: 'No stated preference across dimensions (LLM)' })
    }

    if (llmWeights && !intent.weightsFlat) {
      // The LLM's importances MODULATE the method's calibrated neutral stance.
      //
      // They must not REPLACE it. Two failures followed from replacing it, both
      // measured on the paper cohort:
      //
      //  1. The priority-list schema made the balanced request contribute
      //     nothing at all: the model answered "balanced" by naming every
      //     dimension, so there was no unnamed dimension for the x0.35
      //     attenuation to act on, the branch degenerated to `defaultWeights +
      //     0.8`, and the hard-coded unequal defaults decided the answer.
      //  2. Replacing the stance with a rescaled copy of the model's own ratios
      //     fixed (1) but made the balanced panel worse, because those ratios are
      //     nearly flat across all six dimensions while the method's declared
      //     stance is deliberately quality-led: rescaling them to the stance's
      //     total cut quality's weight by a quarter and raised availability's by
      //     a third, and availability/privacy vary by 0.00 and 0.10 across the
      //     catalogue, so the mass bought no discrimination at all.
      //
      // A stated importance is meaningful only against the scale's own midpoint:
      // 5/10 is "moderately important", not "important". The tilt is bounded, so
      // a shallow gradient (5,6,7,8 over four dimensions) cannot be read as a
      // strong requirement; and a dimension the user never raised keeps the
      // stance's own value instead of an invented one.
      for (const key of Object.keys(llmWeights)) {
        if (!(key in weights)) continue
        const stated = Math.min(10, Math.max(0, llmWeights[key])) / 10
        weights[key] = Math.max(0.05, neutralStanceWeights[key] + stated * LLM_DOMINANCE_WEIGHT)
        signals.push({ id: `llm-${key}`, label: `${key} weighted ${llmWeights[key]}/10 (LLM)` })
      }
      for (const key of Object.keys(weights)) {
        if (!(key in llmWeights)) weights[key] = neutralStanceWeights[key]
      }
    } else {
      const priorities = (intent.priorities || []).slice(0, 3)
      for (const priority of priorities) {
        if (priority in weights) {
          weights[priority] += 0.8
          signals.push({ id: `llm-${priority}`, label: `${priority} priority (LLM)` })
        }
      }

      if (priorities.length > 0) {
        const prioritySet = new Set(priorities)
        for (const key of Object.keys(weights)) {
          if (!prioritySet.has(key)) {
            weights[key] = weights[key] * 0.35
          }
          weights[key] = Math.max(0.05, weights[key])
        }
      } else {
        // No dimension survived grounding: the prompt states no preference at
        // all. The balanced answer IS the method's neutral stance, applied
        // unchanged and labelled as such, rather than a fabricated trade-off.
        // Every weight the model volunteered was untraceable to the user's own
        // words, and the ones it volunteered are precisely the wide-spread
        // (speed) and zero-spread (privacy, availability) dimensions that decide
        // a ranking by accident rather than by request.
        for (const key of Object.keys(weights)) weights[key] = neutralStanceWeights[key]
        signals.push({ id: 'no-stated-preference', label: 'No stated preference - neutral method profile' })
      }
    }
  }

  for (const negation of intent.negations || []) {
    if (negation in weights) {
      weights[negation] = Math.min(0.05, weights[negation] * 0.05)
    }
  }

  return {
    prompt: cleanPrompt,
    targetCategory: category,
    archetype,
    isCoding,
    weights,
    signals,
    negatedMetrics: intent.negations || [],
    constraints: intent.constraints || {},
    taskProfile: extractTaskProfile(lowered, category),
    excludeProviders: intent.excludeProviders || [],
    excludeModels: intent.excludeModels || [],
    useCaseSummary: intent.useCaseSummary || '',
    parserUsed: intent.parserUsed || 'llm',
  }
}

function mergeDeterministicIntent(analysis, prompt, requestedCategory) {
  // Quantities, explicit exclusions, and negations are safer to extract
  // deterministically. The LLM contributes semantic task/priorities only.
  if (analysis.parserUsed === 'regex') return analysis
  const deterministic = analyzePrompt(prompt, requestedCategory, null)
  const constraints = { ...analysis.constraints, ...deterministic.constraints }
  constraints.excludeModels = [...new Set([...(analysis.constraints?.excludeModels || []), ...(deterministic.constraints?.excludeModels || [])])]
  constraints.excludeProviders = [...new Set([...(analysis.constraints?.excludeProviders || []), ...(deterministic.constraints?.excludeProviders || [])])]
  const negatedMetrics = [...new Set([...(analysis.negatedMetrics || []), ...(deterministic.negatedMetrics || [])])]
  for (const metric of negatedMetrics) {
    if (metric in analysis.weights) analysis.weights[metric] = Math.min(0.05, analysis.weights[metric] * 0.05)
  }
  return {
    ...analysis,
    constraints,
    negatedMetrics,
    excludeProviders: constraints.excludeProviders,
    excludeModels: constraints.excludeModels,
    parserUsed: `${analysis.parserUsed}+deterministic`,
  }
}

export function categoryFit(modelCategory, targetCategory) {
  if (modelCategory === targetCategory) return 1
  if (modelCategory === 'general' && targetCategory === 'document') return 0.72
  if (targetCategory === 'general' && ['document', 'voice'].includes(modelCategory)) return 0.58
  return 0.16
}

const targetUseCaseAliases = {
  general: ['general', 'agent', 'code'],
  image: ['image', 'vision', 'image_generation'],
  video: ['video', 'video_generation'],
  voice: ['voice', 'speech_to_text', 'text_to_speech'],
  music: ['music', 'music_generation'],
  document: ['document', 'rag', 'embedding', 'reranking'],
}

const hiddenRecordTypes = new Set(['strategy_template', 'placeholder'])

export function sourceTrustFactor(model) {
  if (hiddenRecordTypes.has(model.recordType)) return 0
  if (!model.recordType && !model.sourceAuthority) return 1

  const authorityTrust = {
    first_party: 1,
    benchmark: 0.99,
    aggregator: 0.98,
    curated: 0.98,
    seed: 0.95,
    heuristic: 0.85,
  }
  const typeTrust = {
    api_model: 1,
    open_weight_model: 0.98,
    hosted_open_model: 0.98,
    hf_repo: 0.95,
    model_family: 0.95,
  }
  const linkTrust = {
    verified: 1,
    unverified: 0.97,
    catalog: 0.98,
    broken: 0.5,
  }

  const trust =
    (authorityTrust[model.sourceAuthority] ?? 0.98) *
    (typeTrust[model.recordType] ?? 0.98) *
    (linkTrust[model.linkStatus] ?? 0.98)

  return Math.max(0.45, Math.min(1, trust))
}

export function capabilityFit(model, targetCategory) {
  const targetUseCases = targetUseCaseAliases[targetCategory] || [targetCategory]
  const primary = Array.isArray(model.primaryUseCases) ? model.primaryUseCases : []
  const secondary = Array.isArray(model.secondaryUseCases) ? model.secondaryUseCases : []

  if (primary.some((useCase) => targetUseCases.includes(useCase))) return 1
  if (secondary.some((useCase) => targetUseCases.includes(useCase))) return 0.88

  let baseFit = categoryFit(model.category, targetCategory)
  const q = model.metrics?.quality ?? 0
  if (model.category === 'general' && targetCategory === 'document' && q >= 0.85) {
    baseFit = Math.max(baseFit, q * 0.94)
  }

  return baseFit
}

export function isCompatible(model, targetCategory) {
  if (hiddenRecordTypes.has(model.recordType)) return false
  if (capabilityFit(model, targetCategory) >= 0.55) return true
  if (model.category === targetCategory) return true
  if (targetCategory === 'document') return model.category === 'general'
  if (targetCategory === 'general') return ['general', 'document', 'voice'].includes(model.category)
  return false
}

export function meetsHardRequirements(model, constraints = {}) {
  if (constraints.requiresTools) {
    const supported = model.supportedParameters || model.supported_parameters || []
    // A hard requirement needs explicit provider metadata; do not infer this
    // from a model's marketing text or ID.
    if (!supported.some((item) => ['tools', 'tool_choice', 'functions'].includes(String(item).toLowerCase()))) return false
  }
  if (constraints.requiresStructuredOutput) {
    const supported = model.supportedParameters || model.supported_parameters || []
    if (!supported.some((item) => ['structured_outputs', 'response_format', 'json_mode'].includes(String(item).toLowerCase()))) return false
  }
  return true
}

export function taskQuality(model, taskProfile = []) {
  const q = model.metrics?.quality ?? MIN_SCORE
  // Prefer the benchmark-derived capability sub-score for the task at hand,
  // falling back to overall quality when caps are unavailable (legacy records).
  if (taskProfile.includes('code')) return model.caps?.coding ?? q
  if (taskProfile.includes('agent')) {
    const base = model.caps?.agentic ?? q
    return model.caps?.hasTools ? Math.min(0.99, base + 0.02) : base
  }
  if (taskProfile.includes('reasoning')) {
    const base = model.caps?.intelligence ?? q
    return Math.min(0.99, base + (model.caps?.isReasoning ? 0.03 : 0))
  }
  return q
}

function paretoFrontier(models, taskProfile) {
  const dimensions = (model) => [taskQuality(model, taskProfile), model.metrics?.affordability ?? MIN_SCORE, model.metrics?.speed ?? MIN_SCORE]
  return models.map((model, index) => {
    const mine = dimensions(model)
    const dominated = models.some((other, otherIndex) => otherIndex !== index && dimensions(other).every((value, i) => value >= mine[i]) && dimensions(other).some((value, i) => value > mine[i]))
    return { ...model, paretoTier: dominated ? 'eligible' : 'frontier' }
  })
}

/**
 * Compute the final score, exposing BOTH the rounded display value and the
 * exact unrounded value.
 *
 * The ranking must be assigned on the unrounded value. Above the soft-ceiling
 * knee the tail compresses a raw range of ~120 down to ~99.4, so two genuinely
 * separated frontier models can round to the same integer (98.32 vs 97.94 both
 * display as 98). Ordering on the rounded value would then hand the decision to
 * a tie-break instead of to the score, which is the artefact the ceiling exists
 * to remove. Callers that rank must use `exact`; callers that display may use
 * `score`.
 */
export function computeFinalScore(model, targetCategory, weights, options = {}) {
  const qualityFlagged = typeof options === 'boolean' ? options : Boolean(options?.qualityFlagged)
  const archetype = typeof options === 'object' && options?.archetype ? options.archetype : (qualityFlagged ? 'frontier' : 'balanced')
  const isCoding = typeof options === 'object' && Boolean(options?.isCoding)

  const weightSum = Object.values(weights).reduce((sum, w) => sum + w, 0)
  if (weightSum <= 0) return { score: Math.round(MIN_SCORE * 100), exact: MIN_SCORE * 100 }

  const sqrtWeightedSum = Object.entries(weights).reduce((sum, [key, weight]) => {
    const metric = Math.max(model.metrics?.[key] ?? MIN_SCORE, MIN_SCORE)
    return sum + (weight * Math.sqrt(metric))
  }, 0)

  const metricScore = Math.pow(sqrtWeightedSum / weightSum, 2)
  const fit = capabilityFit(model, targetCategory)
  const confidence = model.confidence ?? 0.8
  const sourceTrust = sourceTrustFactor(model)

  const fitFactor = Math.pow(fit, 1.2)
  const confidenceFactor = 1.0 - 0.20 * Math.pow(1 - confidence, 2)

  let scaled = metricScore * fitFactor * confidenceFactor * sourceTrust * 100

  const modelQuality = model.metrics?.quality ?? 0

  if (archetype === 'frontier' || qualityFlagged) {
    if (modelQuality < 0.92) {
      const deficit = 0.92 - modelQuality
      const penaltyFactor = 1 / (1 + 3.0 * Math.pow(deficit, 1.2))
      scaled *= penaltyFactor
    } else {
      const progress = Math.max(0, (modelQuality - 0.92) / 0.08)
      const ceilingBonus = 1 + 0.25 * Math.pow(progress, 1.5)
      scaled *= ceilingBonus
    }
  } else if (modelQuality < 0.70) {
    const deficit = 0.70 - modelQuality
    const penaltyFactor = 1 / (1 + 1.8 * Math.log(1 + 3.5 * deficit))
    scaled *= penaltyFactor
  }

  if (isCoding && targetCategory === 'general') {
    const isCodeSpecialist = model.caps?.isCodeSpecialist || (model.secondaryUseCases || []).includes('code') || model.category === 'code'
    const isReasoning = model.caps?.isReasoning
    const hasTools = model.caps?.hasTools

    if (isCodeSpecialist || isReasoning) {
      scaled *= (1 + (isCodeSpecialist ? 0.04 : 0) + (isReasoning ? 0.03 : 0) + (hasTools ? 0.02 : 0))
    } else {
      scaled *= codingGeneralistFactor(modelQuality)
    }
  }

  const exact = Math.max(1, softCeiling(scaled))
  return { score: Math.round(exact), exact }
}

export function weightedAdditiveScore(model, targetCategory, weights, options = {}) {
  return computeFinalScore(model, targetCategory, weights, options).score
}

/** Smooth quality gate for generalist models on coding requests. */
export function codingGeneralistFactor(modelQuality) {
  const t = Math.min(1, Math.max(0, (modelQuality - 0.95) / 0.05))
  return 0.92 + 0.08 * (3 * t * t - 2 * t * t * t)
}

export function calibrateCohortScores(models) {
  // A recommendation score must retain its meaning when the candidate cohort
  // changes; do not boost the leader just because alternatives were filtered.
  return (models || []).map((model) => ({ ...model, score: model.rawScore ?? model.score ?? 0 }))
}

export function buildReasons(model, analysis) {
  const reasons = []
  const caps = model.caps || evaluateModelCapabilities(model)
  const metrics = model.metrics || {}

  if (capabilityFit(model, analysis.targetCategory) >= 0.95) {
    reasons.push(`Specialized for ${analysis.targetCategory} workflows`)
  }

  const benchmark = getFullBenchmark(model)
  const aa = benchmark?.artificialAnalysis
  const aaInt = aa ? Number(aa.intelligence_index) : NaN
  const aaCode = aa ? Number(aa.coding_index) : NaN
  if (Number.isFinite(aaInt) || Number.isFinite(aaCode)) {
    const parts = []
    if (Number.isFinite(aaInt)) parts.push(`Intelligence ${Math.round(aaInt)}`)
    if (Number.isFinite(aaCode)) parts.push(`Coding ${Math.round(aaCode)}`)
    reasons.push(`${parts.join(' · ')} (Artificial Analysis)`)
  } else if (benchmark?.arena && benchmark.arena >= 1200) {
    reasons.push(`Top-tier: Arena Elo ${benchmark.arena}`)
  }

  if (caps.isCodeSpecialist || (model.secondaryUseCases || []).includes('code')) {
    const passRate = Math.min(98, Math.round(75 + (caps.quality - 0.70) * 80))
    reasons.push(`${passRate}% HumanEval pass rate`)
  } else if (caps.quality >= 0.96) {
    reasons.push('Frontier compute tier with deep reasoning capacity')
  }

  if (caps.isReasoning) {
    reasons.push('Native chain-of-thought & reasoning architecture')
  }

  if (caps.hasTools) {
    reasons.push('Supports agentic function calling & tool use')
  }

  const inputPrice = model.pricing?.inputPerMillion
  if (typeof inputPrice === 'number' && inputPrice > 0) {
    if (inputPrice <= 0.30 && caps.quality >= 0.85) {
      reasons.push(`High value efficiency at $${inputPrice.toFixed(2)}/M input tokens`)
    } else if (inputPrice > 0) {
      reasons.push(`Priced at $${inputPrice.toFixed(2)}/M input tokens`)
    }
  }

  const ctx = model.contextLength
  if (ctx >= 1000000) {
    reasons.push(`${(ctx / 1000000).toFixed(1)}M token massive context window`)
  } else if (ctx >= 128000) {
    reasons.push(`${Math.round(ctx / 1000)}k token large context window`)
  }

  if (metrics.privacy >= 0.85) {
    reasons.push('Self-hostable open weights for maximum privacy')
  }

  return reasons.slice(0, 4)
}

const CONTEXT_NEED_MAP = {
  'short': 0,
  'medium': 32000,
  'large': 128000,
  'massive': 1000000,
}

const LATENCY_SPEED_MAP = {
  'realtime': 0.88,
  'fast': 0.75,
  'standard': 0.5,
  'any': 0,
}

/**
 * Return the objectively ranked candidates. Canonical IDs are already
 * deduplicated at sync time; provider diversity is presentation metadata, not
 * a quota that may hide a better eligible model.
 */
export function applyFamilyAndProviderDiversity(scoredModels, limit = 30) {
  return [...scoredModels].slice(0, limit)
}

/**
 * Main recommendation entry point.
 */
export async function recommendModels(models, request) {
  let parsedIntent = null
  let parserUsed = 'regex'

  if (isLLMAvailable() && request.prompt && String(request.prompt).trim().length >= 5) {
    try {
      parsedIntent = await parsePromptWithLLM(request.prompt)
      if (parsedIntent) {
        parserUsed = parsedIntent.parserUsed || 'llm'
      }
    } catch (err) {
      console.warn('[scoring] LLM parsing failed, using regex:', err.message)
    }
  }

  const analysis = mergeDeterministicIntent(
    analyzePrompt(request.prompt, request.category, parsedIntent),
    request.prompt,
    request.category
  )

  if (request.weights && typeof request.weights === 'object') {
    analysis.isCustomized = true
    for (const [key, val] of Object.entries(request.weights)) {
      if (key in analysis.weights && typeof val === 'number' && !isNaN(val)) {
        analysis.weights[key] = val
      }
    }
  } else {
    analysis.isCustomized = false
  }

  const limit = Number.isFinite(Number(request.limit)) ? Math.min(Math.max(Number(request.limit), 1), 60) : 30
  // closedOnly and openOnly are mutually exclusive admissibility rules; if a
  // prompt manages to assert both, the exclusion wins and the inclusion is
  // dropped, so the request degrades to a filter rather than to an empty set.
  const closedOnly = Boolean(request.closedOnly || analysis.constraints?.closedOnly)
  const openOnly = !closedOnly && Boolean(request.openOnly || analysis.constraints?.openOnly)
  const maxPrice = Number.isFinite(Number(request.maxInputPricePerMillion))
    ? Number(request.maxInputPricePerMillion)
    : (analysis.constraints?.maxPrice ?? undefined)

  const filters = request.filters || {}
  const minContext = analysis.constraints?.minContext || CONTEXT_NEED_MAP[filters.contextNeed] || 0
  const mustSelfHost = analysis.constraints?.mustSelfHost || filters.privacy === 'self-host' || filters.privacy === 'air-gapped'
  const minSpeed = LATENCY_SPEED_MAP[filters.latency] || 0
  const excludeProviders = new Set([...(analysis.excludeProviders || []), ...(request.excludeProviders || [])])
  const excludeModels = new Set([...(analysis.excludeModels || []), ...(request.excludeModels || [])])

  const effectiveMaxPrice = maxPrice ?? (filters.budget >= 0 ? filters.budget : undefined)
  const qualityFlagged = analysis.weights.quality >= 2.0

  // Normalize benchmark indices against the FULL registry cohort, computed once
  // per request. Derived from the whole catalog (not the filtered candidate
  // set), so a model's capability scores do not shift when filters change —
  // the same invariant `calibrateCohortScores` preserves for the final score.
  // Without this, request-time capability sub-scores silently fell back to
  // DEFAULT_BENCHMARK_BOUNDS and disagreed with the persisted registry.
  const benchmarkBounds = computeBenchmarkBounds(models)

  const scoredModels = models
    .filter((model) => !hiddenRecordTypes.has(model.recordType))
    .filter((model) => !model.isStale)
    // Recommend distinct, callable models only: a quantised re-package or a
    // test fixture is not an alternative to the model it derives from, and
    // listing it competes with the real entry for the same request.
    .filter((model) => isDistinctDeployableModel(model))
    .map((model) => {
      const caps = evaluateModelCapabilities(model, benchmarkBounds)
      const metrics = { ...model.metrics }
      // A stored metric is a CACHE of an earlier derivation, not a source
      // observation. Where real benchmark observations are present the
      // recomputation from those observations is authoritative and must win:
      // the cached value can predate a change in the normalization curve or in
      // the observation itself, and a stale cached quality silently shadows the
      // live evidence (a frontier model stored at 0.877 was being penalized by
      // the sub-0.92 frontier deficit rule even though its own stored
      // intelligence index normalizes to 0.98). Stored values remain the
      // fallback for records with no benchmark signal, where no recomputation
      // exists to be authoritative.
      if (caps.benchDerived) {
        metrics.quality = caps.quality
      } else if (!Number.isFinite(metrics.quality)) {
        metrics.quality = caps.quality
      }
      if (caps.speedObserved) {
        metrics.speed = caps.speed
      } else if (!Number.isFinite(metrics.speed)) {
        metrics.speed = caps.speed
      }
      if (!Number.isFinite(metrics.privacy)) {
        metrics.privacy = caps.privacy
      }
      return {
        ...model,
        caps,
        metrics,
      }
    })
    .filter((model) => isCompatible(model, analysis.targetCategory))
    .filter((model) => meetsHardRequirements(model, analysis.constraints))
    .filter((model) => !openOnly || model.access !== 'API')
    .filter((model) => !closedOnly || model.access === 'API')
    .filter((model) => !mustSelfHost || model.access === 'Open source')
    .filter((model) => {
      if (effectiveMaxPrice === undefined) return true
      const inputPrice = model.pricing?.inputPerMillion
      return typeof inputPrice !== 'number' || inputPrice <= effectiveMaxPrice
    })
    .filter((model) => {
      if (!minContext) return true
      return (model.contextLength || 0) >= minContext
    })
    .filter((model) => {
      if (!minSpeed) return true
      return (model.metrics?.speed ?? 0) >= minSpeed
    })
    .filter((model) => {
      if (excludeProviders.size === 0) return true
      const provider = String(model.provider || '').toLowerCase()
      for (const ex of excludeProviders) {
        if (provider.includes(ex)) return false
      }
      return true
    })
    .filter((model) => {
      if (excludeModels.size === 0) return true
      const name = String(model.name || '').toLowerCase()
      const id = String(model.id || '').toLowerCase()
      for (const ex of excludeModels) {
        if (name.includes(ex) || id.includes(ex)) return false
      }
      return true
    })
    .filter((model) => {
      if (!filters.modalities?.length) return true
      const modelMods = (model.modalities || []).map(m => m.toLowerCase())
      return filters.modalities.every(required => modelMods.some(m => m.includes(required.toLowerCase())))
    })
    .filter((model) => {
      if (!filters.privacy || filters.privacy === 'cloud-ok') return true
      if (filters.privacy === 'no-training') {
        return (model.metrics?.privacy ?? 0) >= 0.5
      }
      return true
    })
    .map((model) => {
      const profileQuality = taskQuality(model, analysis.taskProfile)
      const rankingModel = { ...model, metrics: { ...model.metrics, quality: profileQuality } }
      const { score, exact } = computeFinalScore(rankingModel, analysis.targetCategory, analysis.weights, {
        qualityFlagged,
        archetype: analysis.archetype,
        isCoding: analysis.isCoding,
      })
      return {
        ...model,
        taskQuality: profileQuality,
        fit: capabilityFit(model, analysis.targetCategory),
        sourceTrust: sourceTrustFactor(model),
        score,
        scoreExact: exact,
        rawScore: score,
        reasons: buildReasons(model, analysis),
        warnings: model.linkStatus === 'catalog' ? ['Source link opens a catalog, not a model page'] : [],
      }
    })
    .sort((a, b) => b.scoreExact - a.scoreExact || (b.taskQuality ?? b.metrics?.quality ?? 0) - (a.taskQuality ?? a.metrics?.quality ?? 0))

  const frontierAnnotated = paretoFrontier(scoredModels, analysis.taskProfile)
    .sort((a, b) => b.scoreExact - a.scoreExact || (b.taskQuality ?? b.metrics?.quality ?? 0) - (a.taskQuality ?? a.metrics?.quality ?? 0))
  const diverseModels = applyFamilyAndProviderDiversity(frontierAnnotated, limit)
  const calibrated = calibrateCohortScores(diverseModels)

  const strategyTemplates = await readStrategyTemplates()
  const strategies = strategyTemplates.filter((t) => t.useCase === analysis.targetCategory)

  return {
    analysis,
    recommendations: calibrated,
    strategies,
    totalMatches: scoredModels.length,
    parserUsed,
  }
}
