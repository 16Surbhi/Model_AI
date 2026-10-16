/**
 * LLM-based prompt understanding module
 *
 * Uses Cohere (COHERE_API_KEY) for structured intent extraction.
 * Falls back to regex when no API key is configured or the call fails.
 */

const COHERE_URL = 'https://api.cohere.ai/v2/chat'
const DEFAULT_COHERE_MODEL = 'command-r7b-12-2024'

const SYSTEM_PROMPT = `You are a structured data extraction assistant for an AI model recommendation engine.
Given a user's natural language description of what they need an AI model for, extract the following:

1. **category**: One of: general, image, video, voice, music, document (Note: coding and reasoning tasks map to 'general')
2. **weights**: Relative importance of each dimension, each an integer 0-10, where 0 means "the
   user expressed no interest in this dimension at all" and 10 means "this is the dominant
   requirement". Ratios matter, not absolute size. Set a dimension to 0 when the user never
   mentions it. A priority LIST cannot express "nothing dominates"; these relative weights can,
   so use them whenever the user compares dimensions rather than ranking one above the others.
   Do not weight throughput above quality/affordability unless the user asked for speed or latency.
3. **priorities**: Array of metric priorities from: quality, affordability, speed, context, privacy, availability
4. **constraints**: Object with optional fields:
   - maxPrice: number (max USD per million tokens)
   - minContext: number (minimum context window in tokens)
   - mustSelfHost: boolean (requires self-hosting)
4. **excludeProviders**: Array of provider names to exclude (e.g., ["openai"])
5. **excludeModels**: Array of specific model names to exclude
6. **negations**: Array of metrics the user explicitly said they DON'T care about
8. **useCaseSummary**: One-sentence summary of the use case

Respond with ONLY valid JSON matching this exact schema:
{
  "category": "string",
  "weights": { "quality": 0, "affordability": 0, "speed": 0, "context": 0, "privacy": 0, "availability": 0 },
  "priorities": ["string"],
  "constraints": { "maxPrice": null, "minContext": null, "mustSelfHost": false },
  "excludeProviders": [],
  "excludeModels": [],
  "negations": [],
  "useCaseSummary": "string"
}`

const VALID_CATEGORIES = ['general', 'image', 'video', 'voice', 'music', 'document']
const VALID_METRICS = ['quality', 'affordability', 'speed', 'context', 'privacy', 'availability']

/**
 * Parse LLM response JSON, with lenient extraction.
 */
function parseResponseJson(text, prompt = '') {
  const jsonMatch = text.match(/\{[\s\S]*\}/)
  if (!jsonMatch) return null

  try {
    const parsed = JSON.parse(jsonMatch[0])
    return sanitizeParsedIntent(parsed, prompt)
  } catch {
    return null
  }
}

// ── Grounding checks ────────────────────────────────────────────────────────
// A language model will happily invent a constraint that the user never wrote
// ("best model for product design code" was returned with maxPrice=100 and
// minContext=100, neither of which appears in the prompt). An invented
// constraint silently removes qualifying models from the candidate set, which
// is a silent wrong answer rather than a visible failure. Every extracted
// constraint must therefore be traceable to a token in the user's own text; a
// constraint that cannot be traced is dropped rather than trusted.
function promptNumbers(text) {
  const found = new Set()
  const matches = String(text).match(/\d+(?:\.\d+)?/g) || []
  for (const m of matches) found.add(Number(m))
  return found
}

function promptMentionsNumber(prompt, value) {
  if (!Number.isFinite(value)) return false
  const numbers = promptNumbers(prompt)
  if (numbers.has(value)) return true
  // Accept the "k" shorthand: 128000 in the field matches "128k" in the prompt.
  for (const n of numbers) {
    if (n >= 1 && n < 1000 && Number((n * 1000).toFixed(6)) === Number(value.toFixed(6))) return true
  }
  return false
}

// ── Dimension grounding ─────────────────────────────────────────────────────
// A weight is a claim that the USER cares about a dimension. The model will
// happily supply one for every dimension it can name -- asked for "balanced
// coding models" it returns all six at 4-8 out of 10 -- but the prompt contains
// no word about latency, context, privacy or hosting. Those invented interests
// are not harmless: each weight multiplies a sub-score whose spread across the
// catalogue differs by an order of magnitude, so a moderate piece of fiction on
// a wide-spread dimension silently outvotes the dimension the user did name.
// Measured on the paper cohort, the invented speed weight flipped every budgeted
// tier from the best-outcome model to a flash model worth 12-13 net points less.
// A weight therefore has to be traceable to the user's own text, exactly like a
// constraint or an exclusion.
const DIMENSION_MENTION = {
  quality: /\b(best|top|sota|state of the art|frontier|highest quality|smartest|most capable|most powerful|accurate|accuracy|reliable|reasoning|quality|flagship|gold standard|complex)\b/i,
  affordability: /\b(cheap|cheaper|cheapest|budget|affordable|affordability|cost|costs|price|pricing|free|inexpensive|frugal|low[- ]cost|per million|spend|spending|dollars|usd|\$)\b/i,
  speed: /\b(fast|faster|fastest|speed|quick|quickly|latency|low[- ]latency|real[- ]?time|instant|throughput|responsive|snappy|tokens per second|tps)\b/i,
  context: /\b(context|context window|long[- ]context|window|pdf|document|documents|repo|repository|codebase|book|manual|thesis|contract|many files|input length)\b/i,
  privacy: /\b(private|privacy|confidential|self[- ]?host(?:ed|ing|able)?|on[- ]?prem(?:ise|ises)?|air[- ]?gap(?:ped)?|local(?:ly)?|offline|secure|security|hipaa|gdpr|soc ?2|data residency)\b/i,
  availability: /\b(availab(?:le|ility)|uptime|sla|always[- ]on|highly available|production|hosted|rate limits?|open[- ]?source|open weights)\b/i,
}

function promptMentionsDimension(prompt, dimension) {
  const pattern = DIMENSION_MENTION[dimension]
  return pattern ? pattern.test(String(prompt || '')) : false
}

function promptMentionsTerm(prompt, term) {
  const t = String(term || '').trim().toLowerCase()
  if (!t) return false
  const haystack = String(prompt).toLowerCase()
  if (haystack.includes(t)) return true
  // Provider ids arrive with vendor punctuation ("meta-llama" vs "meta llama").
  const loose = t.replace(/[^a-z0-9]+/g, ' ').trim()
  return loose.length > 2 && haystack.replace(/[^a-z0-9]+/g, ' ').includes(loose)
}

/**
 * Validate and sanitize parsed intent to ensure type safety and grounding.
 * Exported so the grounding rules can be tested without a network call.
 */
export function sanitizeParsedIntent(raw, prompt = '') {
  const category = VALID_CATEGORIES.includes(raw.category) ? raw.category : 'general'
  const priorities = Array.isArray(raw.priorities)
    ? raw.priorities.filter(p => VALID_METRICS.includes(p))
    : []
  const negations = Array.isArray(raw.negations)
    ? raw.negations.filter(n => VALID_METRICS.includes(n))
    : []

  // Relative-importance weights. These carry the information a priority list
  // cannot: how the dimensions compare to EACH OTHER, and whether the user
  // expressed any preference at all (all zeros / all equal). Only dimensions the
  // model actually weighted are kept, so an absent dimension stays absent rather
  // than inheriting a hard-coded default.
  const weights = {}
  const weightsRaw = {}
  const unstatedDimensions = []
  if (raw.weights && typeof raw.weights === 'object') {
    for (const key of VALID_METRICS) {
      const v = Number(raw.weights[key])
      if (!Number.isFinite(v) || v <= 0) continue
      weightsRaw[key] = Math.min(10, v)
      // Grounded: an unmentioned dimension is not a preference the user stated.
      if (!promptMentionsDimension(prompt, key)) {
        unstatedDimensions.push(key)
        continue
      }
      weights[key] = Math.min(10, v)
    }
  }
  const weighted = Object.keys(weights).length
  const weightValues = Object.values(weights)
  const flat = weighted > 1 && weightValues.every((v) => v === weightValues[0])
  // A priority list is the same kind of claim as a weight, so it is grounded the
  // same way: naming a priority the prompt never raised would otherwise survive
  // through the legacy priority path.
  const groundedPriorities = priorities.filter(p => promptMentionsDimension(prompt, p))
  // An exclusion removes candidates, so it must name something the user wrote.
  const excludeProviders = Array.isArray(raw.excludeProviders)
    ? raw.excludeProviders.map(String).filter((p) => promptMentionsTerm(prompt, p)).slice(0, 10)
    : []
  const excludeModels = Array.isArray(raw.excludeModels)
    ? raw.excludeModels.map(String).filter((m) => promptMentionsTerm(prompt, m)).slice(0, 10)
    : []

  const constraints = {}
  if (raw.constraints && typeof raw.constraints === 'object') {
    if (typeof raw.constraints.maxPrice === 'number' && raw.constraints.maxPrice > 0
        && promptMentionsNumber(prompt, raw.constraints.maxPrice)) {
      constraints.maxPrice = raw.constraints.maxPrice
    }
    // A context minimum below 1k tokens is not a real requirement; reject it
    // rather than filtering the catalogue down to nothing.
    if (typeof raw.constraints.minContext === 'number' && raw.constraints.minContext >= 1000
        && promptMentionsNumber(prompt, raw.constraints.minContext)) {
      constraints.minContext = raw.constraints.minContext
    }
    if (raw.constraints.mustSelfHost === true
        && /\b(self[- ]?host(?:ed|ing|able|s)?|on[- ]?prem(?:ise|ises)?|air[- ]?gap(?:ped)?|locally|local|offline|private deployment)\b/i.test(prompt)) {
      constraints.mustSelfHost = true
    }
  }

  return {
    category,
    priorities: groundedPriorities,
    weights,
    weightsRaw,
    unstatedDimensions,
    // True when the model named several dimensions at identical importance.
    // That is the model saying "no preference", not a set of equal priorities,
    // and the caller must not silently convert it into the unequal hard-coded
    // defaults' array (which is what produced a speed-and-context-heavy profile
    // for a request that stated no preference).
    weightsFlat: flat,
    constraints,
    excludeProviders,
    excludeModels,
    negations,
    useCaseSummary: typeof raw.useCaseSummary === 'string' ? raw.useCaseSummary.slice(0, 200) : '',
  }
}

function extractCohereText(data) {
  const content = data.message?.content
  if (Array.isArray(content)) {
    const textPart = content.find(part => part?.type === 'text' && typeof part.text === 'string')
    if (textPart) return textPart.text
  }

  if (typeof data.text === 'string') return data.text
  return null
}

/**
 * Call Cohere Chat API for structured intent extraction.
 */
async function callCohere(prompt, apiKey) {
  const model = process.env.COHERE_MODEL || DEFAULT_COHERE_MODEL

  const response = await fetch(COHERE_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        {
          role: 'user',
          content: `Generate a JSON object for this user prompt: "${prompt}"`,
        },
      ],
      temperature: 0.1,
      max_tokens: 500,
      response_format: { type: 'json_object' },
    }),
  })

  if (!response.ok) {
    const err = await response.text().catch(() => '')
    throw new Error(`Cohere API error ${response.status}: ${err.slice(0, 200)}`)
  }

  const data = await response.json()
  const text = extractCohereText(data)
  if (!text) throw new Error('Empty Cohere response')

  return parseResponseJson(text, prompt)
}

/**
 * Parse a user prompt using Cohere for structured intent extraction.
 *
 * @param {string} prompt - The user's natural language prompt
 * @returns {Promise<object|null>} ParsedIntent or null if Cohere is unavailable
 */
export async function parsePromptWithLLM(prompt) {
  const cohereKey = process.env.COHERE_API_KEY

  if (!cohereKey) {
    return null
  }

  try {
    const result = await callCohere(prompt, cohereKey)
    if (result) {
      return { ...result, parserUsed: 'cohere' }
    }
  } catch (err) {
    console.warn('[llm-parser] Cohere failed:', err.message)
  }

  return null
}

/**
 * Check if the Cohere API key is configured.
 */
export function isLLMAvailable() {
  return !!process.env.COHERE_API_KEY
}
