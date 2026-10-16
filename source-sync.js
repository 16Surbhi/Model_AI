import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { appendSyncHistory, upsertModels } from './registry-store.js'
import { evaluateModelCapabilities, derivePrivacyScore } from './benchmarks.js'

const openRouterUrl = 'https://openrouter.ai/api/v1/models'
const huggingFaceBaseUrl = 'https://huggingface.co/api/models'

// Artificial Analysis — the real benchmark source. Free tier: 1,000 req/day,
// attribution required (https://artificialanalysis.ai). Env-gated by
// ARTIFICIAL_ANALYSIS_API_KEY; absent key → no enrichment (never crashes).
const artificialAnalysisUrl = 'https://artificialanalysis.ai/api/v2/data/llms/models'
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const aaCachePath = path.resolve(__dirname, '../data/aa-cache.json')
const AA_CACHE_TTL_MS = 12 * 60 * 60 * 1000 // 12h — respect the daily quota
let aaNoticeLogged = false

function clamp(value, min = 0.01, max = 1.0) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return min
  return Math.max(min, Math.min(max, numeric))
}

function normalizeId(prefix, value) {
  return `${prefix}-${String(value || 'model')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 120)}`
}

export function resolveModelCategory(modelOrText, modalities = []) {
  const isObject = typeof modelOrText === 'object' && modelOrText !== null
  const pipelineTag = isObject ? modelOrText.pipeline_tag : null
  const modelId = isObject ? (modelOrText.id || modelOrText.name || '') : String(modelOrText || '')
  const description = isObject ? (modelOrText.description || '') : ''
  const inputMods = isObject ? (modelOrText.architecture?.input_modalities || []) : []
  const outputMods = isObject ? (modelOrText.architecture?.output_modalities || []) : []
  const tags = isObject && Array.isArray(modelOrText.tags) ? modelOrText.tags : []

  // 1. Structured Hugging Face pipeline tag
  if (pipelineTag) {
    const pipelineMap = {
      'text-to-image': 'image',
      'image-to-image': 'image',
      'image-segmentation': 'image',
      'zero-shot-image-classification': 'image',
      'unconditional-image-generation': 'image',
      'object-detection': 'image',
      'depth-estimation': 'image',
      'mask-generation': 'image',
      'text-to-video': 'video',
      'image-to-video': 'video',
      'video-classification': 'video',
      'text-to-speech': 'voice',
      'automatic-speech-recognition': 'voice',
      'audio-to-audio': 'voice',
      'voice-activity-detection': 'voice',
      'audio-classification': 'voice',
      'text-to-audio': 'music',
      'sentence-similarity': 'document',
      'feature-extraction': 'document',
      'text-ranking': 'document',
      'document-question-answering': 'document',
      'table-question-answering': 'document',
      'fill-mask': 'document',
      'text-generation': 'general',
      'text2text-generation': 'general',
      'image-text-to-text': 'general',
      'conversational': 'general',
    }
    if (pipelineMap[pipelineTag]) return pipelineMap[pipelineTag]
  }

  // 2. Structured OpenRouter input & output modalities
  const inputModsLower = inputMods.map((m) => String(m).toLowerCase())
  const outputModsLower = outputMods.map((m) => String(m).toLowerCase())
  const combinedMods = [...inputModsLower, ...outputModsLower, ...modalities, ...tags].map((m) => String(m).toLowerCase())
  const rawText = `${modelId} ${description} ${combinedMods.join(' ')}`.toLowerCase()

  // Video models. Checked before image because a video pipeline is routinely
  // described with image vocabulary (image-to-video, I2V, a diffusers library
  // tag) and would otherwise be filed as an image generator.
  if (outputModsLower.includes('video')
    || /\b(runway|sora|veo|luma|kling|pika|wan-video|wan2|cogvideox|hunyuanvideo|animatediff|ltx|image-to-video|text-to-video|i2v|t2v)\b/i.test(rawText)
    || /(^|[/_-])(i2v|t2v|v2v)([/_-]|$)/i.test(modelId)) {
    return 'video'
  }

  // Music models
  if (/\b(music|suno|udio|audiocraft|musicgen|riffusion|text-to-audio|music-generation)\b/i.test(rawText)) {
    return 'music'
  }

  // Voice models
  if (outputModsLower.includes('audio') || /\b(whisper|elevenlabs|deepgram|tts|stt|speech-to-text|text-to-speech|voice-agent)\b/i.test(rawText)) {
    return 'voice'
  }

  // Image models
  // `diffusers` is a library tag, not a modality: it appears on video and even
  // audio pipelines, so it is deliberately NOT an image trigger on its own. An
  // image label needs an image-generation signal or an image-only output.
  if (
    (outputModsLower.includes('image') && !outputModsLower.includes('text')) ||
    /\b(flux|stable-diffusion|sdxl|midjourney|dall-?e|imagen|image-generation|image-to-image|text-to-image|text2image|sdxl-turbo)\b/i.test(rawText) ||
    (outputModsLower.includes('image') && /image|edit|gen/i.test(modelId))
  ) {
    return 'image'
  }

  // Document / Retrieval / Embedding models. The dense-retrieval families are
  // named by prefix (e5-, gte-, bge-) and by benchmark suite (MTEB); without
  // them a pure embedding model is misread as a generative assistant and enters
  // the general cohort, where it competes for chat and coding requests.
  if (/\b(embedding|embed-|rerank|sentence-similarity|sentence-transformers|colbert|colpali|ocr|bge-|e5-|gte-|mteb|nomic-embed|jina-embed|voyage-)\b/i.test(rawText)) {
    return 'document'
  }

  // 3. Default for all LLMs, reasoning, coding, multimodal chat, and general models
  return 'general'
}

export function normalizeContext(contextLength) {
  const context = Number(contextLength || 0)
  if (!Number.isFinite(context) || context <= 0) {
    return 0.08
  }

  // Log-logistic curve: 1k->0.39, 4k->0.54, 32k->0.71, 128k->0.79, 1M->0.89, 2M->0.91, 10M->0.97
  const L = Math.log10(context)
  return clamp(((L - 2.0) / (L + 1.0)) * 1.55, 0.08, 1)
}

/**
 * Map a token price onto the affordability axis.
 *
 * An UNMEASURED price returns `null`, not a high value. The previous behaviour
 * returned 0.95 for a missing or zero price, which turns absent data into a
 * positive claim of cheapness: a record with no pricing observation outranked a
 * record with a real one on the axis that had the widest spread in the cohort
 * and therefore decided the ranking. `null` is the honest answer, and callers
 * must decide what an unknown axis does to a comparison instead of inheriting a
 * fabricated advantage.
 */
export function priceToAffordability(inputPerMillion, outputPerMillion, category = 'general') {
  const ratios = {
    document: { input: 0.85, output: 0.15 },
    general: { input: 0.60, output: 0.40 },
  }
  const r = ratios[category] || ratios.general
  const blended = Number(inputPerMillion || 0) * r.input + Number(outputPerMillion || 0) * r.output

  if (!Number.isFinite(blended) || blended <= 0) {
    return null
  }

  return clamp(1 - Math.log10(blended + 1) / 2.1, 0.45, 0.98)
}

export function formatProviderName(raw) {
  if (!raw) return 'Unknown provider'
  const p = String(raw).trim().toLowerCase()
  const providerMap = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'google': 'Google',
    'deepseek': 'DeepSeek',
    'deepseek-ai': 'DeepSeek',
    'meta': 'Meta',
    'meta-llama': 'Meta',
    'mistralai': 'Mistral AI',
    'mistral': 'Mistral AI',
    'qwen': 'Qwen (Alibaba)',
    'alibaba': 'Qwen (Alibaba)',
    'cohere': 'Cohere',
    'x-ai': 'xAI',
    'xai': 'xAI',
    'perplexity': 'Perplexity',
    'stability': 'Stability AI',
    'stabilityai': 'Stability AI',
    'black-forest-labs': 'Black Forest Labs',
    'bytedance': 'ByteDance',
    'tencent': 'Tencent',
    'minimax': 'MiniMax',
    'moonshot': 'Moonshot AI',
    'z-ai': 'Zhipu AI',
    'thudm': 'THUDM',
    'elevenlabs': 'ElevenLabs',
    'deepgram': 'Deepgram',
    'suno': 'Suno',
    'runway': 'Runway',
    'luma': 'Luma AI',
    'kling': 'Kuaishou Kling',
    'openrouter': 'OpenRouter',
    'huggingface': 'Hugging Face',
    'nvidia': 'NVIDIA',
    'microsoft': 'Microsoft',
    'amazon': 'Amazon Web Services',
    'aws': 'Amazon Web Services',
  }
  if (providerMap[p]) return providerMap[p]
  return raw.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

// ── Artificial Analysis: fetch, cache, index & fuzzy join ──

function aaNorm(value) {
  // Collapse to bare alphanumerics so "claude-3.5-sonnet", "Claude 3.5 Sonnet"
  // and "claude35sonnet" all match.
  return String(value || '').toLowerCase().replace(/[^a-z0-9]/g, '')
}

// Registry identities and Artificial Analysis slugs are written by different
// producers and disagree in three systematic ways, each of which used to defeat
// the join entirely:
//
//   registry  anthropic/claude-opus-5.5-20260921   (dotted version + date)
//   AA        claude-opus-5-5                      (hyphenated version, no date)
//   registry  "Anthropic: Claude Opus 5.5"        (provider-prefixed name)
//   AA        "Claude Opus 5.5 (Max, Default Fallback)"
//
// Only 39% of the registry's stored benchmark enrichments could be reproduced
// by the previous single-key lookup, which made the registry non-reproducible
// and silently dropped benchmark coverage on every rebuild. The candidate keys
// below normalise those three differences away. Matching stays *exact* against
// the generated keys — there is deliberately no prefix/substring fallback,
// because a loose fallback claims one AA record for several different registry
// models and trades a coverage gain for wrong data.
const IDDATE_RE = /(?:[-_]?)(?:19|20)\d{6}$/
const IDVERSION_SUFFIX_RE = /(?:[-_]?)(?:v|ver)\d+(?:\.\d+)*$/
const ID_PREFIX_SEGMENTS = new Set(['openrouter', 'hf', 'huggingface', 'curated', 'seed', 'api', 'model', 'models'])

function identityVariants(rawValue) {
  const raw = String(rawValue || '').trim()
  if (!raw) return []
  const variants = new Set()

  // Dots are version separators in one source and hyphens in the other.
  const hyphenated = raw.replace(/\./g, '-')
  const base = hyphenated.replace(/\s*\([^)]*\)\s*/g, ' ').trim()

  const push = (value) => {
    let v = aaNorm(value)
    if (!v) return
    variants.add(v)
    const noDate = v.replace(/\d{8}$/, '')
    if (noDate) variants.add(noDate)
  }

  push(base)
  push(base.replace(/\./g, '-'))

  // Drop leading provider / aggregator segments: "openrouter-anthropic-claude…"
  // and "anthropic/claude…" both reduce to "claude…".
  const segments = hyphenated.split(/[/\s]+|-/).filter(Boolean)
  for (let i = 1; i <= Math.min(3, segments.length - 1); i++) {
    const head = segments[i - 1].toLowerCase()
    if (!ID_PREFIX_SEGMENTS.has(head) && i > 1 && !ID_PREFIX_SEGMENTS.has(segments[0].toLowerCase())) break
    push(segments.slice(i).join('-'))
  }

  // Drop a trailing release date / version marker in either form.
  for (const v of [...variants]) {
    const stripped = aaNorm(String(v).replace(IDDATE_RE, '').replace(IDVERSION_SUFFIX_RE, ''))
    if (stripped && stripped !== v) variants.add(stripped)
  }

  return [...variants].filter((v) => v.length >= 4)
}

/** Candidate identity keys for a registry model, strongest first. */
function aaCandidateKeys(model) {
  const rawId = String(model.canonicalId || model.id || '')
  const provider = aaNorm(model.provider || '')
  const keys = []
  const seen = new Set()
  const add = (key, qualified) => {
    if (!key || seen.has(key)) return
    seen.add(key)
    keys.push({ key, qualified })
  }

  // A model's display name is the most reliable identity source: it is written
  // by hand or by the aggregator and carries no route decoration.
  const display = String(model.name || '').replace(/^[^:]{1,30}:\s*/, '')
  if (display) {
    for (const v of identityVariants(display)) add(provider ? `${provider}:${v}` : null, true)
    for (const v of identityVariants(display)) add(v, false)
  }
  for (const v of identityVariants(rawId)) add(v, false)
  for (const v of identityVariants(rawId)) add(provider ? `${provider}:${v}` : null, true)

  return keys
}

function avgFinite(values) {
  const nums = values.map(Number).filter((v) => Number.isFinite(v) && v > 0)
  if (!nums.length) return null
  return nums.reduce((sum, v) => sum + v, 0) / nums.length
}

/**
 * Extract a normalized observation block from one Artificial Analysis record.
 * Agentic index is synthesized from AA's agentic-relevant evaluations
 * (τ-bench, terminal-bench) since the free tier has no single agentic index.
 */
export function aaObservationsFrom(aaRecord) {
  const evals = aaRecord?.evaluations || {}
  const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : null)

  const agentic = avgFinite([
    evals.tau2,
    evals.tau_banking,
    evals.terminalbench_v2_1,
    evals.terminalbench_hard,
  ])

  return {
    intelligence_index: num(evals.artificial_analysis_intelligence_index),
    coding_index: num(evals.artificial_analysis_coding_index),
    math_index: num(evals.artificial_analysis_math_index),
    agentic_index: agentic !== null ? Math.round(agentic * 10) / 10 : null,
    tokens_per_second: num(aaRecord?.median_output_tokens_per_second),
    ttft_seconds: num(aaRecord?.median_time_to_first_token_seconds),
    source: 'Artificial Analysis',
    observedAt: new Date().toISOString(),
  }
}

/**
 * Fetch the Artificial Analysis model list, with a disk cache to respect the
 * daily quota. Returns an array of AA records, or null when no key is set.
 */
export async function fetchArtificialAnalysisModels({ force = false } = {}) {
  const apiKey = process.env.ARTIFICIAL_ANALYSIS_API_KEY
  if (!apiKey) {
    if (!aaNoticeLogged) {
      console.log('   ℹ️ ARTIFICIAL_ANALYSIS_API_KEY not set — skipping benchmark enrichment (structural scoring only).')
      aaNoticeLogged = true
    }
    return null
  }

  if (!force) {
    try {
      const cached = JSON.parse(await readFile(aaCachePath, 'utf8'))
      if (cached && Array.isArray(cached.data) && Date.now() - Number(cached.fetchedAt || 0) < AA_CACHE_TTL_MS) {
        return cached.data
      }
    } catch {
      // no / stale cache — fall through to network
    }
  }

  try {
    const response = await fetch(artificialAnalysisUrl, {
      headers: { accept: 'application/json', 'x-api-key': apiKey },
    })
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`)
    }
    const payload = await response.json()
    const data = Array.isArray(payload?.data) ? payload.data : []
    await writeFile(aaCachePath, `${JSON.stringify({ fetchedAt: Date.now(), count: data.length, data }, null, 2)}\n`, 'utf8').catch(() => {})
    return data
  } catch (error) {
    // On failure, degrade to any stale cache before giving up.
    try {
      const cached = JSON.parse(await readFile(aaCachePath, 'utf8'))
      if (cached && Array.isArray(cached.data)) {
        console.warn(`   ⚠️ Artificial Analysis fetch failed (${error.message}); using cached snapshot.`)
        return cached.data
      }
    } catch {}
    console.warn(`   ⚠️ Artificial Analysis fetch failed (${error.message}); proceeding without benchmark enrichment.`)
    return null
  }
}

/**
 * Build a lookup index over AA records keyed by normalized identity. Each
 * record is registered under several keys (slug, name, creator-qualified) so an
 * OpenRouter/HF row can be matched from whatever identity it exposes.
 */
export function buildAaIndex(aaModels) {
  if (!Array.isArray(aaModels) || !aaModels.length) return null
  const byKey = new Map()

  const register = (key, record) => {
    if (!key) return
    if (!byKey.has(key)) byKey.set(key, record)
  }

  for (const record of aaModels) {
    const creator = aaNorm(record?.model_creator?.slug || record?.model_creator?.name)
    const slugKey = aaNorm(record?.slug)
    const nameKey = aaNorm(record?.name)
    register(slugKey, record)
    register(nameKey, record)
    if (creator) {
      register(`${creator}:${slugKey}`, record)
      register(`${creator}:${nameKey}`, record)
    }
  }

  return { byKey }
}

/**
 * Find the AA record best matching an OpenRouter/HF model. Matching is fuzzy by
 * design (naming differs across sources); partial coverage is acceptable.
 */
export function matchAaRecord(model, aaIndex) {
  if (!aaIndex || !aaIndex.byKey) return null
  const byKey = aaIndex.byKey

  // Provider-qualified keys first (they cannot collide across vendors), then
  // bare identity keys. All lookups are exact set membership.
  const keys = aaCandidateKeys(model)
  for (const { key, qualified } of keys) {
    if (qualified && byKey.has(key)) return byKey.get(key)
  }
  for (const { key } of keys) {
    if (byKey.has(key)) return byKey.get(key)
  }
  return null
}

/**
 * Assign each Artificial Analysis record to at most ONE registry model.
 *
 * Re-running the join per model independently can hand the same benchmark
 * record to several registry rows, which duplicates one model's measured
 * capability onto its quantisations, aliases and re-listings. This resolves the
 * competition deterministically: a provider-qualified match outranks a bare
 * one, and a more specific identity (longer matched key) outranks a shorter one.
 *
 * @returns {{ matches: Map<string, object>, conflicts: Array<{aaId: string, winner: string, losers: string[]}> }}
 */
export function resolveAaMatches(models = [], aaIndex = null) {
  if (!aaIndex || !aaIndex.byKey) return { matches: new Map(), conflicts: [] }
  const byKey = aaIndex.byKey

  const claims = []
  for (const model of models) {
    const keys = aaCandidateKeys(model)
    let record = null
    let matchedKey = ''
    let qualified = false
    for (const candidate of keys) {
      if (candidate.qualified && byKey.has(candidate.key)) {
        record = byKey.get(candidate.key); matchedKey = candidate.key; qualified = true; break
      }
    }
    if (!record) {
      for (const candidate of keys) {
        if (byKey.has(candidate.key)) { record = byKey.get(candidate.key); matchedKey = candidate.key; break }
      }
    }
    if (!record) continue
    claims.push({
      modelId: model.id,
      record,
      aaId: record.id || matchedKey,
      rank: (qualified ? 2 : 1) * 1000 + matchedKey.length,
    })
  }

  claims.sort((a, b) => b.rank - a.rank)
  const matches = new Map()
  const owner = new Map()
  const conflicts = []
  const losers = new Map()
  for (const claim of claims) {
    if (owner.has(claim.aaId)) {
      const winner = owner.get(claim.aaId)
      const list = losers.get(claim.aaId) || []
      list.push(claim.modelId)
      losers.set(claim.aaId, list)
      continue
    }
    owner.set(claim.aaId, claim.modelId)
    matches.set(claim.modelId, claim.record)
  }
  for (const [aaId, list] of losers) {
    conflicts.push({ aaId, winner: owner.get(aaId), losers: list })
  }
  return { matches, conflicts }
}

/**
 * Reconcile a stored category against the current classifier.
 *
 * `resolveModelCategory` runs at ingest, so it only ever classified the record
 * it was first applied to. Records ingested before a classifier improvement keep
 * their old label forever, which is how a whisper checkpoint stayed in the
 * general cohort. This re-derives the category from the record's own signals and
 * returns the label plus the primary/secondary use-case vectors that must follow
 * it.
 *
 * A specific derived category always wins (it is evidence of a media pipeline).
 * A derived `general` does not silently erase a stored specialisation: the
 * stored value is kept only when the record's own primary use case still
 * supports it.
 */
export function reconcileCategory(model) {
  const derived = resolveModelCategory(model)
  const stored = String(model.category || 'general')
  const primary = Array.isArray(model.primaryUseCases) ? model.primaryUseCases : []

  let category
  if (derived !== 'general') category = derived
  else if (stored !== 'general' && primary.includes(stored)) category = stored
  else category = 'general'

  const secondary = (Array.isArray(model.secondaryUseCases) ? model.secondaryUseCases : [])
    .filter((useCase) => useCase !== category)
  const primaryUseCases = category === 'general' && primary.includes('general')
    ? primary
    : [category, ...primary.filter((useCase) => useCase !== stored && useCase !== category)]

  return {
    changed: category !== stored,
    from: stored,
    to: category,
    category,
    primaryUseCases: [...new Set(primaryUseCases)],
    secondaryUseCases: [...new Set(secondary)].filter((useCase) => !primaryUseCases.includes(useCase)),
  }
}

/** Convenience: fetch AA and build the join index in one call (null if no key). */
export async function buildAaIndexFromApi(options = {}) {
  const models = await fetchArtificialAnalysisModels(options)
  return models ? buildAaIndex(models) : null
}

function metricSource(value, source, confidence, observedAt) {
  return { value, source, confidence, observedAt, method: source === 'heuristic' ? 'estimate' : 'observed' }
}

export function deriveSecondaryUseCases(model, primaryCategory) {
  const text = `${model.id || ''} ${model.name || ''} ${model.description || ''} ${(model.tags || []).join(' ')}`.toLowerCase()
  const useCases = new Set()

  if (primaryCategory === 'general') {
    if (/\b(code|coding|coder|python|sql|javascript|typescript|c\+\+|rust|developer|programming|swe-bench|humaneval)\b/i.test(text)) {
      useCases.add('code')
    }
    if (/\b(agent|tool|function|tool_use|call|calling|reasoning|cot|r1|o1|o3|o4|thinking)\b/i.test(text)) {
      useCases.add('agent')
    }
    if (/\b(rag|embed|retriev|search|vector|document|pdf)\b/i.test(text)) {
      useCases.add('rag')
    }
    if (/\b(vision|image|multimodal|vl|ocr)\b/i.test(text)) {
      useCases.add('vision')
    }
  }

  return [...useCases]
}

function openRouterBestFor(category, name, description = '') {
  const label = name.replace(/\s+/g, ' ').trim()

  if (description && description.length > 20) {
    return description.slice(0, 260) + (description.length > 260 ? '...' : '')
  }

  if (category === 'document') {
    return `${label} is designed for long-context document analysis, extraction, embeddings, and RAG workflows.`
  }
  if (category === 'image') {
    return `${label} supports visual tasks such as image generation, editing, and synthesis.`
  }
  if (category === 'video') {
    return `${label} supports video generation, motion synthesis, and cinematic workflows.`
  }
  if (category === 'voice') {
    return `${label} is designed for realtime speech, transcription, and conversational voice agents.`
  }
  if (category === 'music') {
    return `${label} generates music tracks, soundscapes, and instrumental audio.`
  }

  return `${label} is designed for reasoning, coding, conversational agents, and multimodal workflows.`
}

function mapOpenRouterModel(model, aaIndex = null) {
  const modalities = [
    ...(model.architecture?.input_modalities || []),
    ...(model.architecture?.output_modalities || []),
  ]
  const category = resolveModelCategory(model, modalities)
  const contextScore = normalizeContext(model.context_length)
  const inputPerMillion = Number(model.pricing?.prompt || 0) * 1000000
  const outputPerMillion = Number(model.pricing?.completion || 0) * 1000000
  const affordability = priceToAffordability(inputPerMillion, outputPerMillion, category)
  const rawProvider = String(model.id || '').split('/')[0] || 'OpenRouter'
  const provider = formatProviderName(rawProvider)
  const observedAt = new Date().toISOString()

  // Join real Artificial Analysis benchmarks onto the catalog row (if matched).
  const aaRecord = matchAaRecord(model, aaIndex)
  const benchmarkObservations = aaRecord
    ? { design_arena: [], artificial_analysis: aaObservationsFrom(aaRecord) }
    : (model.benchmarks && typeof model.benchmarks === 'object' ? model.benchmarks : {})
  const hasBenchmarks = Boolean(aaRecord)

  // Evaluate capability dynamically — benchmark-first when AA data is present.
  const caps = evaluateModelCapabilities({
    ...model,
    benchmarkObservations,
    pricing: { inputPerMillion, outputPerMillion },
    access: 'API',
  })
  const observedTps = Number(benchmarkObservations?.artificial_analysis?.tokens_per_second) > 0
  const speedSource = observedTps
    ? metricSource(caps.speed, 'Artificial Analysis median output tokens/sec', 0.9, observedAt)
    : metricSource(caps.speed, 'heuristic', 0.25, observedAt)
  const qualitySource = hasBenchmarks
    ? metricSource(caps.quality, 'Artificial Analysis intelligence/coding index', 0.9, observedAt)
    : metricSource(caps.quality, 'heuristic', 0.35, observedAt)

  const secondaryUseCases = deriveSecondaryUseCases(model, category)
  if (caps.isCodeSpecialist && !secondaryUseCases.includes('code')) {
    secondaryUseCases.push('code')
  }
  if (caps.hasTools && !secondaryUseCases.includes('agent')) {
    secondaryUseCases.push('agent')
  }

  return {
    id: normalizeId('openrouter', model.id || model.name),
    name: model.name || model.id,
    provider,
    category,
    recordType: 'api_model',
    sourceAuthority: hasBenchmarks ? 'benchmark' : 'aggregator',
    linkStatus: model.id ? 'verified' : 'catalog',
    primaryUseCases: [category],
    secondaryUseCases,
    access: 'API',
    modalities: modalities.length ? [...new Set(modalities)] : ['text'],
    bestFor: openRouterBestFor(category, model.name || model.id, model.description),
    source: 'OpenRouter Live',
    sourceUrl: model.id ? `https://openrouter.ai/${model.id}` : openRouterUrl,
    benchmarkSources: hasBenchmarks ? ['Artificial Analysis'] : [],
    created: model.created,
    supportedParameters: model.supported_parameters || [],
    reasoning: model.reasoning || null,
    architecture: model.architecture || null,
    benchmarkObservations,
    canonicalId: model.canonical_slug || model.id,
    aliasOf: model.alias_target?.slug || null,
    inputModalities: model.architecture?.input_modalities || [],
    outputModalities: model.architecture?.output_modalities || [],
    lastVerified: observedAt,
    confidence: hasBenchmarks ? 0.92 : 0.70,
    pricing: {
      unit: 'USD per 1M tokens',
      inputPerMillion,
      outputPerMillion,
    },
    contextLength: Number(model.context_length || 0),
    metrics: {
      quality: caps.quality,
      affordability,
      speed: caps.speed,
      context: contextScore,
      privacy: derivePrivacyScore({ access: 'API', provider, id: model.id }),
      availability: 0.90,
    },
    metricSources: {
      quality: qualitySource,
      affordability: metricSource(affordability, 'OpenRouter pricing', 0.98, observedAt),
      context: metricSource(contextScore, 'OpenRouter context_length', 0.98, observedAt),
      speed: speedSource,
      privacy: metricSource(0.45, 'deployment heuristic', 0.4, observedAt),
      availability: metricSource(0.90, 'OpenRouter catalog presence', 0.55, observedAt),
    },
  }
}

export function isValidHuggingFaceModel(model) {
  if (!model || typeof model !== 'object') return false
  const id = String(model.id || model.modelId || '').toLowerCase()
  if (!id || !id.includes('/')) return false

  // Filter out internal test repos, checkpoints, dummy placeholders, evaluation results
  if (/(?:^|\/)(?:trl-internal-testing|dummy|test-|sample-|checkpoint-|temp-|placeholder|demo-only|eval-results|archive-only|docs-)/i.test(id)) {
    return false
  }

  const validLibraries = new Set([
    'transformers', 'diffusers', 'sentence-transformers', 'timm', 'gguf',
    'mlx', 'vllm', 'safetensors', 'onnx', 'openvino', 'fairseq', 'spacy',
    'nemo', 'peft', 'ctranslate2', 'espnet', 'speechbrain', 'unidepth', 'chronos-forecasting'
  ])
  const library = String(model.library_name || '').toLowerCase()
  const tags = Array.isArray(model.tags) ? model.tags.map((t) => String(t).toLowerCase()) : []
  const hasValidLibrary = validLibraries.has(library) || tags.some((t) => validLibraries.has(t))
  const hasValidPipeline = Boolean(model.pipeline_tag)

  if (!hasValidLibrary && !hasValidPipeline) {
    return false
  }

  if (tags.includes('dataset') && !hasValidLibrary) return false

  const downloads = Number(model.downloads || 0)
  const likes = Number(model.likes || 0)
  if (downloads < 20 && likes < 5) {
    return false
  }

  return true
}

function estimateHuggingFaceAvailability(model) {
  const downloads = Number(model.downloads || 0)
  const likes = Number(model.likes || 0)
  return clamp(0.50 + Math.log10(downloads + likes * 10 + 1) / 9, 0.60, 0.98)
}

function mapHuggingFaceModel(model, aaIndex = null) {
  const tags = Array.isArray(model.tags) ? model.tags : []
  const category = resolveModelCategory(model, tags)
  const rawProvider = String(model.id || '').split('/')[0] || 'Hugging Face'
  const provider = formatProviderName(rawProvider)
  const contextMatch = tags.join(' ').match(/(?:context|long-context)[-_ ]?(\d+(?:\.\d+)?)\s*(k|m)?/i)
  const contextLength = contextMatch ? Math.round(Number(contextMatch[1]) * (String(contextMatch[2]).toLowerCase() === 'm' ? 1_000_000 : 1_000)) : 0
  const observedAt = new Date().toISOString()

  const access = model.gated ? 'Hosted open model' : 'Open source'

  const aaRecord = matchAaRecord(model, aaIndex)
  const benchmarkObservations = aaRecord
    ? { design_arena: [], artificial_analysis: aaObservationsFrom(aaRecord) }
    : {}
  const hasBenchmarks = Boolean(aaRecord)

  const caps = evaluateModelCapabilities({
    ...model,
    benchmarkObservations,
    access,
  })

  const secondaryUseCases = deriveSecondaryUseCases(model, category)
  if (caps.isCodeSpecialist && !secondaryUseCases.includes('code')) {
    secondaryUseCases.push('code')
  }

  return {
    id: normalizeId('hf', model.id),
    name: model.id || model.modelId,
    provider,
    category,
    recordType: model.gated ? 'hosted_open_model' : 'open_weight_model',
    sourceAuthority: hasBenchmarks ? 'benchmark' : 'aggregator',
    linkStatus: 'verified',
    primaryUseCases: [category],
    secondaryUseCases,
    access,
    modalities: tags.slice(0, 8).map(String),
    bestFor: `${model.id || 'This open model'} can be evaluated for ${category} workflows, self-hosting, or private deployments.`,
    source: 'Hugging Face Hub',
    sourceUrl: `https://huggingface.co/${model.id}`,
    benchmarkSources: hasBenchmarks ? ['Artificial Analysis'] : [],
    benchmarkObservations,
    downloads: Number(model.downloads || 0),
    likes: Number(model.likes || 0),
    canonicalId: model.base_model || model.id,
    license: model.cardData?.license || tags.find((tag) => String(tag).startsWith('license:'))?.replace('license:', '') || null,
    lastVerified: observedAt,
    confidence: hasBenchmarks ? 0.85 : 0.55,
    pricing: {
      unit: 'infrastructure',
      inputPerMillion: 0,
      outputPerMillion: 0,
    },
    contextLength,
    metrics: {
      quality: caps.quality,
      affordability: 0.95,
      speed: caps.speed,
      context: contextLength ? normalizeContext(contextLength) : 0.08,
      privacy: caps.privacy,
      availability: estimateHuggingFaceAvailability(model),
    },
    metricSources: {
      quality: metricSource(caps.quality, hasBenchmarks ? 'Artificial Analysis intelligence/coding index' : 'heuristic', hasBenchmarks ? 0.9 : 0.35, observedAt),
      affordability: metricSource(0.95, 'unknown self-hosting infrastructure cost', 0.15, observedAt),
      context: metricSource(contextLength ? normalizeContext(contextLength) : 0.08, contextLength ? 'Hugging Face tag' : 'unknown', contextLength ? 0.45 : 0.05, observedAt),
      speed: metricSource(caps.speed, Number(benchmarkObservations?.artificial_analysis?.tokens_per_second) > 0 ? 'Artificial Analysis median output tokens/sec' : 'heuristic', Number(benchmarkObservations?.artificial_analysis?.tokens_per_second) > 0 ? 0.9 : 0.25, observedAt),
      privacy: metricSource(caps.privacy, 'deployment capability', 0.7, observedAt),
      availability: metricSource(estimateHuggingFaceAvailability(model), 'Hub popularity proxy', 0.3, observedAt),
    },
  }
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      accept: 'application/json',
      'user-agent': 'model-compass-live-sync/2.0',
    },
  })

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`)
  }

  return response.json()
}

export async function fetchOpenRouterModels(limit = 400, aaIndex = null) {
  const payload = await fetchJson(openRouterUrl)
  const rawList = payload.data || []

  // Filter out redundant batch and test endpoints
  const filtered = rawList.filter((m) => {
    const id = String(m.id || '').toLowerCase()
    if (id.endsWith(':batch') || m.alias_target) return false
    return true
  })
  const canonical = new Map()
  for (const model of filtered) {
    const key = model.canonical_slug || model.id
    if (!canonical.has(key)) canonical.set(key, model)
  }
  return [...canonical.values()].slice(0, limit).map((model) => mapOpenRouterModel(model, aaIndex))
}

const huggingFaceTasks = [
  'text-generation',
  'text2text-generation',
  'image-text-to-text',
  'text-to-image',
  'image-to-video',
  'automatic-speech-recognition',
  'text-to-speech',
  'text-to-audio',
  'sentence-similarity',
  'feature-extraction',
]

export async function fetchHuggingFaceModels(limit = 80, aaIndex = null) {
  const perTaskLimit = Math.max(4, Math.ceil(limit / huggingFaceTasks.length))
  const allModels = []

  for (const task of huggingFaceTasks) {
    const params = new URLSearchParams({
      filter: task,
      sort: 'downloads',
      direction: '-1',
      limit: String(perTaskLimit * 2),
    })

    const payload = await fetchJson(`${huggingFaceBaseUrl}?${params.toString()}`)
    const validPayload = Array.isArray(payload) ? payload.filter(isValidHuggingFaceModel) : []
    allModels.push(...validPayload.map((model) => mapHuggingFaceModel(model, aaIndex)))
  }

  const deduped = new Map()
  for (const model of allModels) {
    deduped.set(model.id, model)
  }

  return [...deduped.values()].slice(0, limit)
}

export async function syncSources(options = {}) {
  const sources = Array.isArray(options.sources) && options.sources.length
    ? options.sources
    : ['openrouter', 'huggingface']
  const limit = Number.isFinite(Number(options.limit)) ? Number(options.limit) : 400
  const results = []
  const models = []

  // Fetch the real benchmark index once, then join it onto every catalog row.
  const aaIndex = await buildAaIndexFromApi()
  const enriched = Boolean(aaIndex)

  for (const source of sources) {
    try {
      const fetched =
        source === 'huggingface'
          ? await fetchHuggingFaceModels(Math.min(limit, 80), aaIndex)
          : await fetchOpenRouterModels(limit, aaIndex)

      models.push(...fetched)
      results.push({
        source,
        ok: true,
        count: fetched.length,
      })
    } catch (error) {
      results.push({
        source,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }

  const matched = models.filter((m) => (m.benchmarkSources || []).includes('Artificial Analysis')).length
  const writeResult = models.length
    ? await upsertModels(models, { reason: 'live dynamic source sync', sources, benchmarkSource: enriched ? 'Artificial Analysis' : null, benchmarkMatched: matched })
    : undefined

  await appendSyncHistory({
    sources,
    results,
    added: writeResult?.added || 0,
    updated: writeResult?.updated || 0,
    modelCount: writeResult?.after,
    benchmarkEnriched: enriched,
    benchmarkMatched: matched,
  })

  return {
    ok: results.some((result) => result.ok),
    results,
    added: writeResult?.added || 0,
    updated: writeResult?.updated || 0,
    modelCount: writeResult?.after,
    benchmarkEnriched: enriched,
    benchmarkMatched: matched,
  }
}
