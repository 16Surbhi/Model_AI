import { mkdir, readFile, writeFile, rm } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const serverRoot = path.resolve(__dirname, '..')
const dataDir = path.join(serverRoot, 'data')
const seedPath = path.join(dataDir, 'model-registry.seed.json')
const registryPath = path.join(dataDir, 'model-registry.json')
const historyPath = path.join(dataDir, 'sync-history.json')
const lockPath = path.join(dataDir, 'registry.lock')
const strategyPath = path.join(dataDir, 'strategy-templates.json')

const metricKeys = ['quality', 'affordability', 'speed', 'context', 'privacy', 'availability']
const categories = ['general', 'image', 'video', 'voice', 'music', 'document']
const useCases = [
  ...categories,
  'code',
  'agent',
  'rag',
  'vision',
  'image_generation',
  'video_generation',
  'speech_to_text',
  'text_to_speech',
  'music_generation',
  'embedding',
  'reranking',
]
const recordTypes = ['api_model', 'open_weight_model', 'hosted_open_model', 'hf_repo', 'model_family', 'strategy_template']
const sourceAuthorities = ['first_party', 'aggregator', 'benchmark', 'curated', 'seed', 'heuristic']
const linkStatuses = ['verified', 'unverified', 'catalog', 'broken']
const strategyTemplateIds = new Set(['voice-agent', 'embedding-rerank-stack'])

function clamp(value, fallback = 0.5) {
  const numeric = Number(value)

  if (!Number.isFinite(numeric)) {
    return fallback
  }

  return Math.max(0.01, Math.min(1, numeric))
}

function slug(value) {
  return String(value || 'model')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 120)
}

function normalizeEnum(value, allowed, fallback) {
  return allowed.includes(value) ? value : fallback
}

function normalizeUseCases(values, fallback, options = {}) {
  const raw = Array.isArray(values) ? values : fallback
  const normalized = [...new Set(raw.map(String).filter((value) => useCases.includes(value)))]
  if (normalized.length) return normalized
  return options.allowEmpty ? [] : ['general']
}

function inferSecondaryUseCases(model, primaryUseCases) {
  const secondary = new Set(Array.isArray(model.secondaryUseCases) ? model.secondaryUseCases : [])
  const primary = primaryUseCases[0] || model.category || 'general'

  const id = String(model.id || '').toLowerCase()
  const name = String(model.name || '').toLowerCase()
  const desc = String(model.bestFor || '').toLowerCase()
  const modalities = Array.isArray(model.modalities) ? model.modalities.map((m) => String(m).toLowerCase()) : []
  const tags = Array.isArray(model.tags) ? model.tags.map((t) => String(t).toLowerCase()) : []
  const allMods = new Set([...modalities, ...tags])
  const fullText = `${id} ${name} ${desc} ${tags.join(' ')} ${[...allMods].join(' ')}`

  // 1. Coding capability
  const isCoderSpecialist = /coder|coding|starcoder|codestral|codellama|codeqwen|codegeex|wizardcoder|deepseek-coder|qwen-coder|code-alpaca|devstral|python|javascript|typescript|c\+\+|rust|sql|swe-bench|humaneval/i.test(fullText)
  const isCapableGeneralist = primary === 'general' && ((model.metrics?.quality ?? 0) >= 0.75 || Number(model.contextLength || 0) >= 32000 || allMods.has('tools') || /agent|reasoning|cot|r1|o1|o3|o4/i.test(fullText))

  if (isCoderSpecialist || isCapableGeneralist) {
    secondary.add('code')
  }

  // 2. Vision / Image understanding
  if (allMods.has('image') || allMods.has('vision') || /vision|multimodal|vl|image-to-text|visual-qa|ocr/i.test(fullText)) {
    if (primary !== 'image') secondary.add('vision')
  }

  // 3. Image Generation
  if (tags.includes('text-to-image') || tags.includes('image-to-image') || tags.includes('diffusers') || /flux|midjourney|dall-e|stable-diffusion|sdxl|imagen/i.test(fullText)) {
    if (primary !== 'image') secondary.add('image_generation')
  }

  // 4. Video understanding or generation
  if (allMods.has('video') || tags.includes('text-to-video') || tags.includes('image-to-video') || /video-generation|sora|runway|veo|kling|wan-video/i.test(fullText)) {
    if (primary !== 'video') secondary.add('video_generation')
  }

  // 5. Speech & Voice
  if (allMods.has('audio') || /speech-to-text|transcription|whisper|asr/i.test(fullText)) {
    if (primary !== 'voice') secondary.add('speech_to_text')
  }
  if (/text-to-speech|tts|speech-synthesis|narration/i.test(fullText)) {
    if (primary !== 'voice' && primary !== 'music') secondary.add('text_to_speech')
  }

  // 6. Music Generation
  if (/music|suno|udio|musicgen|audiocraft/i.test(fullText)) {
    if (primary !== 'music') secondary.add('music_generation')
  }

  // 7. RAG / Long-context / Document Intelligence
  const contextLength = Number(model.contextLength || 0)
  if (contextLength >= 64000 || allMods.has('file') || /rag|retrieval|document|pdf|contract|colbert|colpali|ocr/i.test(fullText)) {
    if (primary !== 'document') secondary.add('rag')
  }

  // 8. Embeddings & Reranking
  if (/embedding|rerank|sentence-transformers|mteb/i.test(fullText)) {
    if (primary !== 'document') {
      secondary.add('embedding')
      secondary.add('reranking')
    }
  }

  // 9. Agent / Function Calling / Tool Use
  if (allMods.has('tools') || /agent|tools|function-calling|fc|tool_use|act|reasoning|cot/i.test(fullText) || isCapableGeneralist) {
    secondary.add('agent')
  }

  for (const p of primaryUseCases) {
    secondary.delete(p)
  }

  return [...secondary]
}

function inferRecordType(model, id) {
  if (strategyTemplateIds.has(id) || model.recordType === 'strategy_template') return 'strategy_template'
  if (recordTypes.includes(model.recordType)) return model.recordType

  const source = String(model.source || '').toLowerCase()
  const access = String(model.access || '').toLowerCase()
  const provider = String(model.provider || '').toLowerCase()

  if (source === 'hugging face') return access === 'open source' ? 'open_weight_model' : 'hf_repo'
  if (access === 'open source') return 'open_weight_model'
  if (access === 'hosted open model') return 'hosted_open_model'
  if (source === 'seed registry' && (provider.includes(' class') || provider.includes(' / '))) return 'model_family'
  return 'api_model'
}

function inferSourceAuthority(model) {
  if (sourceAuthorities.includes(model.sourceAuthority)) return model.sourceAuthority

  const source = String(model.source || '').toLowerCase()
  if (source.includes('openrouter') || source.includes('hugging face')) return 'aggregator'
  if (source.includes('curated')) return 'curated'
  if (source.includes('seed')) return 'seed'
  return 'heuristic'
}

function inferLinkStatus(model) {
  if (linkStatuses.includes(model.linkStatus)) return model.linkStatus

  const url = String(model.sourceUrl || '')
  if (!url) return 'unverified'
  if (url.includes('/models') || url.includes('/api/v1/models')) return 'catalog'
  return 'verified'
}

// Simple robust lock manager
async function withLock(fn) {
  await ensureDataDir()
  let acquired = false
  for (let i = 0; i < 20; i++) {
    try {
      await mkdir(lockPath)
      acquired = true
      break
    } catch (err) {
      if (err.code !== 'EEXIST') throw err
      await new Promise((resolve) => setTimeout(resolve, 50))
    }
  }
  if (!acquired) {
    throw new Error('Timeout acquiring write lock on registry')
  }
  try {
    return await fn()
  } finally {
    await rm(lockPath, { recursive: true, force: true }).catch(() => {})
  }
}

// ── Distinct, deployable model vs. packaging artefact ──────────────────────
// The catalogue ingests from public hubs, so it accumulates things that are not
// distinct models: quantised re-packages of a model that is already listed
// ("…-GGUF", "…-AWQ", "…-4bit"), training and smoke-test fixtures
// ("trl-internal-testing/tiny-…", "facebook/bart-large-cnn" style research
// checkpoints are real but not generative assistants), and provider catalogue
// pages. Recommending a quantisation next to its own base model is noise, and
// recommending a test fixture is simply wrong. These are excluded from the
// candidate pool while remaining in the catalogue for reference and diagnostics.
const QUANTISATION_ARTEFACT_RE = /(gguf|awq|gptq|exl2|[-_.]?(?:int|fp|nvfp)[0-9]+(?:bit)?\b|[-_.]?(?:2|3|4|5|6|8)bit\b|quantiz|q[2-8]_[k0-9]+)/i
const FIXTURE_ARTEFACT_RE = /(^|[/_-])(tiny|test|testing|dummy|fixture|example|sandbox|scratch)([/_-]|$)/i
const NON_GENERATIVE_FAMILY_RE = /^(google-t5\/|t5-|facebook\/bart|bart-|bert-|google\/bert|roberta|distilbert|albert|electra|sentence-transformers\/)/i

export function isDistinctDeployableModel(model) {
  if (!model || typeof model !== 'object') return false
  const id = String(model.id || '')
  const name = String(model.name || '')
  const hay = `${id} ${name}`
  if (model.recordType === 'strategy_template' || model.recordType === 'placeholder') return false
  if (QUANTISATION_ARTEFACT_RE.test(hay)) return false
  if (FIXTURE_ARTEFACT_RE.test(hay)) return false
  if (NON_GENERATIVE_FAMILY_RE.test(id)) return false
  return true
}

export function normalizeModel(model) {
  const id = slug(model.id || `${model.provider}-${model.name}`)
  const rawCat = String(model.category || 'general')
  const category = rawCat === 'code' ? 'general' : normalizeEnum(rawCat, categories, 'general')
  const primaryUseCases = normalizeUseCases(model.primaryUseCases, [category])
  const secondaryUseCases = normalizeUseCases(
    model.secondaryUseCases,
    inferSecondaryUseCases(model, primaryUseCases),
    { allowEmpty: true }
  ).filter((useCase) => !primaryUseCases.includes(useCase))
  const metrics = {}

  for (const key of metricKeys) {
    metrics[key] = clamp(model.metrics?.[key])
  }

  return {
    id,
    name: String(model.name || id),
    provider: String(model.provider || 'Unknown provider'),
    category,
    primaryUseCases,
    secondaryUseCases,
    recordType: inferRecordType(model, id),
    sourceAuthority: inferSourceAuthority(model),
    linkStatus: inferLinkStatus(model),
    access: String(model.access || 'API'),
    modalities: Array.isArray(model.modalities) ? model.modalities.map(String) : ['text'],
    bestFor: String(model.bestFor || 'General AI model workflows.'),
    source: String(model.source || 'Unknown source'),
    sourceUrl: String(model.sourceUrl || ''),
    benchmarkSources: Array.isArray(model.benchmarkSources) ? model.benchmarkSources.map(String).slice(0, 8) : [],
    lastVerified: String(model.lastVerified || new Date().toISOString()),
    confidence: clamp(model.confidence, 0.5),
    pricing: {
      unit: String(model.pricing?.unit || 'unknown'),
      inputPerMillion: Number(model.pricing?.inputPerMillion || 0),
      outputPerMillion: Number(model.pricing?.outputPerMillion || 0),
    },
    contextLength: Number(model.contextLength || 0),
    metrics,
    // Keep source observations separate from normalized display metrics.  Older
    // registries omit these fields and remain fully compatible.
    metricSources: model.metricSources && typeof model.metricSources === 'object' ? model.metricSources : {},
    benchmarkObservations: model.benchmarkObservations && typeof model.benchmarkObservations === 'object' ? model.benchmarkObservations : {},
    canonicalId: String(model.canonicalId || id),
    aliasOf: model.aliasOf ? String(model.aliasOf) : null,
    inputModalities: Array.isArray(model.inputModalities) ? model.inputModalities.map(String) : [],
    outputModalities: Array.isArray(model.outputModalities) ? model.outputModalities.map(String) : [],
    supportedParameters: Array.isArray(model.supportedParameters)
      ? model.supportedParameters.map(String)
      : (Array.isArray(model.supported_parameters) ? model.supported_parameters.map(String) : []),
    license: model.license ? String(model.license) : null,
    isStale: Boolean(model.isStale),
    // Whether this record is a distinct, callable model rather than a
    // re-packaged quantisation, a test fixture, or a catalogue page. Persisted
    // so clients can label an entry honestly instead of implying it is a model.
    deployable: model.deployable === undefined
      ? isDistinctDeployableModel({ ...model, id, name: String(model.name || id) })
      : Boolean(model.deployable),
  }
}

// Does this record carry a real Artificial Analysis benchmark signal?
function hasAaSignal(model) {
  const aa = model?.benchmarkObservations?.artificial_analysis
  if (!aa || typeof aa !== 'object') return false
  return [aa.intelligence_index, aa.coding_index, aa.agentic_index, aa.math_index]
    .some((value) => Number.isFinite(Number(value)) && Number(value) > 0)
}

export function mergeModels(existing, incoming) {
  const merged = new Map()

  for (const model of existing) {
    const normalized = normalizeModel(model)
    merged.set(normalized.id, normalized)
  }

  for (const model of incoming) {
    const normalized = normalizeModel(model)
    const current = merged.get(normalized.id)

    // Merge, don't clobber: a catalog-only refresh (e.g. an AA-less sync, or a
    // model AA doesn't cover this cycle) must not wipe real benchmark
    // enrichment that a prior keyed sync persisted. When the incoming row has
    // no AA signal but the stored one does, keep the stored observations and
    // the quality/speed metrics + sources those benchmarks produced.
    const preserveBenchmark = !hasAaSignal(normalized) && hasAaSignal(current)

    const metrics = {
      ...current?.metrics,
      ...normalized.metrics,
    }
    const metricSources = { ...normalized.metricSources }
    if (preserveBenchmark) {
      if (Number.isFinite(Number(current.metrics?.quality))) metrics.quality = current.metrics.quality
      if (Number.isFinite(Number(current.metrics?.speed))) metrics.speed = current.metrics.speed
      if (current.metricSources?.quality) metricSources.quality = current.metricSources.quality
      if (current.metricSources?.speed) metricSources.speed = current.metricSources.speed
    }

    merged.set(normalized.id, {
      ...current,
      ...normalized,
      metrics,
      metricSources,
      benchmarkObservations: preserveBenchmark
        ? current.benchmarkObservations
        : normalized.benchmarkObservations,
      pricing: {
        ...current?.pricing,
        ...normalized.pricing,
      },
      benchmarkSources: normalized.benchmarkSources.length
        ? normalized.benchmarkSources
        : (current?.benchmarkSources || []),
      sourceAuthority: preserveBenchmark && current?.sourceAuthority === 'benchmark'
        ? 'benchmark'
        : normalized.sourceAuthority,
      primaryUseCases: normalized.primaryUseCases.length
        ? normalized.primaryUseCases
        : (current?.primaryUseCases || []),
      secondaryUseCases: normalized.secondaryUseCases.length
        ? normalized.secondaryUseCases
        : (current?.secondaryUseCases || []),
    })
  }

  return [...merged.values()].sort((a, b) => a.name.localeCompare(b.name))
}

async function readJson(filePath) {
  const raw = await readFile(filePath, 'utf8')
  return JSON.parse(raw)
}

async function ensureDataDir() {
  await mkdir(dataDir, { recursive: true })
}

export async function readStrategyTemplates() {
  try {
    const raw = await readFile(strategyPath, 'utf8')
    return JSON.parse(raw)
  } catch {
    const defaultStrategies = [
      {
        "id": "realtime-voice-agent",
        "name": "Realtime Voice Agent Stack",
        "useCase": "voice",
        "description": "Voice Agent = STT (Speech-to-Text) + LLM (Reasoning) + TTS (Text-to-Speech) + Streaming Transport. Designed for low-latency conversational audio applications.",
        "recommendedComponents": [
          "Deepgram Nova-2 (STT)",
          "GPT-4o or Claude 3.5 Sonnet (LLM)",
          "ElevenLabs Reader/Multilingual v2 (TTS)"
        ],
        "exampleProviders": [
          "Deepgram",
          "OpenAI",
          "ElevenLabs",
          "Gemini Live API"
        ]
      },
      {
        "id": "embedding-rerank-stack",
        "name": "Embedding & Reranking RAG Stack",
        "useCase": "document",
        "description": "RAG Stack = Vector Embeddings + Vector Database + Cross-Encoder Reranker + LLM. Best for semantic search and precise question-answering over large document catalogs.",
        "recommendedComponents": [
          "Cohere Embed v3 (Embeddings)",
          "Pinecone or pgvector (Database)",
          "Cohere Rerank v3 (Reranking)",
          "Claude 3.5 Sonnet (LLM)"
        ],
        "exampleProviders": [
          "Cohere",
          "Pinecone",
          "Anthropic",
          "Voyage AI"
        ]
      }
    ]
    try {
      await ensureDataDir()
      await writeFile(strategyPath, JSON.stringify(defaultStrategies, null, 2), 'utf8')
    } catch {}
    return defaultStrategies
  }
}

export async function readRegistry() {
  await ensureDataDir()

  try {
    const registry = await readJson(registryPath)
    return {
      ...registry,
      models: (registry.models || []).map(normalizeModel),
    }
  } catch {
    const seed = await readJson(seedPath)
    const models = (seed.models || []).map(normalizeModel)
    const registry = {
      version: seed.version || 1,
      lastUpdated: seed.lastUpdated || new Date().toISOString(),
      models,
    }
    await _writeRegistry(registry.models, { reason: 'initialized from seed registry' })
    return registry
  }
}

async function _writeRegistry(models, metadata = {}) {
  await ensureDataDir()
  const registry = {
    version: 1,
    lastUpdated: new Date().toISOString(),
    metadata,
    models: models.map(normalizeModel),
  }

  await writeFile(registryPath, `${JSON.stringify(registry, null, 2)}\n`, 'utf8')
  return registry
}

export async function writeRegistry(models, metadata = {}) {
  return withLock(() => _writeRegistry(models, metadata))
}

export async function upsertModels(incomingModels, metadata = {}) {
  return withLock(async () => {
    const registry = await readRegistry()
    const before = registry.models.length

    let added = 0
    let updated = 0
    const existingIds = new Set(registry.models.map((m) => m.id))
    for (const model of incomingModels) {
      const normalized = normalizeModel(model)
      if (existingIds.has(normalized.id)) {
        updated++
      } else {
        added++
      }
    }

    const models = mergeModels(registry.models, incomingModels)
    // Merge metadata, don't clobber it: a catalog sync reports its own sources
    // and match counts, but must not erase the cohort `benchmarkBounds` that
    // the offline recalibration persisted. Losing them silently reverts
    // request-time normalization to generic defaults.
    const next = await _writeRegistry(models, { ...registry.metadata, ...metadata })

    return {
      registry: next,
      before,
      after: models.length,
      added,
      updated,
    }
  })
}

export async function appendSyncHistory(entry) {
  return withLock(async () => {
    let history = []

    try {
      history = await readJson(historyPath)
    } catch {
      history = []
    }

    const next = [
      {
        timestamp: new Date().toISOString(),
        ...entry,
      },
      ...history,
    ].slice(0, 50)

    await writeFile(historyPath, `${JSON.stringify(next, null, 2)}\n`, 'utf8')
    return next
  })
}

export async function readSyncHistory() {
  try {
    return await readJson(historyPath)
  } catch {
    return []
  }
}
