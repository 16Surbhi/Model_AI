import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, taskQuality, weightedAdditiveScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'

async function run() {
  const reg = await readRegistry()
  const res = await recommendModels(reg.models, { prompt: 'best coding model', limit: 50 })
  const bounds = computeBenchmarkBounds(reg.models)
  const analysis = res.analysis

  console.log('Analysis weights:', analysis.weights)

  // Let us pick representative models from the actual backend run:
  // Top frontier models: Claude Opus 5.5, Claude Sonnet 5.5, Claude Fable 5.1, GPT-6 Astra, Claude Opus 5, Muse Spark 1.3
  // Mid frontier/high-throughput: GLM 5.3, DeepSeek V4.1 Flash
  // Budget/small models: Llama 3.2 3B Instruct
  const targetIds = [
    'openrouter-anthropic-claude-opus-5-5',
    'openrouter-anthropic-claude-sonnet-5-5',
    'openrouter-anthropic-claude-fable-5-1',
    'openrouter-openai-gpt-6-astra',
    'openrouter-anthropic-claude-opus-5',
    'openrouter-meta-muse-spark-1-3',
    'openrouter-z-ai-glm-5-3',
    'openrouter-deepseek-deepseek-v4-1-flash',
    'openrouter-meta-llama-llama-3-2-3b-instruct',
  ]

  const cohort = targetIds
    .map((id) => {
      const m = reg.models.find((x) => x.id === id)
      if (!m) return null
      const caps = evaluateModelCapabilities(m, bounds)
      const enriched = { ...m, caps, metrics: { ...m.metrics } }
      for (const key of ['quality', 'speed', 'privacy']) {
        if (!Number.isFinite(enriched.metrics[key])) enriched.metrics[key] = caps[key]
      }
      const q = taskQuality(enriched, ['general', 'code'])
      const a = enriched.metrics.affordability
      const s = enriched.metrics.speed
      // Also get the real recommendation score from the backend run
      const rec = res.recommendations.find((r) => r.id === id)
      return {
        id,
        name: m.name,
        q,
        a,
        s,
        backendScore: rec ? rec.score : null,
        backendTier: rec ? rec.paretoTier : null,
        backendRank: rec ? res.recommendations.indexOf(rec) + 1 : '>50',
        raw: enriched,
      }
    })
    .filter(Boolean)

  console.log('\n--- Actual Backend Results for "best coding model" ---')
  cohort.forEach((m) => {
    console.log(
      `${m.name.padEnd(30)} | q: ${m.q.toFixed(3)} | a: ${m.a.toFixed(3)} | s: ${m.s.toFixed(3)} | Score: ${m.backendScore} | Tier: ${m.backendTier} | Backend Rank: ${m.backendRank}`
    )
  })

  // Now, what if someone used WSM, TOPSIS, Cobb-Douglas vs Proposed?
  // Let's compute with the weights inferred by the parser:
  // weights: quality: 3.6, affordability: 0, speed: 0.05, context: 0.2, privacy: 0.05, availability: 0.15
  // And also what if weights were balanced vs quality-focused:
  console.log('\n--- Multi-attribute utility with parser weights ---')
  const weights = analysis.weights
  const wSum = Object.values(weights).reduce((a, b) => a + b, 0)
  
  cohort.forEach((m) => {
    // WSM over the full weights
    let wsmVal = 0
    for (const [k, w] of Object.entries(weights)) {
      wsmVal += (w / wSum) * (m.raw.metrics[k] ?? 0.08)
    }
    m.wsmParsed = wsmVal * 100

    // Cobb-Douglas over the full weights (with floor 0.05)
    let cdVal = 1
    for (const [k, w] of Object.entries(weights)) {
      const xk = Math.max(m.raw.metrics[k] ?? 0.05, 0.05)
      cdVal *= Math.pow(xk, w / wSum)
    }
    m.cdParsed = cdVal * 100

    // Proposed score via backend
    const rankingModel = { ...m.raw, metrics: { ...m.raw.metrics, quality: m.q } }
    m.proposedParsed = weightedAdditiveScore(rankingModel, analysis.targetCategory, weights, {
      qualityFlagged: true,
      archetype: analysis.archetype,
      isCoding: analysis.isCoding,
    })
  })

  console.log('Model'.padEnd(25) + ' q      a      s   WSM(parsed)  Cobb-D(parsed)  Proposed(parsed)')
  cohort.forEach((m) => {
    console.log(
      m.name.padEnd(25) +
        m.q.toFixed(3) +
        '  ' +
        m.a.toFixed(3) +
        '  ' +
        m.s.toFixed(3) +
        '   ' +
        m.wsmParsed.toFixed(1) +
        '         ' +
        m.cdParsed.toFixed(1) +
        '           ' +
        m.proposedParsed
    )
  })
}

run()
