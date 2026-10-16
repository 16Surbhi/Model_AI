/** Regenerate tab:ablation (raw pre-ceiling scores across CES exponents) and the
 *  ceiling-test endpoints from the shipped engine.
 *  Run: node scratch/regen_ablation.js */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, capabilityFit, sourceTrustFactor } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'

const NAMES = [
  ['openrouter-anthropic-claude-opus-5-5', 'Claude Opus 5.5'],
  ['openrouter-anthropic-claude-sonnet-5-5', 'Claude Sonnet 5.5'],
  ['openrouter-anthropic-claude-fable-5-1', 'Claude Fable 5.1'],
  ['openrouter-openai-gpt-6-astra', 'GPT-6 Astra'],
  ['openrouter-anthropic-claude-opus-5', 'Claude Opus 5'],
  ['openrouter-meta-muse-spark-1-3', 'Muse Spark 1.3'],
  ['openrouter-z-ai-glm-5-3', 'GLM 5.3'],
  ['openrouter-deepseek-deepseek-v4-1-flash', 'DeepSeek V4.1 Flash'],
  ['openrouter-meta-llama-llama-3-2-3b-instruct', 'Llama 3.2 3B Instr.'],
]
const EPS = 0.05, KNEE = 86.0, CAP = 99.4
const soft = (s) => (s <= KNEE ? s : KNEE + (CAP - KNEE) * (1 - Math.exp(-(s - KNEE) / (CAP - KNEE))))

// Power mean U_rho over the same weights/metrics the scorer consumes.
function powerMean(metrics, weights, rho) {
  const sum = Object.values(weights).reduce((a, b) => a + b, 0)
  const ks = Object.keys(weights)
  if (rho === 0) {
    let prod = 1
    for (const k of ks) prod *= Math.pow(Math.max(metrics[k] ?? EPS, EPS), weights[k] / sum)
    return prod
  }
  const s = ks.reduce((acc, k) => acc + weights[k] * Math.pow(Math.max(metrics[k] ?? 0, 0), rho), 0)
  return Math.pow(s / sum, 1 / rho)
}

function rawScore(model, category, weights, opts, rho) {
  const m = model.metrics
  let s = powerMean(m, weights, rho)
  s *= Math.pow(capabilityFit(model, category), 1.2)
  s *= 1.0 - 0.20 * Math.pow(1 - (model.confidence ?? 0.8), 2)
  s *= sourceTrustFactor(model)
  s *= 100
  const q = m.quality ?? 0
  if (opts.archetype === 'frontier') {
    s *= q < 0.92 ? 1 / (1 + 3.0 * Math.pow(0.92 - q, 1.2))
                  : 1 + 0.25 * Math.pow(Math.max(0, (q - 0.92) / 0.08), 1.5)
  } else if (q < 0.70) {
    s *= 1 / (1 + 1.8 * Math.log(1 + 3.5 * (0.70 - q)))
  }
  if (opts.isCoding && category === 'general') {
    const caps = model.caps || {}
    const spec = caps.isCodeSpecialist || (model.secondaryUseCodes || []).includes('code') || model.category === 'code'
    if (spec || caps.isReasoning) s *= 1 + (spec ? 0.04 : 0) + (caps.isReasoning ? 0.03 : 0) + (caps.hasTools ? 0.02 : 0)
    else if (q < 0.95) s *= 0.92
  }
  return s
}

const reg = await readRegistry()
const bounds = computeBenchmarkBounds(reg.models)
const { analysis, parserUsed } = await recommendModels(reg.models, { prompt: 'best coding model', limit: 25 })
const w = analysis.weights
console.log(`parserUsed=${parserUsed}  archetype=${analysis.archetype}  isCoding=${analysis.isCoding}`)
console.log(`weights=${JSON.stringify(w)}\n`)

const RHOS = [0, 0.25, 0.5, 1.0]
const rows = []
for (const [id, name] of NAMES) {
  const m = reg.models.find((x) => x.id === id)
  if (!m) { console.log(`${name}: NOT FOUND`); continue }
  const caps = evaluateModelCapabilities(m, bounds)
  const e = { ...m, caps, metrics: { ...m.metrics } }
  for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
  e.metrics.quality = e.caps.coding
  const opts = { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding) }
  rows.push({ name, scores: RHOS.map((r) => rawScore(e, analysis.targetCategory, w, opts, r)) })
}

// Rank within each rho column (ties broken by the raw value itself).
for (const r of RHOS) {
  const col = r === 0 ? 0 : RHOS.indexOf(r)
  const idx = rows.map((_, i) => i).sort((a, b) => rows[b].scores[col] - rows[a].scores[col])
  idx.forEach((i, rank) => { rows[i].ranks = rows[i].ranks || {}; rows[i].ranks[r] = rank + 1 })
}

console.log('=== tab:ablation (raw, deficit penalty active, display ceiling NOT applied) ===')
console.log('model'.padEnd(21) + RHOS.map((r) => `rho=${r}`.padStart(14)).join(''))
for (const row of rows) {
  console.log(row.name.padEnd(21) + row.scores.map((s, i) => `${s.toFixed(1)}~(${row.ranks[RHOS[i]]})`.padStart(14)).join(''))
}

// Spearman across rho pairs, plus the ceiling-test endpoints.
function rank(arr) {
  const order = arr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
  const out = new Array(arr.length)
  order.forEach(([, i], r) => { out[i] = r + 1 })
  return out
}
function spearman(a, b) {
  const n = a.length
  const ra = rank(a), rb = rank(b)
  const d2 = ra.reduce((s, v, i) => s + (v - rb[i]) ** 2, 0)
  return 1 - (6 * d2) / (n * (n * n - 1))
}
console.log('\nSpearman across exponents:',
  RHOS.slice(1).map((r) => `rho0 vs rho${r}=${spearman(rows.map((x) => x.scores[0]), rows.map((x) => x.scores[RHOS.indexOf(r)])).toFixed(3)}`).join('  '))

console.log('\n=== ceiling-test endpoints (rho=0.5 raw -> soft ceiling) ===')
for (const name of ['Claude Opus 5.5', 'Muse Spark 1.3', 'DeepSeek V4.1 Flash']) {
  const row = rows.find((x) => x.name === name)
  const raw = row.scores[2]
  console.log(`${name.padEnd(21)} raw=${raw.toFixed(3)}  soft=${soft(raw).toFixed(3)}  hardClip=99`)
}
