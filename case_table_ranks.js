/** Recompute tab:coding_case with UNROUNDED scores and ranks on those values.
 *  Run: node scratch/case_table_ranks.js */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, capabilityFit, sourceTrustFactor } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'

const IDS = [
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
const KNEE = 86.0, CAP = 99.4, SCALE = CAP - KNEE
const soft = (s) => (s <= KNEE ? s : KNEE + SCALE * (1 - Math.exp(-(s - KNEE) / SCALE)))

function rawScore(model, category, weights, opts) {
  const { archetype, isCoding } = opts
  const wSum = Object.values(weights).reduce((a, b) => a + b, 0)
  const m = model.metrics
  const sqrtSum = Object.entries(weights).reduce((s, [k, w]) => s + w * Math.sqrt(Math.max(m[k] ?? 0.08, 0.08)), 0)
  let s = Math.pow(sqrtSum / wSum, 2)
  s *= Math.pow(capabilityFit(model, category), 1.2)
  s *= 1.0 - 0.20 * Math.pow(1 - (model.confidence ?? 0.8), 2)
  s *= sourceTrustFactor(model)
  s *= 100
  const q = m.quality ?? 0
  if (archetype === 'frontier') {
    s *= q < 0.92 ? 1 / (1 + 3.0 * Math.pow(0.92 - q, 1.2)) : 1 + 0.25 * Math.pow(Math.max(0, (q - 0.92) / 0.08), 1.5)
  } else if (q < 0.70) {
    s *= 1 / (1 + 1.8 * Math.log(1 + 3.5 * (0.70 - q)))
  }
  if (isCoding && category === 'general') {
    const caps = model.caps || {}
    const spec = caps.isCodeSpecialist || (model.secondaryUseCases || []).includes('code') || model.category === 'code'
    if (spec || caps.isReasoning) s *= 1 + (spec ? 0.04 : 0) + (caps.isReasoning ? 0.03 : 0) + (caps.hasTools ? 0.02 : 0)
    else if (q < 0.95) s *= 0.92
  }
  return s
}

const reg = await readRegistry()
const bounds = computeBenchmarkBounds(reg.models)
const { analysis } = await recommendModels(reg.models, { prompt: 'best coding model', limit: 60 })
const w = analysis.weights
const WKEYS = ['quality', 'affordability', 'speed', 'context', 'privacy', 'availability']
const wSumAll = WKEYS.reduce((a, k) => a + (w[k] || 0), 0)
const W3 = [w.quality, w.affordability, w.speed], w3s = W3.reduce((a, b) => a + b, 0) || 1
const NW3 = W3.map((v) => v / w3s)
console.log(`weights=${JSON.stringify(w)} archetype=${analysis.archetype} coding=${analysis.isCoding}\n`)

const rows = []
for (const [id, name] of IDS) {
  const m = reg.models.find((x) => x.id === id)
  if (!m) { console.log(`${name}: NOT FOUND`); continue }
  const caps = evaluateModelCapabilities(m, bounds)
  const e = { ...m, caps, metrics: { ...m.metrics } }
  for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
  e.metrics.quality = caps.coding
  const q = e.metrics.quality, a = e.metrics.affordability, s = e.metrics.speed
  const wsm = 100 * WKEYS.reduce((t, k) => t + (w[k] || 0) / wSumAll * (e.metrics[k] ?? 0.08), 0)
  const wsmFloor = q >= 0.92 ? wsm : null
  const cd = 100 * Math.pow(Math.max(q, .05), NW3[0]) * Math.pow(Math.max(a, .05), NW3[1]) * Math.pow(Math.max(s, .05), NW3[2])
  const proposed = soft(rawScore(e, analysis.targetCategory, w, { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding) }))
  rows.push({ name, q, a, s, wsm, wsmFloor, cd, proposed, raw: e })
}
// TOPSIS over (q,a,s)
const M = rows.map((r) => [r.q, r.a, r.s])
const nr = [0, 1, 2].map((j) => Math.sqrt(M.reduce((t, r) => t + r[j] * r[j], 0)))
const V = M.map((r) => r.map((v, j) => (v / nr[j]) * NW3[j]))
const best = [0, 1, 2].map((j) => Math.max(...V.map((r) => r[j])))
const worst = [0, 1, 2].map((j) => Math.min(...V.map((r) => r[j])))
rows.forEach((r, i) => {
  const dp = Math.sqrt(V[i].reduce((t, v, j) => t + (v - best[j]) ** 2, 0))
  const dm = Math.sqrt(V[i].reduce((t, v, j) => t + (v - worst[j]) ** 2, 0))
  r.topsis = 100 * dm / (dp + dm || 1)
})
// ranks on UNROUNDED values (floor-skipped entries get a relegated rank)
const rankOf = (key, skipFloor) => {
  const order = rows.map((r, i) => [r[key] ?? -1e9, i]).sort((p, q) => q[0] - p[0])
  const out = {}
  order.forEach(([v, i], n) => { out[rows[i].name] = skipFloor && rows[i].wsmFloor == null ? `<fl.` : n + 1 })
  return out
}
const R = { wsm: rankOf('wsm'), floor: rankOf('wsmFloor', true), topsis: rankOf('topsis'), cd: rankOf('cd'), proposed: rankOf('proposed') }
// Pareto status over (q,a,s) within this panel (computed directly)
const pareto = new Set()
for (let i = 0; i < rows.length; i++) {
  const A = rows[i]
  const dominated = rows.some((B, j) => j !== i && B.q >= A.q && B.a >= A.a && B.s >= A.s && (B.q > A.q || B.a > A.a || B.s > A.s))
  if (!dominated) pareto.add(i)
}

console.log('model                q      a      s     | WSM    rk | fl.   rk | TOPSIS rk | CD     rk | PROP    rk  | Pareto')
for (let i = 0; i < rows.length; i++) {
  const r = rows[i]
  const f = (x) => (x == null ? '  --  ' : x.toFixed(2).padStart(6))
  console.log(`${r.name.padEnd(20)} ${r.q.toFixed(3)} ${r.a.toFixed(3)} ${r.s.toFixed(3)} | ${f(r.wsm)} ${String(R.wsm[r.name]).padStart(4)} | ${f(r.wsmFloor)} ${String(R.floor[r.name]).padStart(4)} | ${f(r.topsis)} ${String(R.topsis[r.name]).padStart(4)} | ${f(r.cd)} ${String(R.cd[r.name]).padStart(4)} | ${r.proposed.toFixed(3).padStart(7)} ${String(R.proposed[r.name]).padStart(4)} | ${pareto.has(i) ? 'Non-dom.' : 'Dom.'}`)
}
console.log('\nProposed unrounded (2dp) in row order:')
for (const r of rows) console.log(`  ${r.name.padEnd(20)} ${r.proposed.toFixed(2)}  rank ${R.proposed[r.name]}`)

// Machine-readable output for the paper verifier.
import { writeFileSync } from 'node:fs'
writeFileSync('scratch/case_table_ranks.json', JSON.stringify({
  weights: w,
  archetype: analysis.archetype,
  rows: rows.map((r) => ({
    name: r.name, q: +r.q.toFixed(3), a: +r.a.toFixed(3), s: +r.s.toFixed(3),
    wsm: +r.wsm.toFixed(2), wsm_floor: r.wsmFloor == null ? null : +r.wsmFloor.toFixed(2),
    topsis: +r.topsis.toFixed(2), cd: +r.cd.toFixed(2), proposed: +r.proposed.toFixed(2),
    ranks: { wsm: R.wsm[r.name], floor: R.floor[r.name], topsis: R.topsis[r.name], cd: R.cd[r.name], proposed: R.proposed[r.name] },
    pareto: pareto.has(rows.indexOf(r)) ? 'Non-dom.' : 'Dom.',
  })),
}, null, 2))
console.log('wrote scratch/case_table_ranks.json')
