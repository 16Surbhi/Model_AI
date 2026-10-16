/**
 * Nominal weights versus realised influence (swing weights), on the frozen
 * cohort and the shipped profiles.
 *
 * Responds to the review point that the paper's weights are nominal importance
 * values, not swing weights: a coordinate that moves across the cohort decides
 * a ranking in proportion to weight x dispersion, so a nominal 0.05 can carry
 * 7-9% of the realised ranking power. The standard MAUT remedy is to normalise
 * by the swing (the range the coordinate actually takes over the decision
 * cohort), which is what the "span-normalised" row below does.
 *
 * For each profile we report:
 *   nominal share        w_k / sum(w)
 *   realised influence   w_k * s_k / sum_j(w_j * s_j), s_k = sqrt-space span
 *   span-normalised      w_k / s_k renormalised, i.e. equal influence per unit
 *                        of swing; then renormalised to sum to 1
 * and how much the resulting ranking differs from the nominal-weight ranking.
 *
 * Run: node scratch/weights_swing.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { taskQuality, recommendModels, computeFinalScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'
import { readFileSync, writeFileSync } from 'node:fs'
import { pinnedIntent } from './pinned-intents.js'

const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const EFFORT = (n) => (/max/i.test(n) ? 3 : /xhigh/i.test(n) ? 2 : /high/i.test(n) ? 1 : 0)
const KEYS = ['quality', 'affordability', 'speed', 'context', 'privacy', 'availability']

function spearman(x, y) {
  const rank = (v) => {
    const idx = v.map((val, i) => [val, i]).sort((p, q) => p[0] - q[0]); const r = new Array(v.length)
    for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; const avg = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k][1]] = avg; i = j + 1 }
    return r
  }
  const rx = rank(x), ry = rank(y)
  const m = (v) => v.reduce((a, b) => a + b, 0) / v.length
  const mx = m(rx), my = m(ry)
  let n = 0, dx = 0, dy = 0
  for (let i = 0; i < rx.length; i++) { n += (rx[i] - mx) * (ry[i] - my); dx += (rx[i] - mx) ** 2; dy += (ry[i] - my) ** 2 }
  return n / Math.sqrt(dx * dy)
}

const reg = await readRegistry()
const bounds = computeBenchmarkBounds(reg.models)
const idx = new Map()
for (const m of reg.models) {
  if (m.aliasOf) continue
  for (const k of [norm(m.name), norm(strip(m.name))]) { if (!idx.has(k)) idx.set(k, []); idx.get(k).push(m) }
}
const snap = JSON.parse(readFileSync('scratch/arena_snapshot_2026-10-08.json', 'utf8'))
const rows = Array.isArray(snap) ? snap : snap.models
const groups = new Map()
for (const a of rows) { const h = idx.get(norm(a.model)) || idx.get(norm(strip(a.model))); if (!h) continue; const m = h[0]; const g = groups.get(m.id) || { model: m, rows: [] }; g.rows.push(a); groups.set(m.id, g) }
const cohort = []
for (const { model, rows: rs } of groups.values()) {
  rs.sort((a, b) => EFFORT(b.model) - EFFORT(a.model))
  const caps = evaluateModelCapabilities(model, bounds)
  const e = { ...model, caps, metrics: { ...model.metrics } }
  for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
  e.metrics.quality = taskQuality(e, ['general', 'code'])
  cohort.push(e)
}

// sqrt-space span per coordinate: what a unit of weight buys in ranking power
const spans = {}
for (const k of KEYS) {
  const v = cohort.map((m) => Math.sqrt(Math.max(m.metrics?.[k] ?? 0.08, 0.08)))
  spans[k] = Math.max(...v) - Math.min(...v)
}
console.log(`frozen cohort n=${cohort.length}`)
console.log('sqrt-space spans:', Object.entries(spans).map(([k, v]) => `${k}=${v.toFixed(4)}`).join('  '))

const out = { cohort: cohort.length, spans: {}, panels: {} }
out.spans = spans

for (const prompt of ['best frontier coding models', 'budget coding models', 'balanced coding models']) {
  const { analysis } = await pinnedIntent(prompt)
  const w = analysis.weights
  const wSum = Object.values(w).reduce((a, b) => a + b, 0)
  const influence = {}, nominal = {}
  let inflSum = 0
  for (const k of KEYS) { influence[k] = (w[k] || 0) * spans[k]; inflSum += influence[k] }
  const inflShare = {}, nomShare = {}
  for (const k of KEYS) { inflShare[k] = +(100 * influence[k] / inflSum).toFixed(1); nomShare[k] = +(100 * (w[k] || 0) / wSum).toFixed(1) }

  // span-normalised: equal influence per unit swing -> w_k / s_k, renormalised
  // A coordinate with zero span cannot decide anything, so it is excluded from
  // the normalisation: any weight on it is equivalent.
  const movable = KEYS.filter((k) => spans[k] > 1e-9)
  const sn = {}, snSum = movable.reduce((t, k) => t + (w[k] || 0) / spans[k], 0)
  for (const k of KEYS) sn[k] = movable.includes(k) ? (w[k] || 0) / spans[k] / snSum * wSum : 0

  const opts = { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: analysis.archetype === 'frontier' }
  const base = cohort.map((m) => computeFinalScore(m, analysis.targetCategory, w, opts).exact)
  const swing = cohort.map((m) => computeFinalScore(m, analysis.targetCategory, sn, opts).exact)
  const rho = spearman(base, swing)
  const top = (v) => v.map((val, i) => [val, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((p) => cohort[p[1]].name)
  console.log(`\n===== ${prompt}`)
  console.log('  coord          nominal%  influence%   span-normalised weight')
  for (const k of KEYS) console.log(`  ${k.padEnd(14)} ${String(nomShare[k]).padStart(6)} ${String(inflShare[k]).padStart(11)}      ${sn[k].toFixed(2)}`)
  console.log(`  rank correlation nominal vs span-normalised: r_s = ${rho.toFixed(3)}`)
  console.log(`  top three nominal:            ${top(base).join(' | ')}`)
  console.log(`  top three span-normalised:    ${top(swing).join(' | ')}`)
  out.panels[prompt] = { nominal: nomShare, influence: inflShare, spanNormalised: sn, spearman: +rho.toFixed(3), topNominal: top(base), topSwing: top(swing) }
}

writeFileSync('scratch/weights_swing.json', JSON.stringify(out, null, 1))
console.log('\nwrote scratch/weights_swing.json')
