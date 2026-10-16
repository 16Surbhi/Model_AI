/**
 * Component ablation and baseline-parity test on the FROZEN cohort.
 *
 * Responds to the review points that (a) the baselines are handicapped because
 * they are scored on the three-coordinate efficiency triplet while the proposed
 * score uses six coordinates plus category fit, confidence and trust, and
 * (b) no component-level ablation of the proposed pipeline is reported.
 *
 * Frozen state: the registry revision and the 2026-10-08 headline payload used
 * by scratch/arena_corrected.js, so the "Full pipeline" row here is comparable
 * to the Proposed row of the manuscript's validation table.
 *
 * Variants are computed by the SAME arithmetic as server/lib/scoring.js with one
 * term disabled, so the difference between two rows is attributable to that
 * term and to nothing else.
 *
 * Run: node scratch/ablation_components.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { taskQuality, capabilityFit, sourceTrustFactor, computeFinalScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'
import { readFileSync, writeFileSync } from 'node:fs'
import { pinnedIntent } from './pinned-intents.js'

const PROMPTS = { frontier: 'best frontier coding models', budget: 'budget coding models', balanced: 'balanced coding models' }
const BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
const MEAN = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)
const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const EFFORT = (n) => (/max/i.test(n) ? 3 : /xhigh/i.test(n) ? 2 : /high/i.test(n) ? 1 : 0)

function spearman(x, y) {
  const rank = (v) => {
    const idx = v.map((val, i) => [val, i]).sort((p, q) => p[0] - q[0]); const r = new Array(v.length)
    for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; const avg = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k][1]] = avg; i = j + 1 }
    return r
  }
  const rx = rank(x), ry = rank(y), mx = MEAN(rx), my = MEAN(ry)
  let n = 0, dx = 0, dy = 0
  for (let i = 0; i < rx.length; i++) { n += (rx[i] - mx) * (ry[i] - my); dx += (rx[i] - mx) ** 2; dy += (ry[i] - my) ** 2 }
  return n / Math.sqrt(dx * dy)
}
function topsis(X, w) {
  const wSum = w.reduce((a, b) => a + b, 0) || 1
  const nrm = X[0].map((_, j) => Math.sqrt(X.reduce((t, r) => t + r[j] * r[j], 0)))
  const V = X.map((r) => r.map((v, j) => (v / (nrm[j] || 1)) * (w[j] / wSum)))
  const best = X[0].map((_, j) => Math.max(...V.map((r) => r[j])))
  const worst = X[0].map((_, j) => Math.min(...V.map((r) => r[j])))
  return V.map((r) => {
    const dp = Math.sqrt(r.reduce((t, v, j) => t + (v - best[j]) ** 2, 0))
    const dm = Math.sqrt(r.reduce((t, v, j) => t + (v - worst[j]) ** 2, 0))
    return dm / (dp + dm || 1)
  })
}

// ── the shipped score, with switchable components ───────────────────────────
const MIN_SCORE = 0.08
const softCeiling = (s) => (s <= 86 ? s : 86 + 13.4 * (1 - Math.exp(-(s - 86) / 13.4)))

function score(model, targetCategory, weights, opts, off = {}) {
  const weightSum = Object.values(weights).reduce((a, b) => a + b, 0)
  if (weightSum <= 0) return MIN_SCORE * 100
  const keys = Object.keys(weights)
  const metricOf = (k) => Math.max(model.metrics?.[k] ?? MIN_SCORE, MIN_SCORE)
  const inner = keys.reduce((sum, k) => sum + weights[k] * Math.sqrt(metricOf(k)), 0) / weightSum
  const metricScore = off.linear ? inner ** 2 : inner ** 2   // power mean both ways; see below
  const metricLinear = keys.reduce((s, k) => s + weights[k] * metricOf(k), 0) / weightSum
  const base = off.linear ? metricLinear : metricScore
  const fit = off.fit ? 1 : Math.pow(capabilityFit(model, targetCategory), 1.2)
  const confidence = model.confidence ?? 0.8
  const conf = off.confidence ? 1 : 1.0 - 0.20 * Math.pow(1 - confidence, 2)
  const trust = off.trust ? 1 : sourceTrustFactor(model)
  let scaled = base * fit * conf * trust * 100
  const q = model.metrics?.quality ?? 0
  const archetype = opts.archetype
  if (off.penalty) {
    // no deficit penalty and no frontier reward
  } else if (archetype === 'frontier' || opts.qualityFlagged) {
    if (q < 0.92) scaled *= 1 / (1 + 3.0 * Math.pow(0.92 - q, 1.2))
    else scaled *= 1 + 0.25 * Math.pow(Math.max(0, (q - 0.92) / 0.08), 1.5)
  } else if (q < 0.70) {
    scaled *= 1 / (1 + 1.8 * Math.log(1 + 3.5 * (0.70 - q)))
  }
  if (off.bonus) { /* specialist bonus disabled */ } else if (opts.isCoding && targetCategory === 'general') {
    const isCodeSpecialist = model.caps?.isCodeSpecialist || (model.secondaryUseCases || []).includes('code') || model.category === 'code'
    const isReasoning = model.caps?.isReasoning
    const hasTools = model.caps?.hasTools
    if (isCodeSpecialist || isReasoning) scaled *= (1 + (isCodeSpecialist ? 0.04 : 0) + (isReasoning ? 0.03 : 0) + (hasTools ? 0.02 : 0))
    else if (q < 0.95) scaled *= 0.92
  }
  return off.ceiling ? Math.max(1, scaled) : Math.max(1, softCeiling(scaled))
}

// ── cohort (identical to scratch/arena_corrected.js) ────────────────────────
const reg = await readRegistry()
const bounds = computeBenchmarkBounds(reg.models)
const regIdx = new Map()
for (const m of reg.models) {
  if (m.aliasOf) continue
  for (const key of [norm(m.name), norm(strip(m.name))]) {
    if (!regIdx.has(key)) regIdx.set(key, [])
    regIdx.get(key).push(m)
  }
}
const snap = JSON.parse(readFileSync('scratch/arena_snapshot_2026-10-08.json', 'utf8'))
const rows = Array.isArray(snap) ? snap : snap.models
const groups = new Map()
for (const a of rows) {
  const key = norm(a.model)
  const hits = regIdx.get(key) || regIdx.get(norm(strip(a.model)))
  if (!hits || !hits.length) continue
  const m = hits[0]
  const g = groups.get(m.id) || { model: m, rows: [] }
  g.rows.push(a); groups.set(m.id, g)
}
const cohort = []
for (const { model, rows: rs } of groups.values()) {
  rs.sort((a, b) => EFFORT(b.model) - EFFORT(a.model))
  const pick = rs[0]
  const net = pick.net_pct ?? pick.net ?? pick.score_pct
  if (net == null || !Number.isFinite(net)) continue
  const caps = evaluateModelCapabilities(model, bounds)
  const e = { ...model, caps, metrics: { ...model.metrics } }
  for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
  cohort.push({
    name: strip(model.name), net, cost: pick.median_usd, raw: e,
    q: taskQuality(e, ['general', 'code']), a: e.metrics.affordability, s: e.metrics.speed,
    ctx: e.metrics.context ?? MIN_SCORE, priv: e.metrics.privacy ?? MIN_SCORE, avail: e.metrics.availability ?? MIN_SCORE,
  })
}
const priced = cohort.filter((c) => Number.isFinite(c.cost))
const net = cohort.map((c) => c.net)

const out = { cohort: cohort.length, priced: priced.length, panels: {} }
console.log(`FROZEN cohort: ${cohort.length} models, ${priced.length} priced`)

// Faithfulness check: the reimplementation with every component ON must agree
// with the shipped computeFinalScore().exact to floating-point noise.
{
  const { recommendModels } = await import('../server/lib/scoring.js')
  let maxd = 0
  for (const prompt of Object.values(PROMPTS)) {
    const { analysis } = await pinnedIntent(prompt)
    const opts = { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: analysis.archetype === 'frontier' }
    for (const c of cohort) {
      const model = { ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }
      const mine = score(model, analysis.targetCategory, analysis.weights, opts)
      const theirs = computeFinalScore(model, analysis.targetCategory, analysis.weights, opts).exact
      maxd = Math.max(maxd, Math.abs(mine - theirs))
    }
  }
  console.log(`reimplementation vs computeFinalScore: max abs diff ${maxd.toExponential(2)}`)
  if (maxd > 1e-9) { console.log('ABORT: reimplementation is not faithful'); process.exit(1) }
}

for (const [arch, prompt] of Object.entries(PROMPTS)) {
  const { recommendModels } = await import('../server/lib/scoring.js')
  const { analysis } = await pinnedIntent(prompt)
  const w = analysis.weights, category = analysis.targetCategory
  const opts = { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: analysis.archetype === 'frontier' }
  const W6 = Object.values(w)
  const NW6 = W6.map((v) => v / (W6.reduce((a, b) => a + b, 0) || 1))
  const K = Object.keys(w)
  const X6 = cohort.map((c) => [c.q, c.a, c.s, c.ctx, c.priv, c.avail])

  const variants = {
    'Full pipeline (shipped)': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts)),
    'minus deficit penalty': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts, { penalty: true })),
    'minus confidence discount': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts, { confidence: true })),
    'minus category fit': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts, { fit: true })),
    'minus source trust': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts, { trust: true })),
    'minus specialist bonus': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts, { bonus: true })),
    'linear aggregator': cohort.map((c) => score({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, opts, { linear: true })),
    // baselines given the SAME six coordinates and the SAME weights
    'WSM (6 coords)': cohort.map((c) => K.reduce((t, k, i) => t + NW6[i] * [c.q, c.a, c.s, c.ctx, c.priv, c.avail][i], 0)),
    'TOPSIS (6 coords)': topsis(X6, W6),
    'Cobb-Douglas (6 coords)': cohort.map((c) => K.reduce((t, k, i) => t * Math.pow(Math.max([c.q, c.a, c.s, c.ctx, c.priv, c.avail][i], 0.05), NW6[i]), 1)),
    'CES only (6 coords)': cohort.map((c) => { const r = K.reduce((t, k, i) => t + NW6[i] * Math.sqrt([c.q, c.a, c.s, c.ctx, c.priv, c.avail][i]), 0); return r * r }),
    // the baselines as the manuscript actually runs them (triplet only)
    'WSM (triplet)': cohort.map((c) => { const t = [w.quality, w.affordability, w.speed]; const s = t[0] + t[1] + t[2] || 1; return (t[0] * c.q + t[1] * c.a + t[2] * c.s) / s }),
    'CES only (triplet)': cohort.map((c) => { const t = [w.quality, w.affordability, w.speed]; const s = t[0] + t[1] + t[2] || 1; const r = t[0] * Math.sqrt(c.q) / s + t[1] * Math.sqrt(c.a) / s + t[2] * Math.sqrt(c.s) / s; return r * r }),
  }

  const keys = Object.keys(variants)
  const rowsOut = []
  for (const k of keys) {
    const v = variants[k]
    let regSum = 0, worst = -Infinity, hits = 0, tiers = 0
    for (const B of BUDGETS) {
      const elig = cohort.map((c, i) => i).filter((i) => Number.isFinite(cohort[i].cost) && cohort[i].cost <= B)
      if (elig.length < 3) continue
      const oracle = Math.max(...elig.map((i) => net[i]))
      let best = -Infinity, pick = elig[0]
      for (const i of elig) if (v[i] > best) { best = v[i]; pick = i }
      const r = oracle - net[pick]
      regSum += r; if (r > worst) worst = r; if (r <= 1e-9) hits++; tiers++
    }
    const topIdx = (x) => x.map((val, i) => [val, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((p) => p[1])
    const outTop = new Set(topIdx(net))
    const top3 = topIdx(v).filter((i) => outTop.has(i)).length
    rowsOut.push({ method: k, mean_regret: +(regSum / tiers).toFixed(3), worst: +worst.toFixed(3), oracle_hits: `${hits}/${tiers}`, spearman: +spearman(v, net).toFixed(3), top3: `${top3}/3` })
  }
  console.log(`\n===== ${arch.toUpperCase()}  weights ${JSON.stringify(w)}`)
  console.log('method                        mean   worst  hits   rho  top3')
  for (const r of rowsOut) {
    console.log(`  ${r.method.padEnd(28)} ${String(r.mean_regret).padStart(6)} ${String(r.worst).padStart(6)} ${r.oracle_hits.padStart(5)} ${String(r.spearman).padStart(6)} ${r.top3}`)
  }
  out.panels[arch] = { weights: w, rows: rowsOut }
}

writeFileSync('scratch/ablation_components.json', JSON.stringify(out, null, 1))
console.log('\nwrote scratch/ablation_components.json')
