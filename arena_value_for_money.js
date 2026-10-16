/**
 * Parameter-free cost-aware evaluation of the same picks.
 *
 * The paper's headline metric (regret vs a cost-blind oracle) cannot reward
 * cost awareness by construction. This script instead grades each method on
 * two quantities that are MEASURED, not scalarised:
 *   1. the net improvement actually delivered by its recommendation, and
 *   2. the net improvement per dollar of that recommendation (value for money),
 * plus a parameter-free efficiency test:
 *   3. whether the recommended model is Pareto-dominated on the measured
 *      (net improvement, cost) plane by another eligible model -- i.e. whether
 *      a cheaper model that scores at least as well was available and passed over.
 *
 * Run: node scratch/arena_value_for_money.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, taskQuality, weightedAdditiveScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'
import { readFileSync, writeFileSync } from 'node:fs'

const arena = JSON.parse(readFileSync('scratch/arena_full_2026-10-02.json', 'utf8'))
const PROMPTS = { frontier: 'best frontier coding models', budget: 'budget coding models', balanced: 'balanced coding models' }
const BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const MEAN = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)

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
const EFFORT = (n) => (/max/i.test(n) ? 3 : /xhigh/i.test(n) ? 2 : /high/i.test(n) ? 1 : 0)
function buildCohort(rows0) {
  const groups = new Map()
  for (const a of rows0) {
    const hits = regIdx.get(norm(a.model)) || regIdx.get(norm(strip(a.model)))
    if (!hits || !hits.length) continue
    const m = hits[0]
    const g = groups.get(m.id) || { model: m, arena: [] }
    g.arena.push(a); groups.set(m.id, g)
  }
  const cohort = []
  for (const { model, arena: rows } of groups.values()) {
    rows.sort((a, b) => EFFORT(b.model) - EFFORT(a.model))
    const pick = rows[0]
    const caps = evaluateModelCapabilities(model, bounds)
    const e = { ...model, caps, metrics: { ...model.metrics } }
    for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
    cohort.push({ name: strip(model.name), net: pick.score_pct, cost: pick.median_usd, q: taskQuality(e, ['general', 'code']), a: e.metrics.affordability, s: e.metrics.speed, raw: e })
  }
  return cohort
}

const COHORTS = {
  'n=18 (rank-ordered head)': buildCohort([...arena].sort((a, b) => a.rank - b.rank).slice(0, 20)),
  'n=45 (full leaderboard)': buildCohort(arena),
}
const out = { cohorts: {} }

for (const [cohortLabel, cohort] of Object.entries(COHORTS)) {
  console.log(`\n############################## COHORT ${cohortLabel}: n=${cohort.length}  cost range $${Math.min(...cohort.map(c=>c.cost)).toFixed(4)}-$${Math.max(...cohort.map(c=>c.cost)).toFixed(2)}`)
  out.cohorts[cohortLabel] = { cohort_n: cohort.length, archetypes: {} }

for (const [arch, prompt] of Object.entries(PROMPTS)) {
  const { analysis } = await recommendModels(reg.models, { prompt, limit: 10 })
  const w = analysis.weights, { archetype, targetCategory: category } = analysis
  const flags = { archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: archetype === 'frontier' }
  const W3 = [w.quality, w.affordability, w.speed], wSum = W3.reduce((a, b) => a + b, 0) || 1
  const NW = W3.map((v) => v / wSum)
  const methods = {
    'Proposed': cohort.map((c) => weightedAdditiveScore({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, flags)),
    'Quality only': cohort.map((c) => c.q),
    'WSM': cohort.map((c) => NW[0] * c.q + NW[1] * c.a + NW[2] * c.s),
    'WSM + floor': cohort.map((c) => (c.q >= 0.92 ? NW[0] * c.q + NW[1] * c.a + NW[2] * c.s : -1e9)),
    'TOPSIS': null, // filled below
    'Cobb-Douglas': cohort.map((c) => Math.pow(Math.max(c.q, .05), NW[0]) * Math.pow(Math.max(c.a, .05), NW[1]) * Math.pow(Math.max(c.s, .05), NW[2])),
    'CES only': cohort.map((c) => Math.pow(NW[0] * Math.sqrt(c.q) + NW[1] * Math.sqrt(c.a) + NW[2] * Math.sqrt(c.s), 2)),
  }
  // TOPSIS over the same normalised criterion matrix.
  const M = cohort.map((c) => [c.q, c.a, c.s])
  const vnorm = [0, 1, 2].map((j) => Math.sqrt(M.reduce((s, r) => s + r[j] * r[j], 0)))
  const Z = M.map((r) => r.map((v, j) => v / (vnorm[j] || 1)))
  const best = [0, 1, 2].map((j) => Math.max(...Z.map((r) => r[j])))
  const worst = [0, 1, 2].map((j) => Math.min(...Z.map((r) => r[j])))
  methods['TOPSIS'] = Z.map((r) => {
    const dp = Math.sqrt(r.reduce((s, v, j) => s + NW[j] * (v - best[j]) ** 2, 0))
    const dn = Math.sqrt(r.reduce((s, v, j) => s + NW[j] * (v - worst[j]) ** 2, 0))
    return dn / ((dp + dn) || 1)
  })

  console.log(`\n===== ${arch.toUpperCase()} intent (${prompt}) =====`)
  console.log('method'.padEnd(16) + 'net'.padStart(9) + 'cost'.padStart(10) + 'net/$'.padStart(10) + 'dominated'.padStart(12))
  const res = {}
  for (const [k, v] of Object.entries(methods)) {
    const nets = [], costs = [], effs = []
    let dominated = 0, tiers = 0
    for (const B of BUDGETS) {
      const elig = cohort.map((c, i) => i).filter((i) => cohort[i].cost <= B)
      if (elig.length < 3) continue
      let bv = -Infinity, pick = elig[0]
      for (const i of elig) if (v[i] > bv) { bv = v[i]; pick = i }
      tiers++
      const p = cohort[pick]
      nets.push(p.net); costs.push(p.cost); effs.push(p.net / p.cost)
      // Pareto-dominated on the MEASURED (net, cost) plane among eligible models?
      const dom = elig.some((j) => j !== pick && cohort[j].net >= p.net && cohort[j].cost <= p.cost && (cohort[j].net > p.net || cohort[j].cost < p.cost))
      if (dom) dominated++
    }
    res[k] = { net: +MEAN(nets).toFixed(3), cost: +MEAN(costs).toFixed(4), net_per_usd: +MEAN(effs).toFixed(2), dominated_tiers: dominated, tiers }
    console.log(k.padEnd(16) + MEAN(nets).toFixed(2).padStart(9) + ('$' + MEAN(costs).toFixed(3)).padStart(10) + MEAN(effs).toFixed(1).padStart(10) + `${dominated}/${tiers}`.padStart(12))
  }
  out.cohorts[cohortLabel].archetypes[arch] = { prompt, weights: w, results: res }
}
}
writeFileSync('scratch/arena_value_for_money.json', JSON.stringify(out, null, 2))
console.log('\nwrote scratch/arena_value_for_money.json')
