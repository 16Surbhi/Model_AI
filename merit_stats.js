/**
 * Merit statistics that can actually test preference integration.
 *
 * WHY THE OLD HEADLINE COULD NOT:
 * mean regret is measured against the Agent Arena net-improvement column, which
 * prices neither dollars nor latency. Task quality correlates with that column
 * at ~0.84, so the regret-minimising ordering of an admissible set IS the
 * quality ordering, and any rule that spends weight on cost or speed forgoes
 * regret *by construction*. Reporting it as the primary result therefore cannot
 * separate a cost-aware method from a cost-blind one, and the observed "tie"
 * under a capability-first intent is a property of the reference, not of the
 * method.
 *
 * WHAT REPLACES IT:
 *
 * H1 (primary, one anchored parameter) -- UTILITY REGRET at the market's own
 *    exchange rate. u(m) = net(m) - lambda * cost(m). The scalarisation needs an
 *    exchange rate; instead of choosing one, we MEASURE it from the cohort's own
 *    (cost, net) Pareto frontier: lambda* is the median slope of the frontier,
 *    i.e. the number of net-improvement points the market actually charges per
 *    dollar. The oracle maximises u at each tier, so it prices cost exactly as
 *    the user's budget request does, and a cost-aware method can win. `lambda`
 *    stops being a free knob and becomes a property of the market.
 *
 * H2 (secondary, NO free parameter) -- COST PREMIUM TO TARGET. Fix a target
 *    outcome per tier (95% of the tier oracle, declared in advance); c_min is the
 *    cheapest eligible model that reaches it. Premium = (cost_pick - c_min)/c_min,
 *    and a target the pick fails to reach is counted as a miss. This answers the
 *    practitioner's question directly ("what do I overpay for the same measured
 *    improvement?") and has no tunable constant at all.
 *
 * H3 (robustness) -- the VICTORY INTERVAL: the range of lambda over which the
 *    method has the lowest utility regret among integrative rules. A point
 *    estimate hides the choice of lambda; an interval shows whether the claim
 *    survives any plausible exchange rate.
 *
 * H4 (significance, small-n honest) -- paired bootstrap over models for the
 *    utility DIFFERENCE, plus a per-tier win/loss/tie tally.
 *
 * The cost-blind mean regret is retained but demoted to a reference BOUND: it
 * certifies that the method never does worse than the rules it replaces. It is
 * not a test of preference integration and must not be read as one.
 *
 * Run: node scratch/merit_stats.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, taskQuality, computeFinalScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'
import { readFileSync } from 'node:fs'
import { pinnedIntent } from './pinned-intents.js'

const PROMPTS = { frontier: 'best frontier coding models', budget: 'budget coding models', balanced: 'balanced coding models' }
const BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
const TARGET_FRACTION = 0.95
const MEAN = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)
const MEDIAN = (v) => { const s = [...v].sort((a, b) => a - b); if (!s.length) return NaN; const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2 }
const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const EFFORT = (n) => (/max/i.test(n) ? 3 : /xhigh/i.test(n) ? 2 : /high/i.test(n) ? 1 : 0)

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
function topsis(X, w) {
  const wSum = w.reduce((a, b) => a + b, 0) || 1
  const nrm = [0, 1, 2].map((j) => Math.sqrt(X.reduce((t, r) => t + r[j] * r[j], 0)))
  const V = X.map((r) => r.map((v, j) => (v / (nrm[j] || 1)) * (w[j] / wSum)))
  const best = [0, 1, 2].map((j) => Math.max(...V.map((r) => r[j])))
  const worst = [0, 1, 2].map((j) => Math.min(...V.map((r) => r[j])))
  return V.map((r) => {
    const dp = Math.sqrt(r.reduce((t, v, j) => t + (v - best[j]) ** 2, 0))
    const dm = Math.sqrt(r.reduce((t, v, j) => t + (v - worst[j]) ** 2, 0))
    return dm / (dp + dm || 1)
  })
}

function loadSnapshot(path) {
  const raw = JSON.parse(readFileSync(path, 'utf8'))
  if (Array.isArray(raw)) {
    return { rows: raw.map((r) => ({ model: r.model, net: r.score_pct, median_usd: r.median_usd })) }
  }
  return { rows: raw.models.map((r) => ({ model: r.model, net: r.net_pct, median_usd: r.median_usd })) }
}

/**
 * Market exchange rate: the slope of the cohort's own (cost, net) Pareto
 * frontier. Positive slopes only (a frontier point that costs more must buy more
 * outcome); the median slope is the marginal price of one net-improvement point.
 */
function marketLambda(cohort) {
  const pts = cohort.filter((c) => Number.isFinite(c.cost) && Number.isFinite(c.outcome))
  const frontier = pts.filter((a) => !pts.some((b) => b !== a && b.cost <= a.cost && b.outcome >= a.outcome && (b.cost < a.cost || b.outcome > a.outcome)))
  frontier.sort((a, b) => a.cost - b.cost)
  const slopes = []
  for (let i = 1; i < frontier.length; i++) {
    const dc = frontier[i].cost - frontier[i - 1].cost
    const dn = frontier[i].outcome - frontier[i - 1].outcome
    if (dc > 0 && dn > 0) slopes.push(dn / dc)
  }
  return { lambda: MEDIAN(slopes), frontierSize: frontier.length, slopes }
}

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

function buildCohort(snap) {
  const groups = new Map()
  for (const a of snap.rows) {
    if (a.net == null || !Number.isFinite(a.net)) continue
    const hits = regIdx.get(norm(a.model)) || regIdx.get(norm(strip(a.model)))
    if (!hits || !hits.length) continue
    const m = hits[0]
    const g = groups.get(m.id) || { model: m, rows: [] }
    g.rows.push(a); groups.set(m.id, g)
  }
  const cohort = []
  for (const { model, rows } of groups.values()) {
    rows.sort((a, b) => EFFORT(b.model) - EFFORT(a.model))
    const pick = rows[0]
    const caps = evaluateModelCapabilities(model, bounds)
    const e = { ...model, caps, metrics: { ...model.metrics } }
    for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
    cohort.push({ name: strip(model.name), outcome: pick.net, cost: pick.median_usd, q: taskQuality(e, ['general', 'code']), a: e.metrics.affordability, s: e.metrics.speed, raw: e })
  }
  return cohort
}

const snaps = [{ id: '2026-10-08', file: 'scratch/arena_snapshot_2026-10-08.json' }]
const results = {}

for (const s of snaps) {
  const cohort = buildCohort(loadSnapshot(s.file)).filter((c) => Number.isFinite(c.cost))
  const { lambda, frontierSize, slopes } = marketLambda(cohort)
  const net = cohort.map((c) => c.outcome)
  const u = (i, lam) => net[i] - lam * cohort[i].cost

  console.log('='.repeat(104))
  console.log(`COHORT ${s.id}  |  n=${cohort.length} priced  |  net-improvement column`)
  console.log(`ANCHORED EXCHANGE RATE lambda* = ${lambda.toFixed(3)} net-points per dollar`)
  console.log(`  (median slope of the ${frontierSize}-point (cost, net) Pareto frontier; ${slopes.length} positive slopes, ` +
    `range ${Math.min(...slopes).toFixed(2)}-${Math.max(...slopes).toFixed(2)})`)
  console.log('='.repeat(104))

  for (const [arch, prompt] of Object.entries(PROMPTS)) {
    const { analysis } = await pinnedIntent(prompt)
    const w = analysis.weights
    const { archetype, targetCategory: category } = analysis
    const flags = { archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: archetype === 'frontier' }
    const W3 = [w.quality, w.affordability, w.speed]
    const wSum = W3.reduce((a, b) => a + b, 0) || 1
    const NW = W3.map((v) => v / wSum)
    const X = cohort.map((c) => [c.q, c.a, c.s])
    const methods = {
      // RANK ON `exact`. `weightedAdditiveScore` returns the ROUNDED display
      // value, and above the soft-ceiling knee the tail compresses a wide range
      // onto a handful of integers, so ranking on it hands the decision to array
      // order rather than to the weighted profile. `computeFinalScore` documents
      // the rule and `recommendModels` obeys it by sorting on `scoreExact`; a
      // harness that ranked on `.score` was measuring a different rule than the
      // one it reported, and reported ties that were artefacts of rounding.
      'Proposed': cohort.map((c) => computeFinalScore({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, flags).exact),
      'Quality only': cohort.map((c) => c.q),
      'WSM': cohort.map((c) => NW[0] * c.q + NW[1] * c.a + NW[2] * c.s),
      'WSM + floor': cohort.map((c) => (c.q >= 0.92 ? NW[0] * c.q + NW[1] * c.a + NW[2] * c.s : -1e9)),
      'TOPSIS': topsis(X, W3),
      'Cobb-Douglas': cohort.map((c) => Math.pow(Math.max(c.q, .05), NW[0]) * Math.pow(Math.max(c.a, .05), NW[1]) * Math.pow(Math.max(c.s, .05), NW[2])),
      'CES only': cohort.map((c) => Math.pow(NW[0] * Math.sqrt(c.q) + NW[1] * Math.sqrt(c.a) + NW[2] * Math.sqrt(c.s), 2)),
    }
    const keys = Object.keys(methods)
    const eligibleAt = (B, subset) => subset.filter((i) => cohort[i].cost <= B)
    const pickAt = (v, B, subset) => {
      let best = -Infinity, pick = -1
      for (const i of subset) if (cohort[i].cost <= B && v[i] > best) { best = v[i]; pick = i }
      return pick
    }

    // ── H1: utility regret at the TIER SHADOW PRICE ────────────────────────
    // A single global exchange rate is not identifiable here: the cohort's
    // frontier slopes span three orders of magnitude (0.65 to 2070), so their
    // median is not a price anyone pays, and at a large lambda utility is
    // dominated by cost alone — which rewards picking the cheapest admissible
    // model and stops testing the trade-off at all (a pure cost-minimiser scored
    // a perfect 0.000). The identifiable quantity is the SHADOW PRICE OF THE
    // BUDGET: at tier B the best admissible model delivers oracleNet, so money
    // is worth lambda_B = oracleNet / B net-points per dollar at this tier. That
    // is derived from the tier's own oracle, differs per tier as budgets bind,
    // and cannot be pushed to a degenerate limit by choosing lambda.
    const tiers = []
    for (const B of BUDGETS) {
      const elig = eligibleAt(B, cohort.map((_, i) => i))
      if (elig.length < 3) continue
      const oracleNet = Math.max(...elig.map((i) => net[i]))
      const lamB = oracleNet > 0 ? oracleNet / B : 0
      const uB = (i) => net[i] - lamB * cohort[i].cost
      const oracleU = Math.max(...elig.map((i) => uB(i)))
      const target = TARGET_FRACTION * oracleNet
      const reach = elig.filter((i) => net[i] >= target)
      const cMin = reach.length ? Math.min(...reach.map((i) => cohort[i].cost)) : null
      const row = { B, n: elig.length, lamB, oracleU, oracleNet, target, cMin, elig }
      for (const k of keys) {
        const p = pickAt(methods[k], B, elig)
        row[k] = oracleU - uB(p)
        row['net_' + k] = net[p]
        row['cost_' + k] = cohort[p].cost
        row['name_' + k] = cohort[p].name
      }
      tiers.push(row)
    }

    const summary = {}
    for (const k of keys) {
      const urs = tiers.map((r) => r[k])
      const overpay = [], misses = []
      for (const r of tiers) {
        const reached = r['net_' + k] >= r.target
        if (reached && r.cMin) overpay.push((r['cost_' + k] - r.cMin) / r.cMin)
        if (!reached) misses.push(r.B)
      }
      // Reported as a PAIR, never as a bare average: a method that always takes
      // the cheapest admissible model shows a 0% premium and misses every
      // target, so the premium is only meaningful beside the hit rate.
      summary[k] = {
        ur_mean: MEAN(urs),
        ur_worst: Math.max(...urs),
        hits: tiers.filter((r) => r[k] <= 1e-9).length,
        targetHits: tiers.length - misses.length,
        tiers: tiers.length,
        overpay: overpay.length ? MEDIAN(overpay) : null,
        overpayN: overpay.length,
        misses: misses.length,
      }
    }

    // ── H3: victory interval over a declared lambda grid ───────────────────
    const GRID = [0, 0.5, 1, 2, 5, 10, 20, lambda, 2 * lambda, 5 * lambda, 10 * lambda, 30]
    const grid = [...new Set(GRID.filter((v) => Number.isFinite(v)))].sort((a, b) => a - b)
    const bestAt = {}
    for (const lam of grid) {
      const means = {}
      for (const k of keys) {
        const rs = tiers.map((r) => {
          const elig = r.elig
          const oU = Math.max(...elig.map((i) => u(i, lam)))
          const p = pickAt(methods[k], r.B, elig)
          return oU - u(p, lam)
        })
        means[k] = MEAN(rs)
      }
      const integrative = keys.filter((k) => k !== 'Quality only')
      const best = integrative.reduce((a, b) => (means[b] < means[a] ? b : a))
      // A TIE is not a win. Every aggregation rule is given the method's own
      // weight vector, so on a no-preference request they can collapse onto the
      // same ordering; counting that as a victory would claim a discrimination
      // the request never asked for.
      const bestValue = means[best]
      const tiedWith = integrative.filter((k) => k !== best && Math.abs(means[k] - bestValue) <= 1e-9)
      bestAt[lam] = { best, means, tiedWith }
    }
    const ownWins = grid.filter((lam) => bestAt[lam].best === 'Proposed')
    const ownStrict = ownWins.filter((lam) => bestAt[lam].tiedWith.length === 0)
    const ownTies = ownWins.length - ownStrict.length

    // ── H4: paired bootstrap over models for the utility difference ────────
    const rnd = mulberry32(20261011)
    const REPS = 2000
    const n = cohort.length
    const diffs = {}; for (const k of keys) if (k !== 'Proposed') diffs[k] = []
    for (let rep = 0; rep < REPS; rep++) {
      const idx = Array.from({ length: n }, () => Math.floor(rnd() * n))
      const subNet = idx.map((i) => net[i])
      const subCost = idx.map((i) => cohort[i].cost)
      const subScore = {}
      for (const k of keys) subScore[k] = idx.map((i) => methods[k][i])
      const acc = {}; for (const k of keys) acc[k] = []
      for (const B of BUDGETS) {
        const elig = subCost.map((c, i) => i).filter((i) => subCost[i] <= B)
        if (elig.length < 3) continue
        // Same per-tier shadow price as the point estimate. Reusing the global
        // lambda here measured a different objective than the table reported and
        // inverted the sign of the budget-vs-quality-only difference.
        const oracleNet = Math.max(...elig.map((i) => subNet[i]))
        const lamB = oracleNet > 0 ? oracleNet / B : 0
        const subU = idx.map((i) => subNet[i] - lamB * subCost[i])
        const oU = Math.max(...elig.map((i) => subU[i]))
        for (const k of keys) {
          let best = -Infinity, pick = -1
          for (const i of elig) if (subScore[k][i] > best) { best = subScore[k][i]; pick = i }
          acc[k].push(oU - subU[pick])
        }
      }
      // MEAN, to match the reported point estimate (ur_mean). A median over
      // ~6 tiers per replication is too coarse and produced degenerate
      // intervals such as [0.000, 0.000] that hide the true spread.
      for (const k of keys) if (k !== 'Proposed') diffs[k].push(MEAN(acc.Proposed) - MEAN(acc[k]))
    }
    const ci = {}
    for (const k of Object.keys(diffs)) {
      const v = diffs[k].sort((a, b) => a - b)
      ci[k] = [v[Math.floor(0.025 * v.length)], v[Math.floor(0.975 * v.length)]]
    }
    const excludesZero = Object.entries(ci).filter(([, iv]) => iv[0] > 0 || iv[1] < 0).map(([k]) => k)

    results[arch] = { weights: W3, lambda, lambdaB: tiers.map((r) => r.lamB), summary, ownWins: ownWins.length, ownStrict: ownStrict.length, ownTies, gridSize: grid.length, ci, excludesZero }

    console.log(`\n--- ${arch.toUpperCase()}  "${prompt}"  (w_q=${W3[0]} w_a=${W3[1]} w_s=${W3[2]}) ---`)
    console.log('  tier shadow prices lambda_B: ' + tiers.map((r) => `${r.B}=${r.lamB.toFixed(1)}`).join('  '))
    console.log('method            UR_shad  worst   oracle  target   overpay|n   misses   dUR vs proposed [95% CI]')
    for (const k of keys) {
      const s2 = summary[k]
      const d = ci[k] ? `[${ci[k][0].toFixed(3)}, ${ci[k][1].toFixed(3)}]` : '----'
      console.log([
        k.padEnd(17),
        s2.ur_mean.toFixed(3).padStart(6),
        s2.ur_worst.toFixed(3).padStart(7),
        `${s2.hits}/${s2.tiers}`.padStart(7),
        `${s2.targetHits}/${s2.tiers}`.padStart(6),
        (s2.overpay === null ? '  n/a' : `${(s2.overpay * 100).toFixed(1)}%|${s2.overpayN}`).padStart(11),
        String(s2.misses).padStart(8),
        d,
      ].join('  '))
    }
    console.log(`victory interval: proposed has the lowest integrative utility regret for ${ownWins.length}/${grid.length} tested lambda ` +
      `(${ownStrict.length} strictly, ${ownTies} tied with every other rule) ` +
      `(${grid[0]} to ${grid[grid.length - 1]})`)
    console.log(`paired intervals excluding zero: ${excludesZero.length ? excludesZero.join(', ') : 'NONE'}`)
  }
}

console.log('\n\n' + '='.repeat(104))
console.log('VERIFICATION DIGEST (machine-readable)')
console.log('='.repeat(104))
const digest = {}
for (const [arch, r] of Object.entries(results)) {
  digest[arch] = {
    lambdaB: r.lambdaB.map((v) => +v.toFixed(2)),
    victory: `${r.ownWins}/${r.gridSize}`,
    excludesZero: r.excludesZero,
    methods: Object.fromEntries(Object.entries(r.summary).map(([k, s2]) => [k, {
      ur: +s2.ur_mean.toFixed(3), worst: +s2.ur_worst.toFixed(3),
      oracleHits: `${s2.hits}/${s2.tiers}`, targetHits: `${s2.targetHits}/${s2.tiers}`, misses: s2.misses,
      ci: r.ci[k] ? r.ci[k].map((v) => +v.toFixed(3)) : null,
    }])),
  }
}
console.log(JSON.stringify(digest, null, 1))
