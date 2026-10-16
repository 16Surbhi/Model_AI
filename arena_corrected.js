/**
 * Corrected Agent Arena cohort study.
 *
 * Bug fixed here: the original extractor (scratch/extract_arena_full.py) matched
 * the payload key order `"isPublic":true,"score":...`, which occurs only inside
 * `snapshot.signals[].entries[]`.  The first signal is `task_outcome_explicit`,
 * displayed as **Confirmed Success**, so every figure in the paper's validation
 * section was computed against that ONE signal instead of the headline
 * net-improvement column (`snapshot.rows[].avgScore.value`).
 *
 * This script grades each aggregator against BOTH columns, on both the 10-02
 * snapshot (Confirmed Success only) and the archived 10-08 snapshot (both), and
 * reports metrics beyond mean regret:
 *
 *   mean regret          -- the paper's headline metric
 *   worst-tier regret    -- minimax regret
 *   oracle hit rate      -- fraction of budget tiers where the oracle is chosen
 *   value captured       -- mean(net of pick) / mean(net of oracle)
 *   Spearman r_s         -- global rank fidelity
 *   Kendall tau-b        -- global rank fidelity (tie-aware)
 *   bootstrap CI         -- 95% interval over 2,000 model resamples, for both
 *                           mean regret and r_s (Remark: regret claims lacked
 *                           intervals and rested on a single cohort)
 *
 * Run: node scratch/arena_corrected.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, taskQuality, weightedAdditiveScore, computeFinalScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'
import { readFileSync, writeFileSync } from 'node:fs'
import { pinnedIntent } from './pinned-intents.js'

const PROMPTS = { frontier: 'best frontier coding models', budget: 'budget coding models', balanced: 'balanced coding models' }
const BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
const LAMBDAS = [0, 2, 5, 10, 20, 30, Infinity]
const MEAN = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)
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
function rankify(v) {
  const idx = v.map((val, i) => [val, i]).sort((p, q) => p[0] - q[0]); const r = new Array(v.length)
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; const avg = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k][1]] = avg; i = j + 1 }
  return r
}
function spearman(x, y) {
  const rx = rankify(x), ry = rankify(y), mx = MEAN(rx), my = MEAN(ry)
  let n = 0, dx = 0, dy = 0
  for (let i = 0; i < rx.length; i++) { n += (rx[i] - mx) * (ry[i] - my); dx += (rx[i] - mx) ** 2; dy += (ry[i] - my) ** 2 }
  return n / Math.sqrt(dx * dy)
}
function kendall(x, y) {
  let c = 0, d = 0, tx = 0, ty = 0
  for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) {
    const a = x[i] - x[j], b = y[i] - y[j]
    if (a === 0 && b === 0) { tx++; ty++ } else if (a === 0) tx++
    else if (b === 0) ty++
    else if (a * b > 0) c++; else d++
  }
  const n0 = c + d + tx + ty
  return (c - d) / Math.sqrt((c + d + tx) * (c + d + ty) || 1)
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

// ---- snapshot normalisation: one schema for both vintages ----------------
function loadSnapshot(path) {
  const raw = JSON.parse(readFileSync(path, 'utf8'))
  if (Array.isArray(raw)) {
    // 10-02 vintage: `score_pct` is the Confirmed Success signal (extractor bug).
    return { meta: { path, column: 'confirmed_success (task_outcome_explicit)', sessions: 2158478, lastUpdated: '2026-10-02' },
      rows: raw.map((r) => ({ rank: r.rank, model: r.model, vendor: r.vendor, net: null, confirmed: r.score_pct, median_usd: r.median_usd, contender: r.contender })) }
  }
  return { meta: { path, column: 'headline net improvement + Confirmed Success', sessions: raw.total_sessions, lastUpdated: raw.payload_last_updated },
    rows: raw.models.map((r) => ({ rank: r.rank, model: r.model, vendor: r.vendor, net: r.net_pct, confirmed: r.confirmed_success_pct, median_usd: r.median_usd, contender: r.contender })) }
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

function buildCohort(snap, column) {
  const groups = new Map()
  for (const a of snap.rows) {
    if (a[column] == null || !Number.isFinite(a[column])) continue
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
    cohort.push({ name: strip(model.name), arena: pick.model, outcome: pick[column], cost: pick.median_usd, q: taskQuality(e, ['general', 'code']), a: e.metrics.affordability, s: e.metrics.speed, raw: e })
  }
  return cohort
}

const out = { probe: {}, runs: [] }

const SNAPSHOTS = [
  { id: '2026-10-02', file: 'scratch/arena_full_2026-10-02.json', columns: ['confirmed'] },
  { id: '2026-10-08', file: 'scratch/arena_snapshot_2026-10-08.json', columns: ['net', 'confirmed'] },
]

for (const s of SNAPSHOTS) {
  const snap = loadSnapshot(s.file)
  for (const column of s.columns) {
    const cohort = buildCohort(snap, column)
    const priced = cohort.filter((c) => c.cost != null && Number.isFinite(c.cost))
    const net = cohort.map((c) => c.outcome)
    const run = {
      snapshot: s.id, snapshot_column: column, snapshot_meta: snap.meta,
      n_cohort: cohort.length, n_priced: priced.length,
      cohort: cohort.map((c) => ({ name: c.name, arena: c.arena, outcome: +c.outcome.toFixed(3), cost: c.cost, q: +c.q.toFixed(4), a: +c.a.toFixed(4), s: +c.s.toFixed(4) })),
      archetypes: {},
    }
    console.log(`\n${'#'.repeat(100)}`)
    console.log(`SNAPSHOT ${s.id}  |  outcome column = ${column}  |  n=${cohort.length} (priced ${priced.length})  |  ${snap.meta.column}`)
    console.log(`Spearman(q, outcome) = ${spearman(net, cohort.map((c) => c.q)).toFixed(3)}`)
    run.spearman_q = +spearman(net, cohort.map((c) => c.q)).toFixed(3)

    for (const [arch, prompt] of Object.entries(PROMPTS)) {
      const { analysis } = await pinnedIntent(prompt)
      const w = analysis.weights, { archetype, targetCategory: category } = analysis
      const flags = { archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: archetype === 'frontier' }
      const W3 = [w.quality, w.affordability, w.speed], wSum = W3.reduce((a, b) => a + b, 0) || 1
      const NW = W3.map((v) => v / wSum)
      const X = cohort.map((c) => [c.q, c.a, c.s])
      const methods = {
        // RANK ON `exact`. `weightedAdditiveScore` returns the ROUNDED display
        // value, and below the soft-ceiling knee the composite's raw range
        // compresses onto a handful of integers, so ranking on it hands the
        // decision to array order instead of to the weighted profile and
        // measures a different rule than the shipped `recommendModels`, which
        // sorts on `scoreExact`. Verified: frontier is unaffected
        // (scores sit above the knee), while the budget and balanced intents
        // move materially.
        'Proposed': cohort.map((c) => computeFinalScore({ ...c.raw, metrics: { ...c.raw.metrics, quality: c.q } }, category, w, flags).exact),
        'Quality only': cohort.map((c) => c.q),
        'WSM': cohort.map((c) => NW[0] * c.q + NW[1] * c.a + NW[2] * c.s),
        'WSM + floor': cohort.map((c) => (c.q >= 0.92 ? NW[0] * c.q + NW[1] * c.a + NW[2] * c.s : -1e9)),
        'TOPSIS': topsis(X, W3),
        'Cobb-Douglas': cohort.map((c) => Math.pow(Math.max(c.q, .05), NW[0]) * Math.pow(Math.max(c.a, .05), NW[1]) * Math.pow(Math.max(c.s, .05), NW[2])),
        'CES only': cohort.map((c) => Math.pow(NW[0] * Math.sqrt(c.q) + NW[1] * Math.sqrt(c.a) + NW[2] * Math.sqrt(c.s), 2)),
      }

      const pickAt = (v, B) => {
        let best = -Infinity, pick = -1
        for (let i = 0; i < cohort.length; i++) if (cohort[i].cost != null && cohort[i].cost <= B && v[i] > best) { best = v[i]; pick = i }
        return pick
      }
      const tiers = []
      for (const B of BUDGETS) {
        const elig = cohort.map((c, i) => i).filter((i) => cohort[i].cost != null && cohort[i].cost <= B)
        if (elig.length < 3) continue
        const oracle = Math.max(...elig.map((i) => net[i]))
        const row = { budget: B, n: elig.length, oracle: +oracle.toFixed(2) }
        for (const [k, v] of Object.entries(methods)) {
          const p = pickAt(v, B)
          row[k] = +(oracle - net[p]).toFixed(2)
          row['_pick_' + k] = cohort[p].name
          row['_net_' + k] = +net[p].toFixed(2)
          row['_cost_' + k] = cohort[p].cost
        }
        tiers.push(row)
      }
      const keys = Object.keys(methods)
      const summary = {}
      for (const k of keys) {
        const regs = tiers.map((r) => r[k])
        const nets = tiers.map((r) => r['_net_' + k])
        const oracles = tiers.map((r) => r.oracle)
        const hits = tiers.filter((r) => r[k] <= 1e-9).length
        summary[k] = {
          mean_regret: +MEAN(regs).toFixed(3),
          worst_tier_regret: +Math.max(...regs).toFixed(3),
          oracle_hits: `${hits}/${tiers.length}`,
          value_captured: +(MEAN(nets) / MEAN(oracles)).toFixed(3),
          spearman: +spearman(net, methods[k]).toFixed(3),
          kendall: +kendall(methods[k], net).toFixed(3),
        }
      }

      // Bootstrap over MODELS (paired): resample cohort indices, recompute the
      // tier oracle and each method's regret inside the resample.
      const rnd = mulberry32(20261011)
      const n = cohort.length
      const acc = {}; for (const k of keys) acc[k] = { reg: [], rho: [] }
      const REPS = 2000
      for (let rep = 0; rep < REPS; rep++) {
        const idx = Array.from({ length: n }, () => Math.floor(rnd() * n))
        const sub = idx.map((i) => cohort[i])
        const subNet = sub.map((c) => c.outcome), subCost = sub.map((c) => c.cost)
        const regs = {}; for (const k of keys) regs[k] = []
        for (const B of BUDGETS) {
          const elig = sub.map((c, i) => i).filter((i) => subCost[i] != null && subCost[i] <= B)
          if (elig.length < 3) continue
          const oracle = Math.max(...elig.map((i) => subNet[i]))
          for (const k of keys) {
            let best = -Infinity, pick = elig[0]
            for (const i of elig) if (methods[k][idx[i]] > best) { best = methods[k][idx[i]]; pick = i }
            regs[k].push(oracle - subNet[pick])
          }
        }
        for (const k of keys) { acc[k].reg.push(MEAN(regs[k])); acc[k].rho.push(spearman(subNet, idx.map((i) => methods[k][i]))) }
      }
      const q = (vals, p) => { const s = vals.filter(Number.isFinite).sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.max(0, Math.floor(p * (s.length - 1))))] }
      for (const k of keys) {
        summary[k].mean_regret_ci = [+q(acc[k].reg, 0.025).toFixed(3), +q(acc[k].reg, 0.975).toFixed(3)]
        summary[k].spearman_ci = [+q(acc[k].rho, 0.025).toFixed(3), +q(acc[k].rho, 0.975).toFixed(3)]
      }
      // Paired bootstrap on the regret DIFFERENCE proposed minus each method:
      // a CI excluding 0 means the gap survives resampling the cohort.
      for (const k of keys) {
        if (k === 'Proposed') continue
        const diffs = acc.Proposed.reg.map((v, i) => v - acc[k].reg[i])
        summary[k].paired_diff_vs_proposed = [+q(diffs, 0.025).toFixed(3), +q(diffs, 0.975).toFixed(3)]
      }
      // Top-k overlap with the outcome ordering (k=3): does the method's
      // shortlist contain the models the outcome ranks highest?
      const topIdx = (v) => v.map((val, i) => [val, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((p) => p[1])
      const outTop = new Set(topIdx(net))
      for (const k of keys) {
        const hitsK = topIdx(methods[k]).filter((i) => outTop.has(i)).length
        summary[k].top3_overlap = `${hitsK}/3`
      }
      // Normalised regret: mean regret as a fraction of the mean spread between
      // the oracle and the weakest admissible model in each tier.
      const normReg = {}
      for (const k of keys) {
        const fracs = []
        for (const B of BUDGETS) {
          const elig = cohort.map((c, i) => i).filter((i) => cohort[i].cost != null && cohort[i].cost <= B)
          if (elig.length < 3) continue
          const vals = elig.map((i) => net[i])
          const span = Math.max(...vals) - Math.min(...vals)
          if (span > 1e-9) fracs.push(tiers.find((r) => r.budget === B)[k] / span)
        }
        normReg[k] = +MEAN(fracs).toFixed(3)
      }
      for (const k of keys) summary[k].normalised_regret = normReg[k]

      // lambda sweep: mean regret against u_lambda
      const sweep = {}
      for (const l of LAMBDAS) {
        const u = l === Infinity ? net.map((v, i) => v / Math.max(cohort[i].cost ?? 1e-9, 0.01)) : net.map((v, i) => v - l * (cohort[i].cost ?? 0))
        const per = {}
        for (const [k, v] of Object.entries(methods)) {
          const regs = []
          for (const B of BUDGETS) {
            const elig = cohort.map((c, i) => i).filter((i) => cohort[i].cost != null && cohort[i].cost <= B)
            if (elig.length < 3) continue
            const oracle = Math.max(...elig.map((i) => u[i]))
            let best = -Infinity, pick = elig[0]
            for (const i of elig) if (v[i] > best) { best = v[i]; pick = i }
            regs.push(oracle - u[pick])
          }
          per[k] = +MEAN(regs).toFixed(2)
        }
        sweep[l === Infinity ? 'inf' : String(l)] = per
      }

      // value for money: net, cost, net/$, dominated tiers
      const vfm = {}
      for (const [k, v] of Object.entries(methods)) {
        const effs = []; let dominated = 0, tiersN = 0
        for (const B of BUDGETS) {
          const elig = cohort.map((c, i) => i).filter((i) => cohort[i].cost != null && cohort[i].cost <= B)
          if (elig.length < 3) continue
          const p = pickAt(v, B)
          tiersN++
          effs.push(net[p] / cohort[p].cost)
          if (elig.some((j) => j !== p && net[j] >= net[p] && cohort[j].cost <= cohort[p].cost && (net[j] > net[p] || cohort[j].cost < cohort[p].cost))) dominated++
        }
        vfm[k] = { net_per_usd: +MEAN(effs).toFixed(2), dominated: `${dominated}/${tiersN}` }
      }

      run.archetypes[arch] = { prompt, parserUsed: analysis.parserUsed, weights: { quality: w.quality, affordability: w.affordability, speed: w.speed }, tiers, summary, sweep, vfm, picks: tiers.map((r) => ({ budget: r.budget, proposed: r._pick_Proposed, net: r._net_Proposed, cost: r._cost_Proposed, oracle: r.oracle, ref: r._pick_Quality })) }

      console.log(`\n--- ${arch.toUpperCase()}  "${prompt}"  (parser=${analysis.parserUsed}, w_q=${w.quality} w_a=${w.affordability} w_s=${w.speed}) ---`)
      console.log('method'.padEnd(16) + 'meanReg'.padStart(9) + '95% CI'.padStart(20) + 'worst'.padStart(8) + 'hits'.padStart(8) + 'valCapt'.padStart(9) + 'rho'.padStart(8) + 'rho CI'.padStart(18) + 'tau'.padStart(8))
      for (const k of keys) {
        const s = summary[k]
        console.log(k.padEnd(16) + s.mean_regret.toFixed(2).padStart(9) + `[${s.mean_regret_ci[0]},${s.mean_regret_ci[1]}]`.padStart(20) + s.worst_tier_regret.toFixed(2).padStart(8) + s.oracle_hits.padStart(8) + s.value_captured.toFixed(3).padStart(9) + s.spearman.toFixed(3).padStart(8) + `[${s.spearman_ci[0]},${s.spearman_ci[1]}]`.padStart(18) + s.kendall.toFixed(3).padStart(8))
      }
      console.log('  lambda sweep (Proposed / Quality-only): ' + Object.entries(sweep).map(([l, p]) => `l=${l}:${p.Proposed}/${p['Quality only']}`).join('  '))
      console.log('  value for money: ' + Object.entries(vfm).map(([k, v]) => `${k}=${v.net_per_usd}$(${v.dominated})`).join('  '))
    }
    out.runs.push(run)
  }
}
writeFileSync('scratch/arena_corrected.json', JSON.stringify(out, null, 2))
console.log('\nwrote scratch/arena_corrected.json')
