import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'

// Resolve paths from this artifact checkout so the archive runs after download.
const ROOT = fileURLToPath(new URL('../', import.meta.url))
await import(pathToFileURL(path.join(ROOT, 'server/lib/env.js')).href)
const { readRegistry } = await import(pathToFileURL(path.join(ROOT, 'server/lib/registry-store.js')).href)
const { computeBenchmarkBounds, evaluateModelCapabilities } = await import(pathToFileURL(path.join(ROOT, 'server/lib/benchmarks.js')).href)
const { taskQuality } = await import(pathToFileURL(path.join(ROOT, 'server/lib/scoring.js')).href)

const BUDGETS = [0.1, 0.25, 0.5, 1, 2, 5]
const DELTAS = [0, 0.01, 0.02, 0.05]
const TAUS = [0.01, 0.02, 0.05]
const AXES = ['quality', 'costUtility', 'speed', 'context']
const INTENTS = {
  budget: {
    prompt: 'For coding models, prioritize affordability (10/10), quality (8/10), speed (5/10), and context (3/10).',
    parserUsed: 'cohere',
    weights: { quality: 8, costUtility: 10, speed: 5, context: 3 },
  },
  balanced: {
    prompt: 'For coding models, balance quality, affordability, speed, and context equally.',
    parserUsed: 'cohere',
    weights: { quality: 5, costUtility: 5, speed: 5, context: 5 },
  },
}
if (Object.values(INTENTS).some((x) => x.parserUsed !== 'cohere')) throw new Error('Frozen LLM parser output required')

const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const effort = (s) => (/max/i.test(s) ? 3 : /xhigh/i.test(s) ? 2 : /high/i.test(s) ? 1 : 0)
const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null
function rng32(a) {
  return () => {
    a |= 0
    a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return (t ^ (t >>> 14)) >>> 0
  }
}

const registry = await readRegistry()
const snapshot = JSON.parse(readFileSync(path.join(ROOT, 'scratch/arena_snapshot_2026-10-08.json'), 'utf8'))
const index = new Map()
for (const model of registry.models) {
  if (model.aliasOf) continue
  for (const key of [norm(model.name), norm(strip(model.name))]) {
    if (!index.has(key)) index.set(key, [])
    index.get(key).push(model)
  }
}
const groups = new Map()
for (const row of snapshot.models) {
  if (!Number.isFinite(row.net_pct)) continue
  const hits = index.get(norm(row.model)) || index.get(norm(strip(row.model)))
  if (!hits?.length) continue
  const model = hits[0]
  const group = groups.get(model.id) || { model, rows: [] }
  group.rows.push(row)
  groups.set(model.id, group)
}
const bounds = computeBenchmarkBounds(registry.models)
const cohort = [...groups.values()].map(({ model, rows }) => {
  rows.sort((a, b) => effort(b.model) - effort(a.model))
  const arena = rows[0]
  const caps = evaluateModelCapabilities(model, bounds)
  const raw = { ...model, caps, metrics: { ...model.metrics } }
  for (const key of ['quality', 'speed', 'privacy']) {
    if (!Number.isFinite(raw.metrics[key])) raw.metrics[key] = caps[key]
  }
  const q = taskQuality(raw, ['general', 'code'])
  return {
    name: strip(model.name),
    net: arena.net_pct,
    cost: arena.median_usd,
    q,
    speed: Number.isFinite(raw.metrics.speed) ? raw.metrics.speed : 0.05,
    context: Number.isFinite(raw.metrics.context) ? raw.metrics.context : 0.05,
  }
})
if (cohort.length !== 47) throw new Error('Expected 47 models; found ' + cohort.length)

// Freeze full-cohort scaling for all folds; the shared utility does not contain Arena outcome.
const costs = cohort.map((r) => r.cost).filter((v) => v > 0 && Number.isFinite(v))
const minLog = Math.log(Math.min(...costs))
const maxLog = Math.log(Math.max(...costs))
const ranges = {
  quality: [Math.min(...cohort.map((r) => r.q)), Math.max(...cohort.map((r) => r.q))],
  speed: [Math.min(...cohort.map((r) => r.speed)), Math.max(...cohort.map((r) => r.speed))],
  context: [Math.min(...cohort.map((r) => r.context)), Math.max(...cohort.map((r) => r.context))],
}
const scale = (x, [lo, hi]) => hi > lo ? Math.max(0, Math.min(1, (x - lo) / (hi - lo))) : 0
for (const row of cohort) {
  row.x = {
    quality: scale(row.q, ranges.quality),
    costUtility: scale(maxLog - Math.log(row.cost), [0, maxLog - minLog]),
    speed: scale(row.speed, ranges.speed),
    context: scale(row.context, ranges.context),
  }
}
function normalizedWeights(intent) {
  const sum = AXES.reduce((s, key) => s + intent.weights[key], 0)
  return AXES.map((key) => intent.weights[key] / sum)
}
function utility(row, weights) {
  return AXES.reduce((s, key, i) => s + weights[i] * row.x[key], 0)
}
function pick(rows, weights, method, delta = 0) {
  if (method === 'WSM') {
    return rows.reduce((best, row) => utility(row, weights) > utility(best, weights) ? row : best, rows[0])
  }
  if (method === 'CES') {
    const score = (row) => Math.pow(AXES.reduce((s, key, i) => s + weights[i] * Math.sqrt(Math.max(row.x[key], 0.0001)), 0), 2)
    return rows.reduce((best, row) => score(row) > score(best) ? row : best, rows[0])
  }
  if (method === 'TOPSIS') {
    const den = AXES.map((key) => Math.sqrt(rows.reduce((s, row) => s + row.x[key] ** 2, 0)) || 1)
    const v = rows.map((row) => AXES.map((key, i) => row.x[key] / den[i] * weights[i]))
    const hi = AXES.map((_, i) => Math.max(...v.map((row) => row[i])))
    const lo = AXES.map((_, i) => Math.min(...v.map((row) => row[i])))
    const scores = v.map((row) => {
      const dp = Math.sqrt(row.reduce((s, x, i) => s + (x - hi[i]) ** 2, 0))
      const dm = Math.sqrt(row.reduce((s, x, i) => s + (x - lo[i]) ** 2, 0))
      return dm / (dp + dm || 1)
    })
    return rows[scores.indexOf(Math.max(...scores))]
  }
  const qmax = Math.max(...rows.map((r) => r.q))
  const band = rows.filter((r) => r.q >= qmax - delta)
  return band.reduce((best, row) => {
    if (method === 'Quality-band budget') {
      return row.cost < best.cost || (row.cost === best.cost && utility(row, weights) > utility(best, weights)) ? row : best
    }
    return utility(row, weights) > utility(best, weights) ? row : best
  }, band[0])
}
const eligible = (rows, budget) => rows.filter((row) => Number.isFinite(row.cost) && row.cost <= budget)
const summarize = (rows, weights) => ({
  n: rows.length,
  meanUtilityRegret: mean(rows.map((r) => r.oracleUtility - utility(r.pick, weights))),
  meanArenaNetRegret: mean(rows.map((r) => r.oracleNet - r.pick.net)),
  arenaOracleHitRate: mean(rows.map((r) => Math.abs(r.oracleNet - r.pick.net) < 1e-9 ? 1 : 0)),
  meanCostUSD: mean(rows.map((r) => r.pick.cost)),
  meanQuality: mean(rows.map((r) => r.pick.q)),
})

const result = {
  protocol: {
    snapshot: snapshot.payload_last_updated,
    cohortN: cohort.length,
    repeats: 30,
    folds: 5,
    budgetCapsUSD: BUDGETS,
    candidateDeltas: DELTAS,
    utilityRegretCaps: TAUS,
    parser: 'Frozen Cohere-derived profiles recorded in quality_guardrail_eval.js; regex parser not used.',
    utility: 'Weighted arithmetic mean of fixed full-cohort normalized coding quality, log-inverse observed Arena median task cost, speed, and context.',
    hardConstraint: 'For every method, only held-out candidates with observed Arena median task cost <= the budget tier are eligible; budget-tier candidate pools with fewer than three models are omitted.',
    calibration: 'For each outer split, choose delta using training models only: lowest Arena regret among deltas whose mean shared-utility regret is <= tau; then apply the selected delta unchanged to the held-out model fold. Budget chooses lowest cost within the quality band (WSM breaks exact cost ties); balanced chooses highest WSM utility within the band.',
    methods: 'Quality-band (nested), WSM, CES rho=0.5, and TOPSIS. All graded by identical shared intent utility and Arena net improvement.',
    limitations: 'Repeated model-level cross-validation over one previously inspected 47-model snapshot; overlapping folds are descriptive and not independent external validation or user-choice evidence. The training-fold calibration optimizes Arena regret subject to a shared-utility regret cap; report this explicitly.',
    profiles: INTENTS,
  },
  results: {},
}

for (const [intentName, intent] of Object.entries(INTENTS)) {
  const w = normalizedWeights(intent)
  result.results[intentName] = {}
  for (const tau of TAUS) {
    const values = { 'Quality-band (nested)': [], WSM: [], CES: [], TOPSIS: [] }
    const deltaCounts = Object.fromEntries(DELTAS.map((d) => [String(d), 0]))
    let heldoutBudgetDecisions = 0
    const random = rng32(20261013 + (intentName === 'budget' ? 0 : 10000))
    for (let repeat = 0; repeat < 30; repeat++) {
      const order = cohort.map((_, i) => i)
      for (let i = order.length - 1; i > 0; i--) {
        const j = random() % (i + 1)
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      for (let fold = 0; fold < 5; fold++) {
        const heldoutIndices = new Set(order.filter((_, i) => i % 5 === fold))
        const train = cohort.filter((_, i) => !heldoutIndices.has(i))
        const test = cohort.filter((_, i) => heldoutIndices.has(i))
        const trainRecords = []
        for (const budget of BUDGETS) {
          const pool = eligible(train, budget)
          if (pool.length < 3) continue
          const oracleUtility = Math.max(...pool.map((row) => utility(row, w)))
          const oracleNet = Math.max(...pool.map((row) => row.net))
          for (const delta of DELTAS) {
            const chosen = pick(pool, w, intentName === 'budget' ? 'Quality-band budget' : 'Quality-band', delta)
            trainRecords.push({ delta, utilityRegret: oracleUtility - utility(chosen, w), arenaRegret: oracleNet - chosen.net })
          }
        }
        if (!trainRecords.length) continue
        const candidates = DELTAS.map((delta) => {
          const records = trainRecords.filter((r) => r.delta === delta)
          return {
            delta,
            meanUtilityRegret: mean(records.map((r) => r.utilityRegret)),
            meanArenaRegret: mean(records.map((r) => r.arenaRegret)),
          }
        })
        const feasible = candidates.filter((x) => x.meanUtilityRegret <= tau + 1e-12)
        const ranked = [...(feasible.length ? feasible : candidates)].sort((a, b) => feasible.length
          ? a.meanArenaRegret - b.meanArenaRegret || a.delta - b.delta
          : a.meanUtilityRegret - b.meanUtilityRegret || a.meanArenaRegret - b.meanArenaRegret || a.delta - b.delta)
        const delta = ranked[0].delta
        deltaCounts[String(delta)]++
        for (const budget of BUDGETS) {
          const pool = eligible(test, budget)
          if (pool.length < 3) continue
          heldoutBudgetDecisions++
          const oracleUtility = Math.max(...pool.map((row) => utility(row, w)))
          const oracleNet = Math.max(...pool.map((row) => row.net))
          for (const [method, d] of Object.entries({
            'Quality-band (nested)': delta,
            WSM: null,
            CES: null,
            TOPSIS: null,
          })) {
            const selected = method === 'Quality-band (nested)'
              ? pick(pool, w, intentName === 'budget' ? 'Quality-band budget' : 'Quality-band', d)
              : pick(pool, w, method)
            values[method].push({ pick: selected, oracleUtility, oracleNet })
          }
        }
      }
    }
    result.results[intentName][String(tau)] = {
      selectedDeltaCounts: deltaCounts,
      heldoutBudgetDecisions,
      methods: Object.fromEntries(Object.entries(values).map(([method, rows]) => [method, summarize(rows, w)])),
    }
  }
}
writeFileSync(path.join(ROOT, 'scratch/quality_guardrail_nested_cv.json'), JSON.stringify(result, null, 2) + '\n')
console.log(JSON.stringify(result.results, null, 2))
