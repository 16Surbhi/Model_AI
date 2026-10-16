/**
 * Offline intent-aligned comparison on the frozen Arena cohort.
 * All rankers see the same four inputs and hard cost-filtered candidate sets.
 * The ranking score is not used to grade itself: each selection is graded by
 * the same fixed weighted utility for its intent.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { readRegistry } from '../../server/lib/registry-store.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../../server/lib/benchmarks.js'
import { computeFinalScore, taskQuality } from '../../server/lib/scoring.js'

const BUDGETS = [0.1, 0.25, 0.5, 1, 2, 5]
const AXES = ['quality', 'costUtility', 'speed', 'context']
const INTENTS = {
  budget: {
    prompt: 'budget coding models', parserUsed: 'cohere',
    // Live API profile: affordability 2.5, quality 0.85, speed 0.60, context 0.65.
    // Renormalized over the four evaluated attributes; other dimensions were
    // not specified by this prompt and are not in this utility.
    weights: { quality: 0.85, costUtility: 2.5, speed: 0.6, context: 0.65 },
  },
  balanced: {
    prompt: 'balanced coding models', parserUsed: 'cohere',
    // Explicit evaluation assumption: the request balances four named
    // dimensions equally. The live parser supplied no explicit weights.
    weights: { quality: 1, costUtility: 1, speed: 1, context: 1 },
  },
}
const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const effort = (s) => (/max/i.test(s) ? 3 : /xhigh/i.test(s) ? 2 : /high/i.test(s) ? 1 : 0)
const mean = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN

function buildCohort(registry, snapshot) {
  const index = new Map()
  for (const m of registry.models) {
    if (m.aliasOf) continue
    for (const key of [norm(m.name), norm(strip(m.name))]) {
      if (!index.has(key)) index.set(key, [])
      index.get(key).push(m)
    }
  }
  const groups = new Map()
  for (const row of snapshot.models) {
    if (!Number.isFinite(row.net_pct)) continue
    const hits = index.get(norm(row.model)) || index.get(norm(strip(row.model)))
    if (!hits?.length) continue
    const model = hits[0]
    const group = groups.get(model.id) || { model, rows: [] }
    group.rows.push(row); groups.set(model.id, group)
  }
  const bounds = computeBenchmarkBounds(registry.models)
  const cohort = [...groups.values()].map(({ model, rows }) => {
    rows.sort((a, b) => effort(b.model) - effort(a.model))
    const arena = rows[0]
    const caps = evaluateModelCapabilities(model, bounds)
    const raw = { ...model, caps, metrics: { ...model.metrics } }
    for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(raw.metrics[k])) raw.metrics[k] = caps[k]
    const q = taskQuality(raw, ['general', 'code'])
    return {
      name: strip(model.name), outcome: arena.net_pct, cost: arena.median_usd,
      q, speed: Number.isFinite(raw.metrics.speed) ? raw.metrics.speed : 0.05,
      context: Number.isFinite(raw.metrics.context) ? raw.metrics.context : 0.05,
      raw,
    }
  })
  if (cohort.length !== 47) throw new Error(`Expected frozen cohort n=47, got ${cohort.length}`)

  // Fixed full-cohort normalisation. It does not change with each budget tier.
  const costs = cohort.map((r) => r.cost).filter((x) => Number.isFinite(x) && x > 0)
  const minLogCost = Math.log(Math.min(...costs)), maxLogCost = Math.log(Math.max(...costs))
  const minQ = Math.min(...cohort.map((r) => r.q)), maxQ = Math.max(...cohort.map((r) => r.q))
  const minS = Math.min(...cohort.map((r) => r.speed)), maxS = Math.max(...cohort.map((r) => r.speed))
  const minC = Math.min(...cohort.map((r) => r.context)), maxC = Math.max(...cohort.map((r) => r.context))
  const scale = (x, lo, hi) => hi > lo ? (x - lo) / (hi - lo) : 0
  for (const r of cohort) {
    r.x = {
      quality: scale(r.q, minQ, maxQ),
      costUtility: scale(maxLogCost - Math.log(r.cost), 0, maxLogCost - minLogCost),
      speed: scale(r.speed, minS, maxS),
      context: scale(r.context, minC, maxC),
    }
  }
  return { cohort, normalization: { cost: '1 - log(cost/min_cost)/log(max_cost/min_cost)', fixedCohortRange: true, minCost: Math.min(...costs), maxCost: Math.max(...costs), qualityRange: [minQ, maxQ], speedRange: [minS, maxS], contextRange: [minC, maxC] } }
}

function normalizedWeights(weights) {
  const sum = AXES.reduce((s, k) => s + weights[k], 0)
  return AXES.map((k) => weights[k] / sum)
}
function wsm(rows, w) { return rows.map((r) => AXES.reduce((s, k, j) => s + w[j] * r.x[k], 0)) }
function powerMean(rows, w, rho) {
  return rows.map((r) => Math.pow(AXES.reduce((s, k, j) => s + w[j] * Math.pow(Math.max(r.x[k], 0.0001), rho), 0), 1 / rho))
}
function topsis(rows, weights) {
  const denom = AXES.map((k) => Math.sqrt(rows.reduce((s, r) => s + r.x[k] ** 2, 0)) || 1)
  const v = rows.map((r) => AXES.map((k, j) => r.x[k] / denom[j] * weights[j]))
  const best = AXES.map((_, j) => Math.max(...v.map((r) => r[j])))
  const worst = AXES.map((_, j) => Math.min(...v.map((r) => r[j])))
  return v.map((r) => {
    const dBest = Math.sqrt(r.reduce((s, x, j) => s + (x - best[j]) ** 2, 0))
    const dWorst = Math.sqrt(r.reduce((s, x, j) => s + (x - worst[j]) ** 2, 0))
    return dWorst / (dBest + dWorst || 1)
  })
}
function proposed(rows, intent) {
  // Re-map only the affordability input to the observed, task-level cost
  // utility. This harmonized variant lets us isolate the scorer from the
  // catalogue-price vs. session-cost mismatch in the original ranking.
  const weights = { ...intent.weights, affordability: intent.weights.costUtility, costUtility: undefined }
  delete weights.costUtility
  return rows.map((r) => {
    const metrics = { ...r.raw.metrics, quality: r.q, affordability: r.x.costUtility, speed: r.x.speed, context: r.x.context }
    return computeFinalScore({ ...r.raw, metrics }, 'general', weights, { archetype: 'balanced', isCoding: true, qualityFlagged: false }).exact
  })
}
function utility(r, normalized, type) {
  if (type === 'linear') return AXES.reduce((s, k, j) => s + normalized[j] * r.x[k], 0)
  // Sensitivity reference: a concave CES utility with rho=0.5, identical
  // preference weights and identical normalized attributes.
  return Math.pow(AXES.reduce((s, k, j) => s + normalized[j] * Math.sqrt(Math.max(r.x[k], 0.0001)), 0), 2)
}
function rankScores(rows, intent, method) {
  const w = normalizedWeights(intent.weights)
  if (method === 'Proposed (harmonized)') return proposed(rows, intent)
  if (method === 'WSM') return wsm(rows, w)
  if (method === 'TOPSIS') return topsis(rows, w)
  if (method === 'Cobb-Douglas') return rows.map((r) => AXES.reduce((product, k, j) => product * Math.pow(Math.max(r.x[k], 0.0001), w[j]), 1))
  if (method === 'CES rho=0.5') return powerMean(rows, w, 0.5)
  throw new Error(`Unknown method: ${method}`)
}
function summarize(picks, eligibleRows, weights, utilityType) {
  const u = (r) => utility(r, weights, utilityType)
  const utilities = picks.map((p) => u(p.picked))
  const regrets = picks.map((p) => p.oracleUtility - u(p.picked))
  const dominatedQualityCost = picks.filter(({ picked, eligible }) => eligible.some((r) => r !== picked && r.cost <= picked.cost && r.q >= picked.q && (r.cost < picked.cost || r.q > picked.q))).length
  const dominatedAllAxes = picks.filter(({ picked, eligible }) => eligible.some((r) => r !== picked && AXES.every((k) => r.x[k] >= picked.x[k]) && AXES.some((k) => r.x[k] > picked.x[k]))).length
  return {
    meanUtilityRegret: mean(regrets), worstTierUtilityRegret: Math.max(...regrets),
    meanQuality: mean(picks.map((p) => p.picked.q)), meanSessionCost: mean(picks.map((p) => p.picked.cost)),
    meanSpeed: mean(picks.map((p) => p.picked.speed)), meanContext: mean(picks.map((p) => p.picked.context)),
    paretoDominatedQualityCost: `${dominatedQualityCost}/${picks.length}`,
    paretoDominatedAllAxes: `${dominatedAllAxes}/${picks.length}`,
    picks: picks.map((p) => ({ budget: p.budget, model: p.picked.name, utilityRegret: +(p.oracleUtility - u(p.picked)).toFixed(6), cost: p.picked.cost, quality: p.picked.q, oracleModel: p.oracle.name })),
  }
}

const registry = await readRegistry()
const snapshot = JSON.parse(readFileSync(new URL('./arena_snapshot_2026-10-08.json', import.meta.url), 'utf8'))
const { cohort, normalization } = buildCohort(registry, snapshot)
const METHODS = ['Proposed (harmonized)', 'WSM', 'TOPSIS', 'Cobb-Douglas', 'CES rho=0.5']
const output = {
  protocol: {
    snapshot: snapshot.payload_last_updated, cohortN: cohort.length, parserUsed: 'cohere', budgetCapsUSD: BUDGETS,
    inputAxes: AXES, normalization,
    budgetWeightsRaw: INTENTS.budget.weights, balancedWeightsRaw: INTENTS.balanced.weights,
    utility: 'Primary: weighted arithmetic mean of fixed full-cohort normalized quality, log-inverse observed Arena session cost, speed, and context. Sensitivity: weighted CES rho=0.5 on those same fixed attributes.',
    rule: 'Hard filter by observed median Arena session cost <= tier cap before every ranker chooses. Grade every selection with identical intent utility, regardless of ranking method.',
    pareto: 'Reports both quality-cost dominance (no higher observed session cost and no lower predicted coding quality, one strict) and full four-axis dominance (no worse normalized quality, cost utility, speed, or context, one strict).',
    scope: 'Descriptive offline proxy. Same frozen cohort and six budget tiers; no independent user preferences or uncertainty interval. No Arena outcome is used in the primary utility.'
  },
  intents: {},
}

for (const [intentName, intent] of Object.entries(INTENTS)) {
  const w = normalizedWeights(intent.weights)
  const primary = Object.fromEntries(METHODS.map((method) => [method, []]))
  const ces = Object.fromEntries(METHODS.map((method) => [method, []]))
  for (const budget of BUDGETS) {
    const eligible = cohort.filter((r) => Number.isFinite(r.cost) && r.cost <= budget)
    if (eligible.length < 3) continue
    const oracleUtility = Math.max(...eligible.map((r) => utility(r, w, 'linear')))
    const oracle = eligible.find((r) => utility(r, w, 'linear') === oracleUtility)
    const oracleCES = Math.max(...eligible.map((r) => utility(r, w, 'ces')))
    const oracleCESModel = eligible.find((r) => utility(r, w, 'ces') === oracleCES)
    for (const method of METHODS) {
      const scores = rankScores(eligible, intent, method)
      let index = 0
      for (let i = 1; i < scores.length; i++) if (scores[i] > scores[index]) index = i
      primary[method].push({ budget, picked: eligible[index], eligible, oracleUtility, oracle })
      const cesUtility = (r) => utility(r, w, 'ces')
      const cesScores = eligible.map(cesUtility)
      const cesOracleUtility = oracleCESModel ? cesUtility(oracleCESModel) : Math.max(...cesScores)
      ces[method].push({ budget, picked: eligible[index], eligible, oracleUtility: cesOracleUtility, oracle: oracleCESModel })
    }
  }
  const results = {}
  for (const method of METHODS) {
    results[method] = {
      linearUtility: summarize(primary[method], cohort, w, 'linear'),
      cesUtilitySensitivity: summarize(ces[method], cohort, w, 'ces'),
    }
  }
  output.intents[intentName] = { prompt: intent.prompt, parserUsed: intent.parserUsed, weightsNormalized: Object.fromEntries(AXES.map((k, i) => [k, +w[i].toFixed(4)])), results }
  console.log(`\n${intentName.toUpperCase()} — normalized utility weights ${JSON.stringify(output.intents[intentName].weightsNormalized)}`)
  console.log('method'.padEnd(25), 'linearReg'.padStart(11), 'CESReg'.padStart(11), 'quality'.padStart(9), 'meanCost'.padStart(10), 'speed'.padStart(9), 'context'.padStart(9), 'Q/C dom'.padStart(9), '4D dom'.padStart(9))
  for (const [method, row] of Object.entries(results)) {
    const a = row.linearUtility, b = row.cesUtilitySensitivity
    console.log(method.padEnd(25), a.meanUtilityRegret.toFixed(4).padStart(11), b.meanUtilityRegret.toFixed(4).padStart(11), a.meanQuality.toFixed(4).padStart(9), a.meanSessionCost.toFixed(4).padStart(10), a.meanSpeed.toFixed(4).padStart(9), a.meanContext.toFixed(4).padStart(9), a.paretoDominatedQualityCost.padStart(9), a.paretoDominatedAllAxes.padStart(9))
  }
  for (const [method, row] of Object.entries(results)) {
    console.log(`  ${method} picks: ${row.linearUtility.picks.map((p) => `$${p.budget}:${p.model} (U-regret ${p.utilityRegret.toFixed(3)}, cost $${p.cost.toFixed(3)}, q ${p.quality.toFixed(3)})`).join(' | ')}`)
  }
}

writeFileSync(new URL('./intent_utility_eval.json', import.meta.url), JSON.stringify(output, null, 2) + '\n')
console.log('\nWrote artifact/scratch/intent_utility_eval.json')
