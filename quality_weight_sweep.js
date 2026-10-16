/**
 * Diagnostic only: quality-weight sweep using the live Cohere-parsed intents
 * captured from the API on 2026-10-13 and the frozen 2026-10-08 Arena cohort.
 * Does not modify the production scorer or manuscript.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { readRegistry } from '../../server/lib/registry-store.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../../server/lib/benchmarks.js'
import { computeFinalScore, taskQuality } from '../../server/lib/scoring.js'

const BUDGETS = [0.1, 0.25, 0.5, 1, 2, 5]
const PROMPTS = {
  budget: {
    prompt: 'budget coding models', parserUsed: 'cohere', archetype: 'budget',
    weights: { quality: 0.85, affordability: 2.5, speed: 0.6, context: 0.65, privacy: 0.05, availability: 0.05 },
  },
  balanced: {
    prompt: 'balanced coding models', parserUsed: 'cohere', archetype: 'balanced',
    weights: { quality: 1.25, affordability: 0.05, speed: 0.05, context: 0.65, privacy: 0.05, availability: 0.05 },
  },
}
const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const effort = (n) => (/max/i.test(n) ? 3 : /xhigh/i.test(n) ? 2 : /high/i.test(n) ? 1 : 0)
const mean = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN

function loadCohort(registry, snapshot) {
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
  return [...groups.values()].map(({ model, rows }) => {
    rows.sort((a, b) => effort(b.model) - effort(a.model))
    const chosen = rows[0]
    const caps = evaluateModelCapabilities(model, bounds)
    const raw = { ...model, caps, metrics: { ...model.metrics } }
    for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(raw.metrics[k])) raw.metrics[k] = caps[k]
    return {
      name: strip(model.name), outcome: chosen.net_pct, cost: chosen.median_usd,
      q: taskQuality(raw, ['general', 'code']), a: raw.metrics.affordability, s: raw.metrics.speed, raw,
    }
  })
}

function topsis(rows, weights) {
  const W = weights.map((x) => x / (weights.reduce((a, b) => a + b, 0) || 1))
  const cols = [0, 1, 2].map((j) => Math.sqrt(rows.reduce((sum, r) => sum + [r.q, r.a, r.s][j] ** 2, 0)))
  const V = rows.map((r) => [r.q, r.a, r.s].map((x, j) => x / (cols[j] || 1) * W[j]))
  const best = [0, 1, 2].map((j) => Math.max(...V.map((r) => r[j])))
  const worst = [0, 1, 2].map((j) => Math.min(...V.map((r) => r[j])))
  return V.map((r) => {
    const dp = Math.sqrt(r.reduce((t, x, j) => t + (x - best[j]) ** 2, 0))
    const dm = Math.sqrt(r.reduce((t, x, j) => t + (x - worst[j]) ** 2, 0))
    return dm / (dp + dm || 1)
  })
}

const registry = await readRegistry()
const snapshot = JSON.parse(readFileSync(new URL('./arena_snapshot_2026-10-08.json', import.meta.url), 'utf8'))
const cohort = loadCohort(registry, snapshot)
const net = cohort.map((r) => r.outcome)

function scoreRows(profile, weights) {
  const options = { archetype: profile.archetype, isCoding: true, qualityFlagged: profile.archetype === 'frontier' }
  return cohort.map((r) => computeFinalScore({ ...r.raw, metrics: { ...r.raw.metrics, quality: r.q } }, 'general', weights, options).exact)
}
function evaluate(scores) {
  const eligible = BUDGETS.map((budget) => cohort.map((r, i) => i).filter((i) => Number.isFinite(cohort[i].cost) && cohort[i].cost <= budget))
    .filter((indices) => indices.length >= 3)
  const tiers = eligible.map((ids) => {
    const oracle = Math.max(...ids.map((i) => net[i]))
    let pick = ids[0]
    for (const i of ids) if (scores[i] > scores[pick]) pick = i
    return { regret: oracle - net[pick], oracle, value: net[pick], cost: cohort[pick].cost, name: cohort[pick].name, ids }
  })
  const dominated = tiers.filter((t) => t.ids.some((i) => i !== cohort.findIndex((r) => r.name === t.name) && net[i] >= t.value && cohort[i].cost <= t.cost && (net[i] > t.value || cohort[i].cost < t.cost))).length
  return {
    meanRegret: mean(tiers.map((t) => t.regret)), worstTierRegret: Math.max(...tiers.map((t) => t.regret)),
    oracleHits: `${tiers.filter((t) => t.regret <= 1e-9).length}/${tiers.length}`,
    valueCaptured: mean(tiers.map((t) => t.value)) / mean(tiers.map((t) => t.oracle)),
    meanCost: mean(tiers.map((t) => t.cost)), netPerDollar: mean(tiers.map((t) => t.value / t.cost)),
    dominated: `${dominated}/${tiers.length}`, picks: tiers.map(({ regret, name, cost }) => ({ regret, name, cost })),
  }
}

const output = { metadata: { parserUsed: 'cohere', snapshot: snapshot.payload_last_updated, cohort: cohort.length, budgets: BUDGETS }, profiles: {} }
for (const [name, profile] of Object.entries(PROMPTS)) {
  const base = profile.weights
  const w3 = [base.quality, base.affordability, base.speed]
  const sum = w3.reduce((a, b) => a + b, 0)
  const normalized = w3.map((x) => x / sum)
  const wsm = cohort.map((r) => normalized[0] * r.q + normalized[1] * r.a + normalized[2] * r.s)
  const reference = {
    'Quality only': cohort.map((r) => r.q),
    WSM: wsm,
    TOPSIS: topsis(cohort, w3),
    'Cobb-Douglas': cohort.map((r) => Math.pow(Math.max(r.q, 0.05), normalized[0]) * Math.pow(Math.max(r.a, 0.05), normalized[1]) * Math.pow(Math.max(r.s, 0.05), normalized[2])),
    CES: cohort.map((r) => Math.pow(normalized[0] * Math.sqrt(r.q) + normalized[1] * Math.sqrt(r.a) + normalized[2] * Math.sqrt(r.s), 2)),
  }
  const variants = []
  const qualityLevels = [...new Set([base.quality, base.quality * 2, 3.6, 5.4, 7.2, 10, 15, 20, 30])]
  for (const targetQualityWeight of qualityLevels) {
    const multiplier = targetQualityWeight / base.quality
    const weights = { ...base, quality: targetQualityWeight }
    const score = scoreRows(profile, weights)
    variants.push({ qualityMultiplier: +multiplier.toFixed(3), qualityWeight: +targetQualityWeight.toFixed(3), metrics: evaluate(score) })
  }
  const customWeightSets = name === 'budget'
    ? [
        { label: 'budget-q20-a1.25-s0.3-c0.2', weights: { ...base, quality: 20, affordability: 1.25, speed: 0.3, context: 0.2 } },
        { label: 'budget-q30-a1.25-s0.3-c0.2', weights: { ...base, quality: 30, affordability: 1.25, speed: 0.3, context: 0.2 } },
        { label: 'budget-q20-a0.625-s0.15-c0.1', weights: { ...base, quality: 20, affordability: 0.625, speed: 0.15, context: 0.1 } },
        { label: 'budget-q30-a0.625-s0.15-c0.1', weights: { ...base, quality: 30, affordability: 0.625, speed: 0.15, context: 0.1 } },
      ]
    : [
        { label: 'balanced-q3.6-c0.325', weights: { ...base, quality: 3.6, context: 0.325 } },
        { label: 'balanced-q3.6-c0.1', weights: { ...base, quality: 3.6, context: 0.1 } },
        { label: 'balanced-q5.4-c0.325', weights: { ...base, quality: 5.4, context: 0.325 } },
        { label: 'balanced-q10-c0.1', weights: { ...base, quality: 10, context: 0.1 } },
        { label: 'balanced-q15-c0.05', weights: { ...base, quality: 15, context: 0.05 } },
      ]
  const custom = customWeightSets.map(({ label, weights }) => ({ label, weights, metrics: evaluate(scoreRows(profile, weights)) }))
  output.profiles[name] = {
    prompt: profile.prompt, parserUsed: profile.parserUsed, baseWeights: base, variants, custom,
    references: Object.fromEntries(Object.entries(reference).map(([k, v]) => [k, evaluate(v)])),
  }
  console.log(`\n${name} | LLM parser=${profile.parserUsed} | n=${cohort.length} | baseline=${JSON.stringify(base)}`)
  console.log('method/config'.padEnd(24), 'meanReg'.padStart(8), 'worst'.padStart(8), 'hits'.padStart(7), 'value'.padStart(8), 'mean$'.padStart(8), 'net/$'.padStart(8), 'dominated'.padStart(11))
  for (const [method, metrics] of Object.entries(output.profiles[name].references)) {
    console.log(method.padEnd(24), metrics.meanRegret.toFixed(2).padStart(8), metrics.worstTierRegret.toFixed(2).padStart(8), metrics.oracleHits.padStart(7), metrics.valueCaptured.toFixed(3).padStart(8), metrics.meanCost.toFixed(3).padStart(8), metrics.netPerDollar.toFixed(2).padStart(8), metrics.dominated.padStart(11))
  }
  for (const variant of variants) {
    const m = variant.metrics
    const label = `proposed q×${variant.qualityMultiplier} (q=${variant.qualityWeight})`
    console.log(label.padEnd(24), m.meanRegret.toFixed(2).padStart(8), m.worstTierRegret.toFixed(2).padStart(8), m.oracleHits.padStart(7), m.valueCaptured.toFixed(3).padStart(8), m.meanCost.toFixed(3).padStart(8), m.netPerDollar.toFixed(2).padStart(8), m.dominated.padStart(11))
  }
  for (const variant of custom) {
    const m = variant.metrics
    console.log(variant.label.padEnd(24), m.meanRegret.toFixed(2).padStart(8), m.worstTierRegret.toFixed(2).padStart(8), m.oracleHits.padStart(7), m.valueCaptured.toFixed(3).padStart(8), m.meanCost.toFixed(3).padStart(8), m.netPerDollar.toFixed(2).padStart(8), m.dominated.padStart(11))
  }
  for (const variant of variants) {
    const m = variant.metrics
    console.log(' picks', `q×${variant.qualityMultiplier}: ${m.picks.map((p) => `${p.name}[$${p.cost.toFixed(2)},r${p.regret.toFixed(2)}]`).join(' | ')}`)
  }
}
writeFileSync(new URL('./quality_weight_sweep.json', import.meta.url), JSON.stringify(output, null, 2))
console.log('\nWrote artifact/scratch/quality_weight_sweep.json')
