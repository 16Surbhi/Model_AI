/**
 * Exploratory quality-band guardrail evaluation.
 *
 * The only preference inputs are frozen outputs from fresh Cohere parser runs
 * (parserUsed=cohere); this script never invokes the regex parser. It partitions
 * the 47-model Arena cohort into repeated five-fold candidate pools, then
 * evaluates the guardrail and four integrative rules on the same eligible
 * candidates. Arena net improvement is retained only as an outcome proxy.
 *
 * Run from the project root: node artifact/scratch/quality_guardrail_eval.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { taskQuality, computeFinalScore } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'
import { readFileSync, writeFileSync } from 'node:fs'

const BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
const DELTAS = [0, 0.01, 0.02, 0.05]
const REPEATS = 20
const SEED = 20261013
const AXES = ['quality', 'affordability', 'speed', 'context']
const INTENTS = {
  balanced: {
    prompt: 'For coding models, balance quality, affordability, speed, and context equally.',
    parserUsed: 'cohere',
    weights: { quality: 5, affordability: 5, speed: 5, context: 5 },
    weightsFlat: true,
  },
  budget: {
    prompt: 'For coding models, prioritize affordability (10/10), quality (8/10), speed (5/10), and context (3/10).',
    parserUsed: 'cohere',
    weights: { quality: 8, affordability: 10, speed: 5, context: 3 },
    weightsFlat: false,
  },
}
if (Object.values(INTENTS).some((intent) => intent.parserUsed !== 'cohere')) {
  throw new Error('This analysis requires frozen LLM parser outputs; regex fallback is not accepted')
}
const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const strip = (s) => String(s).replace(/^[^:]+:\s*/, '').trim()
const effort = (s) => (/max/i.test(s) ? 3 : /xhigh/i.test(s) ? 2 : /high/i.test(s) ? 1 : 0)
const mean = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }
}
function topsis(rows, w) {
  const norms = AXES.map((_, j) => Math.sqrt(rows.reduce((s, r) => s + r.x[j] ** 2, 0)) || 1)
  const ws = w.reduce((s, v) => s + v, 0)
  const v = rows.map((r) => r.x.map((x, j) => x / norms[j] * w[j] / ws))
  const best = AXES.map((_, j) => Math.max(...v.map((r) => r[j])))
  const worst = AXES.map((_, j) => Math.min(...v.map((r) => r[j])))
  return v.map((r) => {
    const dp = Math.sqrt(r.reduce((s, x, j) => s + (x - best[j]) ** 2, 0))
    const dm = Math.sqrt(r.reduce((s, x, j) => s + (x - worst[j]) ** 2, 0))
    return dm / (dp + dm || 1)
  })
}

const reg = await readRegistry()
const bounds = computeBenchmarkBounds(reg.models)
const idx = new Map()
for (const m of reg.models) {
  if (m.aliasOf) continue
  for (const k of [norm(m.name), norm(strip(m.name))]) {
    if (!idx.has(k)) idx.set(k, [])
    idx.get(k).push(m)
  }
}
const snapshot = JSON.parse(readFileSync('scratch/arena_snapshot_2026-10-08.json', 'utf8'))
const groups = new Map()
for (const row of snapshot.models) {
  if (!Number.isFinite(row.net_pct)) continue
  const hits = idx.get(norm(row.model)) || idx.get(norm(strip(row.model)))
  if (!hits?.length) continue
  const model = hits[0]
  const group = groups.get(model.id) || { model, rows: [] }
  group.rows.push(row)
  groups.set(model.id, group)
}
const cohort = []
for (const { model, rows } of groups.values()) {
  rows.sort((a, b) => effort(b.model) - effort(a.model))
  const arena = rows[0]
  const caps = evaluateModelCapabilities(model, bounds)
  const raw = { ...model, caps, metrics: { ...model.metrics } }
  for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(raw.metrics[k])) raw.metrics[k] = caps[k]
  const q = taskQuality(raw, ['general', 'code'])
  raw.metrics.quality = q
  const metrics = { quality: q, affordability: raw.metrics.affordability ?? 0.05, speed: raw.metrics.speed ?? 0.05, context: raw.metrics.context ?? 0.05 }
  cohort.push({ name: strip(model.name), net: arena.net_pct, cost: arena.median_usd, raw, metrics, x: AXES.map((k) => metrics[k]) })
}
if (cohort.length !== 47) throw new Error(`Expected frozen cohort n=47, got ${cohort.length}`)

const rng = mulberry32(SEED)
const acc = {}
for (const intentName of Object.keys(INTENTS)) {
  acc[intentName] = {}
  const intent = INTENTS[intentName]
  const w = AXES.map((k) => intent.weights[k])
  for (const method of ['Proposed', 'WSM', 'TOPSIS', 'Cobb-Douglas', 'CES', ...DELTAS.map((d) => `Guardrail delta=${d}`)]) {
    acc[intentName][method] = { regret: [], cost: [], hit: [] }
  }
  for (let repeat = 0; repeat < REPEATS; repeat++) {
    const order = cohort.map((_, i) => i)
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]] }
    for (let fold = 0; fold < 5; fold++) {
      const rows = order.filter((_, i) => i % 5 === fold).map((i) => cohort[i])
      const normW = w.map((x) => x / w.reduce((a, b) => a + b, 0))
      const methodScores = {
        Proposed: rows.map((c) => computeFinalScore(c.raw, 'general', intent.weights, { archetype: intentName, isCoding: true, qualityFlagged: false }).exact),
        WSM: rows.map((c) => c.x.reduce((s, x, j) => s + normW[j] * x, 0)),
        TOPSIS: topsis(rows, w),
        'Cobb-Douglas': rows.map((c) => c.x.reduce((s, x, j) => s * Math.pow(Math.max(x, 0.05), normW[j]), 1)),
        CES: rows.map((c) => c.x.reduce((s, x, j) => s + normW[j] * Math.sqrt(Math.max(x, 0.05)), 0) ** 2),
      }
      for (const B of BUDGETS) {
        const eligible = rows.map((_, i) => i).filter((i) => rows[i].cost != null && Number.isFinite(rows[i].cost) && rows[i].cost <= B)
        if (eligible.length < 3) continue
        const oracle = Math.max(...eligible.map((i) => rows[i].net))
        const record = (method, pick) => {
          const a = acc[intentName][method]
          a.regret.push(oracle - rows[pick].net)
          a.cost.push(rows[pick].cost)
          a.hit.push(oracle - rows[pick].net <= 1e-9 ? 1 : 0)
        }
        for (const [name, scores] of Object.entries(methodScores)) {
          const pick = eligible.reduce((best, i) => scores[i] > scores[best] ? i : best, eligible[0])
          record(name, pick)
        }
        const qMax = Math.max(...eligible.map((i) => rows[i].metrics.quality))
        for (const delta of DELTAS) {
          const band = eligible.filter((i) => rows[i].metrics.quality >= qMax - delta)
          let pick
          if (intentName === 'budget') {
            pick = band.reduce((best, i) => rows[i].cost < rows[best].cost || (rows[i].cost === rows[best].cost && methodScores.WSM[i] > methodScores.WSM[best]) ? i : best, band[0])
          } else {
            pick = band.reduce((best, i) => methodScores.WSM[i] > methodScores.WSM[best] ? i : best, band[0])
          }
          record(`Guardrail delta=${delta}`, pick)
        }
      }
    }
  }
}

const result = {
  protocol: { cohort_n: cohort.length, snapshot: 'Agent Arena headline net improvement, payload 2026-10-08', budgets_usd: BUDGETS, repeated_partitions: REPEATS, folds_per_partition: 5, seed: SEED, note: 'Repeated candidate-pool subsampling; the partitions reuse models and are descriptive, not independent inferential replicates. Outcomes grade choices only; no Arena outcome enters a score.' },
  intents: INTENTS,
  metrics: 'Mean Arena net-improvement regret (oracle minus selected model); mean median input-token cost (USD); oracle-hit fraction. Lower cost and regret are preferred; higher hit fraction is preferred.',
  results: {},
}
for (const [intent, methods] of Object.entries(acc)) {
  result.results[intent] = {}
  result.protocol[intent + '_valid_pool_budget_decisions'] = methods.Proposed.regret.length
  for (const [method, values] of Object.entries(methods)) {
    result.results[intent][method] = { n: values.regret.length, mean_regret: mean(values.regret), mean_cost_usd: mean(values.cost), oracle_hit_rate: mean(values.hit) }
  }
}
writeFileSync('scratch/quality_guardrail_eval.json', JSON.stringify(result, null, 2) + '\n')
console.log(JSON.stringify(result.results, null, 2))
