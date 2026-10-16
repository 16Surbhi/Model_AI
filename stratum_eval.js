/**
 * STRATUM EVALUATION — the measurement the outcome-based tables cannot make.
 *
 * The merit table's cohort is 46/47 benchmark-measured models, while 366 of 671
 * registry records carry no measurement at all, so the tables can see 0.3% of the
 * defect population. Every number in them is a claim about the measured stratum,
 * silently generalised to a catalogue that is more than half unmeasured.
 *
 * This script measures the defect directly, on the whole catalogue, without
 * needing an outcome reference:
 *
 *   UNSERVED TIER — a budget tier whose highest-scoring eligible record is
 *   UNMEASURED while at least one MEASURED record was eligible. That is the user
 *   experience the live product actually has: the top recommendation for the
 *   request is a model nobody measured, while a measured one was available.
 *
 *   STRATUM SHIFT — the engine's top-K composition against the catalogue base
 *   rate. If 54% of the catalogue is unmeasured, a ranking that puts none of them
 *   in the top 10 is doing work the base rate cannot explain, and one that puts
 *   the estimator first is doing the opposite.
 *
 * Because it needs no outcomes, it runs on ANY registry, including one whose
 * models the arena never covered — which is exactly why it can see the repair.
 *
 * Run: node scratch/stratum_eval.js
 *      node scratch/stratum_eval.js <registry.json>     (evaluate a saved file)
 */
import '../server/lib/env.js'
import { readFileSync } from 'node:fs'
import { readRegistry, normalizeModel } from '../server/lib/registry-store.js'
import { evaluateModelCapabilities, computeBenchmarkBounds } from '../server/lib/benchmarks.js'
import { computeFinalScore, taskQuality, extractTaskProfile, recommendModels } from '../server/lib/scoring.js'

const BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
const PROMPTS = {
  frontier: 'best frontier coding models',
  budget: 'budget coding models',
  balanced: 'balanced coding models',
  context: 'coding model with a long context window for a big repo',
}

const normFile = (p) => p.replace(/\\/g, '/')

async function loadRegistry(argPath) {
  if (!argPath) return (await readRegistry()).models
  const raw = JSON.parse(readFileSync(argPath, 'utf8'))
  return (raw.models || []).map(normalizeModel)
}

/**
 * Measured = quality derived from real benchmark observations. Deliberately
 * computed from the capabilities rather than read from a provenance flag, so the
 * same criterion applies to a registry written before provenance existed.
 */
function stratumOf(models) {
  const bounds = computeBenchmarkBounds(models)
  const map = new Map()
  let measured = 0
  for (const m of models) {
    const caps = evaluateModelCapabilities(m, bounds)
    map.set(m.id, caps.benchDerived)
    if (caps.benchDerived) measured++
  }
  return { map, measured, total: models.length, bounds }
}

/** Cost axis used for eligibility: input price per million, as a buyer reads it. */
const costOf = (m) => {
  const v = Number(m.pricing?.inputPerMillion)
  return Number.isFinite(v) && v > 0 ? v : Infinity
}

async function evaluate(models, label) {
  const { map: measuredById, measured, total, bounds } = stratumOf(models)
  const unmeasured = total - measured

  console.log(`\n${'='.repeat(96)}`)
  console.log(`${label}`)
  console.log(`  catalogue: ${total} records, ${measured} measured (${(measured / total * 100).toFixed(1)}%), ${unmeasured} unmeasured`)
  console.log('='.repeat(96))
  console.log('prompt      archetype  top1       top-10 unmeasured  first-unmeasured-rank   unserved tiers   eligible base rate')

  const summary = { unservedTiers: 0, tiers: 0, topKUnmeasured: 0, topK: 0, byPrompt: {} }
  for (const [key, prompt] of Object.entries(PROMPTS)) {
    const res = await recommendModels(models, { prompt, limit: 60 })
    const { analysis } = res
    const flags = { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding), qualityFlagged: analysis.archetype === 'frontier' }
    const taskProfile = extractTaskProfile(prompt, analysis.targetCategory)

    // Score the WHOLE catalogue on the request's own profile, so tier eligibility
    // is not limited to the diversity-limited recommendation list.
    const scored = models.map((m) => {
      const caps = evaluateModelCapabilities(m, bounds)
      const metrics = { ...m.metrics }
      if (caps.benchDerived) metrics.quality = caps.quality
      else if (!Number.isFinite(metrics.quality)) metrics.quality = caps.quality
      if (caps.speedObserved) metrics.speed = caps.speed
      const profileQuality = taskQuality({ ...m, caps, metrics }, taskProfile)
      const rankingModel = { ...m, caps, metrics: { ...metrics, quality: profileQuality } }
      return { id: m.id, name: m.name, score: computeFinalScore(rankingModel, analysis.targetCategory, analysis.weights, flags).exact, cost: costOf(m), measured: Boolean(measuredById.get(m.id)) }
    }).sort((a, b) => b.score - a.score)

    const firstUnmeasured = scored.findIndex((s) => !s.measured)
    const top10 = scored.slice(0, 10)
    const topKUnmeasured = top10.filter((s) => !s.measured).length

    // Unserved tiers: the winner inside a budget is unmeasured although a
    // measured record was eligible at that budget.
    //
    // The eligible-set base rate is reported alongside each tier, because a bare
    // "the winner was unmeasured" is NOT evidence of a defect on its own. A
    // request that raises an axis only open-weight models occupy ("private",
    // "self-hosted") legitimately selects from a mostly-unmeasured eligible set,
    // and flagging that as unserved would be a false positive. The base rate is
    // what tells a reader whether an unmeasured win is surprising.
    const unserved = []
    const baseRates = []
    for (const B of BUDGETS) {
      const elig = scored.filter((s) => s.cost <= B)
      if (elig.length < 3) continue
      const winner = elig[0]
      const measuredEligible = elig.filter((s) => s.measured).length
      baseRates.push(measuredEligible / elig.length)
      summary.tiers++
      if (!winner.measured && measuredEligible > 0) {
        summary.unservedTiers++
        unserved.push({ B, winner: winner.name, score: winner.score, bestMeasured: elig.find((s) => s.measured), measuredShare: measuredEligible / elig.length })
      }
    }
    const measuredBaseRate = baseRates.length ? baseRates.reduce((a, b) => a + b, 0) / baseRates.length : NaN
    summary.baseRates = [...(summary.baseRates || []), measuredBaseRate]

    summary.topKUnmeasured += topKUnmeasured
    summary.topK += 10
    summary.byPrompt[key] = { firstUnmeasured, topKUnmeasured, unserved: unserved.length }

    console.log(
      `${key.padEnd(11)} ${String(analysis.archetype).padEnd(10)} ${(scored[0].measured ? 'measured' : 'UNMEASURED').padEnd(10)} ` +
      `${topKUnmeasured}/10       ${String(firstUnmeasured + 1).padStart(3)} of ${scored.length}            ${unserved.length}/${BUDGETS.length}` +
      `        ${(measuredBaseRate * 100).toFixed(0)}% eligible are measured`
    )
    for (const u of unserved) {
      console.log(`      $${u.B}: picked ${u.winner.slice(0, 26)} (score ${u.score.toFixed(2)}, unmeasured) over ${u.bestMeasured.name.slice(0, 26)} (${u.bestMeasured.score.toFixed(2)})  [${(u.measuredShare * 100).toFixed(0)}% of the eligible set was measured]`)
    }
  }
  return summary
}

const argPath = process.argv[2]
const current = await loadRegistry(argPath)
const backupPath = argPath ? null : 'server/data/model-registry.backup-2026-10-11T10-37-38-141Z.json'

console.log('STRATUM EVALUATION — does an unmeasured record win a tier a measured record could have won?')
if (backupPath) {
  const pre = await loadRegistry(backupPath)
  const before = await evaluate(pre, `PRE-OVERHAUL  (${normFile(backupPath)})`)
  const after = await evaluate(current, 'POST-OVERHAUL (server/data/model-registry.json)')
  console.log(`\n${'='.repeat(96)}`)
  console.log('VERDICT')
  console.log(`${'='.repeat(96)}`)
  console.log(`  unserved tiers (an unmeasured winner where a measured record was eligible)`)
  console.log(`    pre-overhaul  ${before.unservedTiers}/${before.tiers}`)
  console.log(`    post-overhaul ${after.unservedTiers}/${after.tiers}`)
  console.log(`  unmeasured records inside the top-10 (four prompts, 40 slots)`)
  console.log(`    pre-overhaul  ${before.topKUnmeasured}/40`)
  console.log(`    post-overhaul ${after.topKUnmeasured}/40`)
  const meanBase = (s) => s.baseRates.reduce((a, b) => a + b, 0) / s.baseRates.length
  console.log(`  NOTE: both rows are scored with the CURRENT stance, so this compares REGISTRIES only.`)
  console.log(`        The 4/24 -> 0/24 repair was measured before the stance change; the two fixes are`)
  console.log(`        not separately identified here. Eligible-set measured share: ${(meanBase(before) * 100).toFixed(1)}% -> ${(meanBase(after) * 100).toFixed(1)}%.`)
} else {
  await evaluate(current, `REGISTRY ${argPath}`)
}
