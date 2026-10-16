/** Check tab:missing_telemetry against a replication of the augmented-attribute test.
 *  Run: node scratch/regen_missing_telemetry.js */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { recommendModels, capabilityFit, sourceTrustFactor } from '../server/lib/scoring.js'
import { computeBenchmarkBounds, evaluateModelCapabilities } from '../server/lib/benchmarks.js'

const EPS = 0.05
function powerMean(metrics, weights, rho) {
  const sum = Object.values(weights).reduce((a, b) => a + b, 0)
  const ks = Object.keys(weights)
  if (rho === 0) {
    let prod = 1
    for (const k of ks) prod *= Math.pow(Math.max(metrics[k] ?? EPS, EPS), weights[k] / sum)
    return prod
  }
  const s = ks.reduce((acc, k) => acc + weights[k] * Math.pow(Math.max(metrics[k] ?? 0, 0), rho), 0)
  return Math.pow(s / sum, 1 / rho)
}
function raw(model, category, weights, opts, rho) {
  const m = model.metrics
  let s = powerMean(m, weights, rho)
  s *= Math.pow(capabilityFit(model, category), 1.2)
  s *= 1.0 - 0.20 * Math.pow(1 - (model.confidence ?? 0.8), 2)
  s *= sourceTrustFactor(model)
  s *= 100
  const q = m.quality ?? 0
  if (opts.archetype === 'frontier') {
    s *= q < 0.92 ? 1 / (1 + 3.0 * Math.pow(0.92 - q, 1.2))
                  : 1 + 0.25 * Math.pow(Math.max(0, (q - 0.92) / 0.08), 1.5)
  } else if (q < 0.70) {
    s *= 1 / (1 + 1.8 * Math.log(1 + 3.5 * (0.70 - q)))
  }
  if (opts.isCoding && category === 'general') {
    const caps = model.caps || {}
    const spec = caps.isCodeSpecialist || (model.secondaryUseCases || []).includes('code') || model.category === 'code'
    if (spec || caps.isReasoning) s *= 1 + (spec ? 0.04 : 0) + (caps.isReasoning ? 0.03 : 0) + (caps.hasTools ? 0.02 : 0)
    else if (q < 0.95) s *= 0.92
  }
  return s
}

const reg = await readRegistry()
const bounds = computeBenchmarkBounds(reg.models)
const { analysis } = await recommendModels(reg.models, { prompt: 'best coding model', limit: 25 })
const base = analysis.weights
const W = { ...base, context: 0.30 }   // augmented vector from the paper's stress test
console.log('base weights   =', JSON.stringify(base))
console.log('augmented      =', JSON.stringify(W), '\n')

const m = reg.models.find((x) => x.id === 'openrouter-anthropic-claude-opus-5-5')
const caps = evaluateModelCapabilities(m, bounds)
const e = { ...m, caps, metrics: { ...m.metrics } }
for (const k of ['quality', 'speed', 'privacy']) if (!Number.isFinite(e.metrics[k])) e.metrics[k] = caps[k]
e.metrics.quality = e.caps.coding
console.log('Opus 5.5 metrics =', JSON.stringify(Object.fromEntries(Object.entries(e.metrics).map(([k, v]) => [k, Number(v.toFixed(4))]))))
console.log('caps.coding =', caps.coding.toFixed(4), '\n')

const opts = { archetype: analysis.archetype, isCoding: Boolean(analysis.isCoding) }
const cols = [['CD(no fl.)', -1], ['CD(eps)', 0], ['rho=.25', 0.25], ['rho=.5', 0.5], ['linear', 1]]
const ctxFull = e.metrics.context
for (const [label, ctx] of [['missing ctx=0', 0], ['full telemetry', ctxFull]]) {
  const mm = { ...e.metrics, context: ctx }
  const row = cols.map(([, r]) => {
    if (r === -1) { // unfloored Cobb-Douglas
      const sum = Object.values(W).reduce((a, b) => a + b, 0)
      let prod = 1
      for (const k of Object.keys(W)) prod *= Math.pow(mm[k] ?? 0, W[k] / sum)
      let s = prod * Math.pow(capabilityFit(e, analysis.targetCategory), 1.2) * (1 - 0.20 * Math.pow(1 - (e.confidence ?? 0.8), 2)) * sourceTrustFactor(e) * 100
      const q = mm.quality ?? 0
      s *= q < 0.92 ? 1 / (1 + 3.0 * Math.pow(0.92 - q, 1.2)) : 1 + 0.25 * Math.pow(Math.max(0, (q - 0.92) / 0.08), 1.5)
      const c = e.caps || {}
      const spec = c.isCodeSpecialist || (e.secondaryUseCases || []).includes('code') || e.category === 'code'
      if (spec || c.isReasoning) s *= 1 + (spec ? 0.04 : 0) + (c.isReasoning ? 0.03 : 0) + (c.hasTools ? 0.02 : 0)
      else if (q < 0.95) s *= 0.92
      return s
    }
    return raw({ ...e, metrics: mm }, analysis.targetCategory, W, opts, r)
  })
  console.log(label.padEnd(16) + row.map((v, i) => `${cols[i][0]}=${v.toFixed(1)}`).join('  '))
}
console.log('\nPaper tab:missing_telemetry: missing -> 0.0 / 69.2 / 58.7 / 75.5 / 87.0 ; full (rho=.5) -> 97.8')
