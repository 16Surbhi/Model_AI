/**
 * Registry overhaul: one MEASURED value per model identity.
 *
 * THE DEFECT THIS EXISTS TO REMOVE
 *
 * The catalogue held 671 records of which 388 carried no benchmark observation
 * at all, and quality for those came from the structural fallback in
 * `evaluateModelCapabilities`, which clamps at `STRUCTURAL_QUALITY_CAP`. Every
 * sufficiently large unobserved model therefore landed on exactly 0.900 -- 28 of
 * them, all reachable in recommendations, all claiming the 91st percentile from
 * a parameter-count heuristic.
 *
 * Worse, the same real model was often recorded twice, and the two records
 * disagreed because only one of them had been measured:
 *
 *   Llama 3.3 70B Instruct   llama-3-3-70b                        q=0.900  no observations
 *                            openrouter-meta-llama-...-instruct  q=0.503  AA 7.7 / 11.9
 *   Nemotron 3 Nano 30B A3B  ...-nano-30b-a3b                     q=0.479  measured
 *                            ...-nano-30b-a3b-free                q=0.855  unobserved
 *   gpt-oss-120b             openrouter-openai-gpt-oss-120b       q=0.639  measured
 *                            ...-gpt-oss-120b-free                 q=0.900  unobserved
 *
 * 21 name groups had a quality spread above 0.05, and in all 21 at least one
 * side was benchmark-observed. The per-record guard in `recommendModels` ("where
 * real benchmark observations are present the recomputation from those
 * observations is authoritative and must win") cannot catch this, because it
 * compares a record against its own cache -- never one record against its
 * sibling. So the unmeasured record kept a saturated estimate, and on a
 * no-preference request that estimate won.
 *
 * WHAT THIS DOES
 *
 * 1. Group records by model IDENTITY (`normalized name`), which is the key under
 *    which the divergent duplicates actually collide.
 * 2. Within a group, elect the measurement AUTHORITY: the record carrying real
 *    benchmark observations.
 * 3. PROPAGATE those observations to every sibling that lacks them. A serving
 *    tier, a `:free` endpoint and a `-fast` variant of one model are one model,
 *    so they share one measured capability. Only the observation travels --
 *    pricing, access and context stay per record, because those genuinely differ
 *    by endpoint.
 * 4. Recompute every metric from the resulting observations and record WHERE
 *    each one came from in `metricSources`, so an estimate is never silently
 *    indistinguishable from a measurement.
 * 5. Derive affordability from the record's own pricing, and leave it `null`
 *    when the price is unknown rather than fabricating a high value.
 *
 * Records that still have no measured authority anywhere in their group keep
 * their structural estimate but are flagged `estimated`, which the ranking can
 * price explicitly instead of trusting the point estimate.
 */
import { evaluateModelCapabilities, computeBenchmarkBounds } from './benchmarks.js'
import { buildAaIndexFromApi, resolveAaMatches, aaObservationsFrom, priceToAffordability } from './source-sync.js'

// Must agree with `STRUCTURAL_QUALITY_CAP` in benchmarks.js.
const STRUCTURAL_QUALITY_CAP = 0.90

// Trust floors by provenance. A benchmarked number is worth at least this much
// confidence even if the source record under-declared it; an estimate is worth at
// most this much even if the source record over-declared it.
const MEASURED_CONFIDENCE_FLOOR = 0.90
const ESTIMATED_CONFIDENCE_CEILING = 0.55

const norm = (s) => String(s || '').replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const stripProviderPrefix = (s) => String(s || '').replace(/^[^:]+:\s*/, '').trim()

/**
 * Identity of the underlying model, independent of which endpoint serves it.
 *
 * Uses the display name rather than the id because that is where the divergent
 * duplicates actually collide: `llama-3-3-70b` and
 * `openrouter-meta-llama-llama-3-3-70b-instruct` share no id substring but do
 * share a name, while `-free` and `-fast` siblings differ in the id only.
 */
export function identityKey(model) {
  return norm(stripProviderPrefix(model?.name)) || norm(model?.id)
}

/** Does this record carry real Artificial Analysis observations? */
function hasObservations(model) {
  const aa = model?.benchmarkObservations?.artificial_analysis
  if (!aa || typeof aa !== 'object') return false
  return [aa.intelligence_index, aa.coding_index, aa.agentic_index, aa.math_index]
    .some((v) => Number.isFinite(Number(v)) && Number(v) > 0)
}

/** How much measured signal a record carries, for electing an authority. */
function measurementWeight(model) {
  const aa = model?.benchmarkObservations?.artificial_analysis
  if (!aa) return 0
  let n = 0
  for (const k of ['intelligence_index', 'coding_index', 'agentic_index', 'math_index', 'tokens_per_second']) {
    if (Number.isFinite(Number(aa[k])) && Number(aa[k]) > 0) n++
  }
  return n
}

/**
 * Elect one authority per identity group and report what it will repair.
 * Pure function of the records, so it can be tested without the filesystem.
 */
export function electAuthorities(models) {
  const groups = new Map()
  for (const m of models) {
    const key = identityKey(m)
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key).push(m)
  }

  const authorities = new Map()   // identityKey -> authority record
  const propagations = []         // { key, from, to }
  for (const [key, members] of groups) {
    const measured = members.filter(hasObservations)
    if (!measured.length) continue
    // Most measured signal wins; ties resolved by confidence, then by id so the
    // election is deterministic across runs.
    measured.sort((a, b) =>
      measurementWeight(b) - measurementWeight(a) ||
      (Number(b.confidence) || 0) - (Number(a.confidence) || 0) ||
      String(a.id).localeCompare(String(b.id)))
    const authority = measured[0]
    authorities.set(key, authority)
    for (const m of members) {
      if (m.id === authority.id) continue
      if (hasObservations(m)) continue
      propagations.push({ key, from: authority.id, to: m.id, name: m.name })
    }
  }
  return { groups, authorities, propagations }
}

/**
 * Rebuild the registry. Returns the new model list plus an audit of the run; it
 * does not write anything, so callers choose whether to persist.
 */
export async function rebuildRegistry(models, options = {}) {
  const aaIndex = options.aaIndex !== undefined ? options.aaIndex : await buildAaIndexFromApi()
  const observedAt = options.observedAt || new Date().toISOString()

  // ── Pass 1: per-record AA join, one-to-one, so a quantised re-package cannot
  //    silently inherit its parent's measured capability. ──
  const { matches: aaMatches, conflicts: aaConflicts } = resolveAaMatches(models, aaIndex)

  const joined = models.map((m) => {
    const record = aaIndex ? aaMatches.get(m.id) : null
    if (!record) return m
    const existing = m.benchmarkObservations && typeof m.benchmarkObservations === 'object' ? m.benchmarkObservations : {}
    return {
      ...m,
      benchmarkObservations: {
        design_arena: Array.isArray(existing.design_arena) ? existing.design_arena : [],
        artificial_analysis: aaObservationsFrom(record),
      },
    }
  })

  // ── Pass 2: elect one authority per identity and propagate its observations
  //    to the siblings that have none. ──
  const { groups, authorities, propagations } = electAuthorities(joined)
  const byId = new Map()
  for (const m of joined) {
    const key = identityKey(m)
    const authority = authorities.get(key)
    if (!authority || authority.id === m.id || hasObservations(m)) {
      byId.set(m.id, m)
      continue
    }
    byId.set(m.id, {
      ...m,
      benchmarkObservations: {
        design_arena: Array.isArray(m.benchmarkObservations?.design_arena) ? m.benchmarkObservations.design_arena : [],
        artificial_analysis: { ...authority.benchmarkObservations.artificial_analysis },
      },
      // Provenance: this record's capability was measured on a sibling endpoint.
      inheritedObservationsFrom: authority.id,
    })
  }

  const propagated = [...byId.values()]
  const bounds = computeBenchmarkBounds(propagated)

  // ── Pass 3: recompute metrics and stamp provenance. ──
  const audit = {
    input: models.length,
    identities: groups.size,
    aaConflicts: aaConflicts.length,
    propagated: propagations.length,
    measured: 0,
    estimated: 0,
    estimatedSaturated: 0,
    affordabilityUnknown: 0,
    affordabilityDerived: 0,
    propagationSamples: propagations.slice(0, 12),
  }

  const rebuilt = propagated.map((m) => {
    const caps = evaluateModelCapabilities(m, bounds)
    const metricSources = { ...(m.metricSources || {}) }

    const qualitySource = caps.benchDerived
      ? (m.inheritedObservationsFrom ? 'reference (measured sibling endpoint)' : 'Artificial Analysis intelligence/coding index')
      : (m.metricSources?.quality?.source || 'structural estimate (parameter scale)')
    if (caps.benchDerived) audit.measured++
    else {
      audit.estimated++
      // A structural estimate that has saturates at the cap carries no ordering
      // information between the models that reach it.
      if (caps.quality >= STRUCTURAL_QUALITY_CAP - 1e-9) audit.estimatedSaturated++
    }
    metricSources.quality = { source: qualitySource, observed: caps.benchDerived, observedAt }

    const speedSource = caps.speedObserved
      ? 'Artificial Analysis median output tokens/sec'
      : 'structural estimate (scale and reasoning overhead)'
    metricSources.speed = { source: speedSource, observed: caps.speedObserved, observedAt }

    // Affordability is a function of this endpoint's own price, so it is derived
    // here from the record rather than inherited, and unknown stays unknown.
    const affordability = priceToAffordability(
      m.pricing?.inputPerMillion,
      m.pricing?.outputPerMillion,
      m.category === 'document' ? 'document' : 'general',
    )
    if (affordability === null) audit.affordabilityUnknown++
    else audit.affordabilityDerived++
    metricSources.affordability = {
      source: affordability === null ? 'unmeasured (no price observation)' : 'OpenRouter pricing',
      observed: affordability !== null,
      observedAt,
    }

    metricSources.privacy = {
      source: 'access model (open weights vs hosted API)',
      observed: true,
      observedAt,
    }

    const metrics = { ...m.metrics }
    metrics.quality = caps.quality
    metrics.speed = caps.speed
    metrics.privacy = caps.privacy
    // Leave the stored affordability when the price is unknown, but say so in
    // `metricSources`; never overwrite it with a fabricated value.
    if (affordability !== null) metrics.affordability = affordability

    // ── Confidence must mean "trust in this number", and a number nobody
    //    measured cannot be trusted like one that was. This was inverted in the
    //    catalogue: the unmeasured Llama record carried confidence 0.95 and
    //    sourceAuthority 'curated', while the record of the SAME model with real
    //    Artificial Analysis observations carried 0.70. `confidenceFactor` in
    //    `computeFinalScore` spans 1.00 to 0.80 over the whole confidence range,
    //    so a 0.55-vs-0.95 gap is a ~4% score handicap -- larger than the ~1%
    //    point-estimate advantage a saturated 0.90 estimate holds over a measured
    //    model one percentile below it. Deriving confidence from provenance is
    //    therefore enough to make an unvalidated capacity claim lose to a
    //    measurement, without inventing a new constant. ──
    const confidence = caps.benchDerived
      ? Math.max(Number(m.confidence) || 0, MEASURED_CONFIDENCE_FLOOR)
      : Math.min(Number(m.confidence) || 0, ESTIMATED_CONFIDENCE_CEILING)

    return {
      ...m,
      provider: m.provider,
      confidence,
      sourceAuthority: caps.benchDerived ? 'benchmark' : (m.sourceAuthority || 'heuristic'),
      benchmarkSources: caps.benchDerived
        ? [...new Set([...(m.benchmarkSources || []), 'Artificial Analysis'])]
        : (m.benchmarkSources || []),
      metrics,
      metricSources,
      estimatedCapability: !caps.benchDerived,
    }
  })

  audit.output = rebuilt.length
  return { models: rebuilt, bounds, audit }
}
