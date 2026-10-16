/**
 * Intent-parser evaluation (LLM parser only).
 *
 * The paper's premise is a natural-language intent parser, but it was never
 * evaluated. This script measures:
 *
 *  1. Run-to-run reproducibility. Each prompt is parsed REPS times; the
 *     returned intent (category, priority order) and the weight vector the
 *     pipeline derives from it are compared across runs.
 *  2. Cross-variant stability. Singular/plural and paraphrase variants of the
 *     same request are parsed and their derived weight vectors compared, so
 *     stability is not claimed from a single lexical pair.
 *  3. Deployed weights and shortlist coverage, so the reproducibility of the
 *     numbers the paper reports can be checked.
 *
 * Only the hosted LLM parser is exercised; the deterministic fallback is not
 * part of the deployed configuration and is not evaluated here.
 *
 * The Cohere trial key allows 20 calls/minute, so calls are throttled and
 * retried.
 *
 * Run: node scratch/parser_eval.js
 */
import '../server/lib/env.js'
import { readRegistry } from '../server/lib/registry-store.js'
import { parsePromptWithLLM } from '../server/lib/llm-parser.js'
import { recommendModels, analyzePrompt } from '../server/lib/scoring.js'
import { readFileSync, writeFileSync } from 'node:fs'

const REPS = Number(process.env.REPS || 4)
const MIN_INTERVAL_MS = 3300

let lastCall = 0
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
async function parseThrottled(prompt) {
  for (let attempt = 0; attempt < 5; attempt++) {
    const wait = Math.max(0, lastCall + MIN_INTERVAL_MS - Date.now())
    if (wait) await sleep(wait)
    lastCall = Date.now()
    const intent = await parsePromptWithLLM(prompt)
    if (intent) return intent
    await sleep(15000)
  }
  return null
}

const FAMILIES = [
  { label: 'frontier-coding', variants: ['best coding model', 'best coding models', 'best frontier coding models', 'most capable coding model, cost no object'] },
  { label: 'budget-coding', variants: ['budget coding models', 'cheap coding model', 'low-cost coding models', 'affordable code assistant'] },
  { label: 'balanced-coding', variants: ['balanced coding models', 'balanced coding model', 'well-rounded model for coding', 'a model that is good all round for code'] },
  { label: 'speed', variants: ['fastest model for code completion', 'low latency coding model'] },
  { label: 'other-category', variants: ['image generation model', 'self-hosted model for document processing', 'cheap fast model for bulk text extraction'] },
]
const flat = []
for (const f of FAMILIES) for (const v of f.variants) flat.push({ label: f.label, prompt: v })

const WKEYS = ['quality', 'affordability', 'speed', 'context', 'privacy', 'availability']
const wvec = (w) => WKEYS.map((k) => (w?.[k] ?? 0).toFixed(3)).join(',')
const modal = (arr) => {
  if (!arr.length) return { top: '(none)', n: 0, distinct: 0 }
  const counts = new Map()
  for (const v of arr) counts.set(v, (counts.get(v) || 0) + 1)
  const ent = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]
  return { top: ent[0], n: ent[1], distinct: counts.size }
}

const out = { reps: REPS, prompts: [], families: [] }
console.log(`LLM intent parser evaluation — ${REPS} parses per prompt\n`)

for (const { label, prompt } of flat) {
  const runs = []
  for (let r = 0; r < REPS; r++) {
    const intent = await parseThrottled(prompt)
    if (!intent) continue
    const analysis = analyzePrompt(prompt, null, intent)
    runs.push({ intent, weights: analysis.weights, archetype: analysis.archetype, category: analysis.targetCategory })
  }
  if (!runs.length) { console.log(`!! no successful parses for "${prompt}"`); continue }
  const cat = modal(runs.map((r) => r.category))
  const prio = modal(runs.map((r) => r.intent.priorities.join('+')))
  const wv = modal(runs.map((r) => wvec(r.weights)))
  const devs = WKEYS.map((k) => Math.max(...runs.map((r) => r.weights[k])) - Math.min(...runs.map((r) => r.weights[k])))
  const rec = {
    label, prompt, parsed: runs.length, archetype: runs[0].archetype,
    category: cat.top, category_agreement: `${cat.n}/${runs.length}`,
    priorities_raw: prio.top, priority_agreement: `${prio.n}/${runs.length}`,
    derived_weights: runs[0].weights,
    weight_vector_agreement: `${wv.n}/${runs.length}`, weight_distinct: wv.distinct,
    max_weight_deviation: +Math.max(...devs).toFixed(4),
    run_priorities: runs.map((r) => r.intent.priorities.join('+')),
  }
  out.prompts.push(rec)
  console.log(`${label.padEnd(19)} ${prompt.slice(0, 40).padEnd(42)} ${rec.archetype.padEnd(9)} cat=${rec.category.padEnd(8)} ${rec.category_agreement.padStart(4)}  prio=${rec.priority_agreement.padStart(4)} distinct=${prio.distinct}  wvec=${rec.weight_vector_agreement.padStart(4)} dev=${rec.max_weight_deviation}`)
  console.log(`    raw priorities per run: ${rec.run_priorities.join(' | ')}`)
}

for (const f of FAMILIES) {
  const rows = out.prompts.filter((p) => p.label === f.label)
  const m = modal(rows.map((r) => wvec(r.derived_weights)))
  out.families.push({ label: f.label, variants: rows.length, modal_weights: m.top, agreement: `${m.n}/${rows.length}`, distinct: m.distinct })
}
console.log('\ncross-variant stability of the derived weight vector (modal within family):')
for (const f of out.families) console.log(`  ${f.label.padEnd(19)} ${String(f.agreement).padStart(5)}  distinct=${f.distinct}  modal=[${f.modal_weights}]`)

const reg = await readRegistry()
const snap = JSON.parse(readFileSync('scratch/arena_snapshot_2026-10-08.json', 'utf8'))
const norm = (s) => String(s).replace(/\(.*?\)/g, ' ').replace(/[^a-z0-9]/gi, '').toLowerCase()
const arenaNames = new Set(snap.models.map((m) => norm(m.model)))

out.deployed = []
for (const p of ['best coding model', 'best frontier coding models', 'budget coding models', 'balanced coding models']) {
  await sleep(MIN_INTERVAL_MS)
  lastCall = Date.now()
  const { analysis, recommendations, parserUsed } = await recommendModels(reg.models, { prompt: p, limit: 10 })
  const names = recommendations.map((r) => r.model?.name || r.name)
  const covered = names.filter((n) => arenaNames.has(norm(n))).length
  const row = { prompt: p, parserUsed, archetype: analysis.archetype, weights: analysis.weights, top10: names, arena_covered: `${covered}/10` }
  out.deployed.push(row)
  console.log(`\ndeployed "${p}": parser=${parserUsed} archetype=${row.archetype} covered=${row.arena_covered}`)
  console.log(`  weights: ${wvec(row.weights)}`)
  console.log(`  top-10 : ${names.join(', ')}`)
}

writeFileSync('scratch/parser_eval.json', JSON.stringify(out, null, 2))
console.log('\nwrote scratch/parser_eval.json')
