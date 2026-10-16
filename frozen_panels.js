/**
 * Read the frozen arena_corrected.json run and print the 2026-10-08 headline
 * panels in the form the manuscript tables need.
 *
 * Run: node scratch/frozen_panels.js
 */
import { readFileSync } from 'node:fs'

const j = JSON.parse(readFileSync('scratch/arena_corrected.json', 'utf8'))
const r = j.runs.find((x) => x.snapshot === '2026-10-08' && x.snapshot_column === 'net')
const S = (v) => (v === undefined ? '-' : JSON.stringify(v))

console.log(`run ${r.snapshot} | column ${r.snapshot_column} | n_cohort ${r.n_cohort} | n_priced ${r.n_priced} | spearman(q,outcome) ${r.spearman_q}`)
for (const a of ['frontier', 'budget', 'balanced']) {
  const A = r.archetypes[a]
  console.log(`\n##### ${a}  weights ${JSON.stringify(A.weights)}`)
  for (const m of Object.keys(A.summary)) {
    const s = A.summary[m]
    console.log(
      `   ${m.padEnd(16)} mean=${String(s.mean_regret).padStart(6)} ci=${S(s.mean_regret_ci).padEnd(18)}` +
      ` pd=${S(s.paired_diff_vs_proposed).padEnd(20)} worst=${String(s.worst_tier_regret).padStart(6)}` +
      ` vc=${String(s.value_captured).padStart(6)} rho=${String(s.spearman).padStart(6)} tau=${String(s.kendall).padStart(6)}` +
      ` top3=${s.top3_overlap} hits=${s.oracle_hits}`)
  }
  console.log(`   sweep: ${JSON.stringify(A.sweep)}`)
  console.log(`   vfm:   ${JSON.stringify(A.vfm)}`)
  console.log('   picks:')
  for (const p of A.picks) {
    const t = A.tiers.find((x) => x.budget === p.budget) || {}
    console.log(
      `     $${String(p.budget).padEnd(5)} oracle=${String(p.oracle).padStart(6)}` +
      ` proposal pick=${p.proposed.padEnd(20)} net=${String(p.net).padStart(6)} cost=${p.cost.toFixed(3)}` +
      ` | ref pick=${String(t['_pick_Quality only']).padEnd(20)} net=${String(t['_net_Quality only']).padStart(6)} cost=${Number(t['_cost_Quality only']).toFixed(3)}`)
  }
}
