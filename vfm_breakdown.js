/**
 * Break down the n=45 value-for-money result: which model each method picks at
 * each budget tier, and how the "or" answer depends on the statistic's form.
 *
 *   mean-of-ratios   = MEAN_tier( net(pick) / cost(pick) )   <- used in the paper
 *   ratio-of-means   = MEAN_tier( net(pick) ) / MEAN_tier( cost(pick) )
 *
 * Run: node scratch/vfm_breakdown.js
 */
import { readFileSync } from 'node:fs'

const exp = JSON.parse(readFileSync('scratch/arena_regret_expanded.json', 'utf8'))
const vfm = JSON.parse(readFileSync('scratch/arena_value_for_money.json', 'utf8'))

const look = new Map(exp.cohort.map((c) => [c.name, c]))
const M = (v) => v.reduce((a, b) => a + b, 0) / v.length

for (const arch of ['frontier', 'budget', 'balanced']) {
  const tiers = exp.archetypes[arch].tiers
  const keys = Object.keys(tiers)
  console.log(`\n================ ${arch.toUpperCase()} (n=45, ${keys.length} tiers) ================`)
  console.log('  budget   Proposed pick                 net     cost    net/$   | Quality-only pick             net     cost    net/$')

  const acc = { Proposed: { net: [], cost: [] }, 'Quality only': { net: [], cost: [] } }
  for (const k of keys) {
    const t = tiers[k]
    let line = `  $${k.padEnd(6)}`
    for (const meth of ['Proposed', 'Quality only']) {
      const name = t['_pick_' + meth]
      const c = look.get(name)
      const ratio = c.net / c.cost
      acc[meth].net.push(c.net); acc[meth].cost.push(c.cost)
      // is this pick Pareto-dominated by another eligible model on (net, cost)?
      const dom = exp.cohort.some((o) => o.cost <= Number(k) &&
        o.net >= c.net && o.cost <= c.cost && (o.net > c.net || o.cost < c.cost))
      line += `${name.padEnd(30)}${String(c.net).padStart(7)}${('$' + c.cost.toFixed(3)).padStart(9)}${ratio.toFixed(1).padStart(8)}${dom ? ' D' : '  '} | `
    }
    console.log(line)
  }

  console.log('  ---- summary (D = Pareto-dominated on the measured plane at that tier) ----')
  for (const meth of ['Proposed', 'Quality only']) {
    const j = vfm.cohorts['n=45 (full leaderboard)'].archetypes[arch].results[meth]
    const mor = j.net_per_usd
    const rom = M(acc[meth].net) / M(acc[meth].cost)
    console.log(`  ${meth.padEnd(14)} mean-of-ratios=${mor.toFixed(2).padStart(6)}   ` +
      `ratio-of-means=${rom.toFixed(2).padStart(6)}   dominated=${j.dominated_tiers}/${j.tiers}`)
  }
}
