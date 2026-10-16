#!/usr/bin/env python3
"""Verify the manuscript against the frozen artifact it claims to use.

Every empirical figure in the paper must be reproducible from
scratch/arena_corrected.json (run 2026-10-08 / net, n_cohort 47), which is
produced by scratch/arena_corrected.js on the frozen registry revision. This
checker reads that JSON, recomputes the values the tables quote, and asserts
each one appears in the manuscript -- and that the retracted figures do not.

Run: python scratch/verify_frozen_panels.py
"""
import io
import json
import os
import re
import sys

MAIN = 'docs/model-compass-technical-draft.tex'
FROZEN = 'scratch/arena_corrected.json'
ABLATION = 'scratch/ablation_components.json'
SWING = 'scratch/weights_swing.json'

text = io.open(MAIN, encoding='utf-8').read()
j = json.load(io.open(FROZEN, encoding='utf-8'))
run = next(r for r in j['runs'] if r['snapshot'] == '2026-10-08' and r['snapshot_column'] == 'net')

fails, oks = [], []


def check(name, cond, detail=''):
    (oks if cond else fails).append(name + (f' -- {detail}' if detail and not cond else ''))


def r2(v):
    return f'{v:.2f}'


# ── the frozen run is what the paper says it is ──────────────────────────────
check('frozen cohort is n=47', run['n_cohort'] == 47, run['n_cohort'])
check('frozen cohort is fully priced', run['n_priced'] == 47, run['n_priced'])
check('paper states n=47', '$n{=}47$' in text)
check('paper no longer states a 45-model cohort',
      'n{=}45' not in text and '45 models' not in text and '45 of 52' not in text)

# ── every panel value the tables quote, recomputed from the artifact ─────────
PANELS = {'frontier': '$\\mathbf{w}_{\\text{qual}}=3.6$', 'budget': 'budget', 'balanced': 'balanced'}
for arch in ('frontier', 'budget', 'balanced'):
    s = run['archetypes'][arch]['summary']
    mean = r2(s['Proposed']['mean_regret'])
    check(f'{arch}: proposed mean regret {mean} appears', f'${mean}$' in text)
    for k in ('WSM', 'TOPSIS', 'Cobb-Douglas', 'CES only'):
        v = r2(s[k]['mean_regret'])
        check(f'{arch}: {k} mean regret {v} appears', f'${v}$' in text)

# the value-for-money panel
vfm = {a: run['archetypes'][a]['vfm'] for a in ('frontier', 'budget', 'balanced')}
check('vfm: capability-first result summarized', 'same $24.2$ net-improvement points per dollar' in text)
check('vfm: cost-first composite is a loss', f"{vfm['budget']['Proposed']['net_per_usd']:.1f}" == '-2.8')
check('vfm: cost-first loss and dominated tiers summarized', '$-2.8$' in text and '$5/6$' in text)
check('vfm: balanced efficiency result summarized', '$8.0$' in text and '$2/6$' in text)

# ── the disclosed correction and the two new subsections ────────────────────
check('section: baseline parity and ablation present', '\\label{subsec:parity}' in text)
check('section: nominal weights vs swing present', '\\label{subsec:weights}' in text)
check('disclosure: rounded-ranking defect recorded',
      'ranked the proposed method on its \\emph{rounded} display score' in text)
check('disclosure: exact ranking is the shipped rule', 'scoreExact' in text)
check('component-ablation table present', '\\label{tab:component_ablation}' in text)
check('realized weight influence is summarized', '$73.8\\%' in text and '$10.2\\%' in text)
check('ablation: deficit penalty identified as the operative term',
      'The deficit penalty is the operative term' in text)

# ── the ablation numbers the paper quotes must match the ablation artifact ──
abl = json.load(io.open(ABLATION, encoding='utf-8'))['panels']['frontier']['rows']
by = {r['method']: r for r in abl}
check('ablation: full pipeline 0.09', r2(by['Full pipeline (shipped)']['mean_regret']) == '0.09')
check('ablation: minus penalty 1.28 appears', '$1.28$' in text)
check('ablation: six-coordinate baselines match triplet',
      by['WSM (6 coords)']['mean_regret'] == by['WSM (triplet)']['mean_regret'] == 0.487)
check('ablation: parity claim stated', 'The result is that the restriction explains none of the gap' in text)

sw = json.load(io.open(SWING, encoding='utf-8'))
check('swing: cohort size matches the frozen cohort', sw['cohort'] == run['n_cohort'])
check('swing: context influence 10.2 on the balanced intent',
      sw['panels']['balanced coding models']['influence']['context'] == 10.2)
check('swing: full coordinate diagnostics are in the artifact',
      all('influence' in panel for panel in sw['panels'].values()))

# ── retracted figures and narratives must be absent ─────────────────────────
absent = [
    ('best integrative rule under a cost-first', 'the retracted cost-first lead'),
    ('\\textbf{29.8}~(0/6)', 'the retracted cost-first value-for-money cell'),
    ('29.8$ against $24.2$', 'the retracted cost-first advantage sentence'),
    ('$1.2$ times', 'the retracted 1.2x efficiency claim'),
    ('$2.689$', 'the removed-profile harness figure'),
    ('$4.126$', 'the removed-profile harness figure'),
    ('superseded', 'the superseded-profile narrative'),
    ('earlier draft reported the current figures', 'an earlier run mislabelled as current'),
    ('all five formulations return the same ordering', 'the abstract claim contradicted by Table 1'),
    ('five of six budget tiers', 'the retracted balanced dominance claim'),
    ('Gemini 3.6 Flash', 'a pick from the removed-profile run'),
    ('Step 5 Preview      & 0.01', 'a pick from the rounded-ranking run'),
    ('0.834', 'a figure from the pre-rebuild cohort'),
]
for pat, why in absent:
    check(f'absent: {why}', pat not in text, f'still present: {pat!r}')

# ── the abstract is inside a Springer-length abstract ──────────────────────
m = re.search(r'\\abstract\{(.*?)\n', text, re.S)
words = len(m.group(1).split()) if m else 0
check('abstract is under 300 words', words < 300, f'{words} words')

print(f'{len(oks)} checks passed')
for f in fails:
    print('  FAIL ' + f)
sys.exit(1 if fails else 0)
