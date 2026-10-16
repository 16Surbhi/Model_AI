#!/usr/bin/env python3
"""Point the artifact harnesses at the pinned intent vectors.

The hosted parser is not part of a public reproduction, so every
`recommendModels(...)` call in the bundied harnesses becomes `pinnedIntent(...)`.
Arithmetic, cohort construction and evaluation are untouched.

Run: python scratch/use-pinned-intents.py
"""
import io
import os
import re

TARGETS = ['arena_corrected.js', 'ablation_components.js', 'weights_swing.js', 'parser_eval.js', 'merit_stats.js']
CALL = re.compile(r'(?:await\s+)?recommendModels\(reg\.models,\s*\{\s*prompt(?:\s*,\s*limit:\s*\d+)?\s*\}\)')
IMPORT_LINE = "import { pinnedIntent } from './pinned-intents.js'\n"

changed = []
for name in TARGETS:
    p = os.path.join('scratch', name)
    if not os.path.isfile(p):
        continue
    src = io.open(p, encoding='utf-8').read()
    if 'pinnedIntent' in src:
        continue
    n = len(CALL.findall(src))
    if not n:
        continue
    src = CALL.sub('pinnedIntent(prompt)', src)
    # insert the import after the last top-level import
    lines = src.split('\n')
    last = max(i for i, l in enumerate(lines) if l.startswith('import '))
    lines.insert(last + 1, IMPORT_LINE.rstrip('\n'))
    io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    changed.append(f'{name}: {n} call(s) pinned')

print('\n'.join(changed) if changed else 'nothing to change')
