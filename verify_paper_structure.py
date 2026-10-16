#!/usr/bin/env python3
r"""Assembled-document structure and build check.

Static checks that always run: the manuscript plus every \input it pulls in is
read as one document and its braces, environments and math shifts are counted,
plus three failure modes that a brace count cannot see and that have each broken
the real build at least once:

  * mis-doubled command escapes (\\\\(section where \\(section is meant) -- this
    arrived in docs/coverage.tex and stopped the build with
    "There's no line here to end",
  * corrupted tabular column specs (\\{, where @{ is meant), which LaTeX rejects
    with "Illegal pream-token",
  * non-ASCII characters, which a pdflatex build at a journal can render as
    missing glyphs.

If a TeX engine is on PATH the document is then actually compiled in a temporary
directory and the compile must finish with no error and no LaTeX warning. Without
an engine that step is skipped and says so, rather than reporting a pass.
"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

MAIN = 'docs/model-compass-technical-draft.tex'
main = io.open(MAIN, encoding='utf-8').read()

whole = main
for inc in re.findall(r'\\input\{([^}]+)\}', main):
    p = os.path.join('docs', inc + '.tex')
    if not os.path.isfile(p):
        print('MISSING INPUT ' + p)
        sys.exit(1)
    whole += '\n' + io.open(p, encoding='utf-8').read()

bad = 0
ob, cb = whole.count('{'), whole.count('}')
print('assembled braces: { %d } %d  %s' % (ob, cb, 'BALANCED' if ob == cb else '*** UNBALANCED ***'))
bad += ob != cb

b = sorted(re.findall(r'\\begin\{([^}]+)\}', whole))
e = sorted(re.findall(r'\\end\{([^}]+)\}', whole))
only_b = [x for x in b if x not in e]
only_e = [x for x in e if x not in b]
print('assembled environments: %s  only-begin=%s only-end=%s'
      % ('BALANCED' if b == e else '*** MISMATCH ***', only_b, only_e))
bad += b != e

# every math shift must balance: count unescaped $ modulo 2
unescaped = len(re.findall(r'(?<!\\)\$', whole))
print('unescaped $ count: %d  %s' % (unescaped, 'EVEN' if unescaped % 2 == 0 else '*** ODD ***'))
bad += unescaped % 2 != 0

# --- mis-doubled command escapes: a run of backslashes before a command name ---
mis_doubled = []
for path in ['docs/model-compass-technical-draft.tex', 'docs/coverage.tex',
             'docs/threats-to-validity.tex']:
    if not os.path.isfile(path):
        continue
    text = io.open(path, encoding='utf-8').read()
    for m in re.finditer(r'\\{2,}(?=[A-Za-z@%&_#${])', text):
        mis_doubled.append('%s:%d' % (path, text.count('\n', 0, m.start()) + 1))
print('mis-doubled command escapes:', len(mis_doubled), mis_doubled[:5])
bad += len(mis_doubled) > 0

# --- corrupted tabular column specs (@{ written as \{) ---
bad_pream = []
for i, line in enumerate(main.split('\n'), 1):
    if re.search(r'\\begin\{tabular', line) and re.search(r'\\\{', line):
        bad_pream.append(i)
print('corrupted tabular preambles:', len(bad_pream), bad_pream[:5])
bad += len(bad_pream) > 0

# --- non-ASCII characters (a pdflatex build at the journal can drop them) ---
non_ascii = {}
for path in ['docs/model-compass-technical-draft.tex', 'docs/coverage.tex',
             'docs/threats-to-validity.tex']:
    if not os.path.isfile(path):
        continue
    text = io.open(path, encoding='utf-8').read()
    n = sum(1 for c in text if ord(c) > 127)
    if n:
        non_ascii[path] = n
print('non-ASCII characters in sources:', non_ascii if non_ascii else 0)
bad += len(non_ascii) > 0

# --- the artifact bundle must carry the same manuscript as docs/ ---
drift = []
for path in glob.glob('docs/*.tex'):
    twin = os.path.join('artifact', path)
    if os.path.isfile(twin):
        if io.open(path, encoding='utf-8').read() != io.open(twin, encoding='utf-8').read():
            drift.append(path)
print('artifact/docs copies that have drifted from docs/:', drift if drift else 0)
bad += len(drift) > 0

print('stray coverage-and-threats references:', main.count('coverage-and-threats'))
print('old combined file removed:', not os.path.exists('docs/coverage-and-threats.tex'))
print('pre-patch backup present:', os.path.exists('scratch/backup_pre_patch12.tex'))
print('assembled line count:', len(whole.split('\n')))
bad += main.count('coverage-and-threats')

# --- real compile, when an engine is available ---
engine = None
if shutil.which('latexmk'):
    engine = ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error']
elif shutil.which('pdflatex'):
    engine = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error']
elif shutil.which('tectonic'):
    engine = ['tectonic', '-k']

if not engine:
    print('compile check: SKIPPED (no TeX engine on PATH -- install TeX Live/MiKTeX '
          'or tectonic to enable it)')
else:
    with tempfile.TemporaryDirectory() as td:
        for f in glob.glob('docs/*.tex') + glob.glob('docs/*.cls'):
            shutil.copy(f, td)
        proc = subprocess.run(engine + ['model-compass-technical-draft.tex'], cwd=td,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors='replace')
        out = proc.stdout or ''
        warns = [l for l in out.splitlines() if 'LaTeX Warning' in l]
        over = [l for l in out.splitlines() if 'Overfull' in l]
        halted = proc.returncode != 0
        no_bundle = 'bundle' in out.lower() and 'download' in out.lower()
        print('compile check (%s): exit=%d  latex-warnings=%d  overfull=%d'
              % (engine[0], proc.returncode, len(warns), len(over)))
        if halted and no_bundle:
            print('compile check: SKIPPED (engine could not fetch its TeX bundle)')
        else:
            for w in warns[:5]:
                print('  ' + w.strip())
            bad += halted or len(warns) > 0

print('structure checks ' + ('FAILED' if bad else 'passed'))
sys.exit(1 if bad else 0)
