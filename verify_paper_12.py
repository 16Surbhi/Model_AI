"""Validate manuscript structure and required declarations for the paper artifact."""
import io
import os
import re
import sys

MAIN = 'docs/model-compass-technical-draft.tex'
text = io.open(MAIN, encoding='utf-8').read()
checks = []

def check(name, condition):
    checks.append((name, bool(condition)))

inputs = re.findall(r'\\input\{([^}]+)\}', text)
check('all LaTeX input files exist', all(os.path.isfile(os.path.join('docs', i + '.tex')) for i in inputs))
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs = re.findall(r'\\(?:ref|eqref|autoref|Cref|cref)\{([^}]+)\}', text)
check('all internal references resolve', all(r in labels for r in refs))
check('braces are balanced', text.count('{') == text.count('}'))
begins = sorted(re.findall(r'\\begin\{([^}]+)\}', text))
ends = sorted(re.findall(r'\\end\{([^}]+)\}', text))
check('environments are balanced', begins == ends)
check('abstract under 250 words', len(re.sub(r'\\[A-Za-z]+', '', re.search(r'\\abstract\{(.*?)\n', text, re.S).group(1)).split()) < 250)
check('no fourth-level paragraph headings', r'\paragraph{' not in text)
check('coverage and validity sections are inlined', r'\label{subsec:coverage}' in text and r'\label{subsec:threats}' in text)
check('smooth coding factor is defined', 'h(t)=3t^2-2t^3' in text and 'q=0.95' in text)
check('obsolete cutoff claim removed', 'remaining discontinuity in the score' not in text and 'whenever $q < 0.95$' not in text)
check('funding declaration present', r'\bmhead{Funding}' in text)
check('data availability declaration present', r'\bmhead{Data availability}' in text)
check('code availability declaration present', r'\bmhead{Code availability}' in text)
check('AI-use disclosure present', r'\bmhead{AI use in manuscript preparation}' in text)
check('no manuscript-section file dependencies', not inputs)

for name, passed in checks:
    print(('PASS' if passed else 'FAIL') + ' ' + name)
print(f'{sum(p for _, p in checks)}/{len(checks)} checks passed')
sys.exit(0 if all(p for _, p in checks) else 1)
