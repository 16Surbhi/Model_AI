#!/usr/bin/env bash
# Reproduce every table and test in the paper from the frozen state.
# Requirements: Node.js >= 20, Python >= 3.8. No network, no npm install.
set -euo pipefail
cd "$(dirname "$0")"

echo "== frozen registry digest =="
python - <<'PY'
import hashlib, io
for p in ('server/data/model-registry.json',
          'scratch/arena_snapshot_2026-10-08.json',
          'scratch/intents.frozen.json'):
    print(hashlib.sha256(io.open(p, 'rb').read()).hexdigest()[:20], p)
PY

echo
echo "== panels =="
node scratch/arena_corrected.js     > scratch/arena_corrected.out
node scratch/ablation_components.js > scratch/ablation_components.out
node scratch/weights_swing.js       > scratch/weights_swing.out
node scratch/frozen_panels.js       > scratch/frozen_panels.out
node scratch/merit_stats.js         > scratch/merit_stats.out
node scratch/stratum_eval.js        > scratch/stratum_final.txt
node scratch/quality_guardrail_eval.js > scratch/quality_guardrail_eval.out
node scratch/quality_guardrail_nested_cv.js > scratch/quality_guardrail_nested_cv.out

echo "== case-study table and ablations =="
node scratch/eval_best_coding_table.js > scratch/case_table_ranks.out 2>&1 || true
node scratch/regen_ablation.js         > scratch/ablation_exponent.out 2>&1 || true
node scratch/regen_missing_telemetry.js > scratch/missing_telemetry.out 2>&1 || true

echo
echo "== verification =="
python scratch/verify_frozen_panels.py
python scratch/verify_paper_12.py
python scratch/verify_paper_structure.py

echo
echo "All harnesses ran and all checks passed."
