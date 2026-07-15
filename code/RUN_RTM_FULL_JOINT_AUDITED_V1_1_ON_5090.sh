#!/usr/bin/env bash
set -euo pipefail

TRIALS="${TRIALS:-1000000000}"
BATCH="${BATCH:-1000000}"
PYTHON="${PYTHON:-python}"
SCRIPT="RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION_AUDITED_V1_1.py"

$PYTHON "$SCRIPT" --mode all-exact --output RTM_14_15_FULL_JOINT_RESULTS_AUDITED_V1_1.json

$PYTHON "$SCRIPT" --mode gpu --gpu-model fixed_letters \
  --trials "$TRIALS" --batch-size "$BATCH" --seed 20260715 \
  --checkpoint fixed_letters_audited_v1_1_checkpoint.json \
  --output fixed_letters_audited_v1_1_results.json

$PYTHON "$SCRIPT" --mode gpu --gpu-model joint_letters \
  --trials "$TRIALS" --batch-size "$BATCH" --seed 20260716 \
  --checkpoint joint_letters_audited_v1_1_checkpoint.json \
  --output joint_letters_audited_v1_1_results.json

$PYTHON "$SCRIPT" --mode gpu --gpu-model sequential_letters \
  --trials "$TRIALS" --batch-size "$BATCH" --seed 20260717 \
  --checkpoint sequential_letters_audited_v1_1_checkpoint.json \
  --output sequential_letters_audited_v1_1_results.json
