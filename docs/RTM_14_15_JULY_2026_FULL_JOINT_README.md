# RTM 14→15 July 2026 — Full Joint Validation

This package tests the **complete locked structures**, not isolated equations.

## What counts as a hit

### Day 2 — 15 July full network
A full hit requires **all nine same-day families at once**:

`A & B & C & D & E & G & H & I & J`

No partial score is counted as a full hit.

### Sequential 14→15 event
A full sequential hit requires all of the following at once:

1. previous day satisfies the locked **14 July 9/9** structure;
2. next day satisfies the locked **15 July full 9/9** structure;
3. the documented chronological carryover is present.

The carryover is reported as chronology, **not multiplied as an extra independent equation probability**.

## Deterministic self-test

The script refuses a GPU run if the observed record does not first reproduce exactly.
Expected baseline:

- 16/16 exact Hebrew strings PASS
- 19/19 arithmetic checks PASS
- total deterministic audit: **35/35 PASS**
- 14 July observed structure: **9/9 PASS**
- 15 July observed structure: **9/9 PASS**
- full sequential 14→15 gate: **PASS**

## Frozen date scan

The default exhaustive window is 1900-01-01 through 2100-12-31:

- 73,414 dates
- 73,413 consecutive date pairs

The code generates each candidate date mechanically under the declared conventions and applies the same full gates.

## Exact finite-space null models

The program includes exact combinatorial counting, so the main finite-space probabilities do **not** depend on Monte Carlo resolution.

### `letters`

- first name: 3 independent letters, uniform over the 22 ordinary Hebrew letters;
- surname: 4 independent letters, same distribution;
- age: uniform integer 18..90;
- birth year: `measurement_year - age`.

### `numeric`

- first-name gematria: uniform 3..1200;
- surname gematria: uniform 4..1600;
- age: uniform integer 18..90;
- birth year: `measurement_year - age`.

### Sequential extension

- apartment: uniform integer 1..120;
- RTM anchor 116 stays fixed, matching the 14 July record.

These are explicit conditional null models, not universal population models.

## Run the exact validation first

Place these two files in the same directory:

- `RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py`
- `RTM_15_JULY_2026_EXACT_LEDGER.json`

Then run:

```bash
python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py \
  --mode all-exact \
  --output RTM_14_15_FULL_JOINT_RESULTS_EXACT.json
```

## GPU Monte Carlo cross-check on a CUDA server

Use a Python environment with `pyluach`, `numpy`, `scipy`, and a CUDA-enabled `torch` build.

### Fixed 15 July + random 3-letter/4-letter Hebrew-letter observer

```bash
python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py \
  --mode gpu \
  --gpu-model fixed_letters \
  --trials 1000000000 \
  --batch-size 2000000 \
  --seed 20260715 \
  --checkpoint fixed_letters_checkpoint.json \
  --output fixed_letters_results.json
```

### Random date + random observer — all 15 July-style closures jointly

```bash
python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py \
  --mode gpu \
  --gpu-model joint_letters \
  --trials 1000000000 \
  --batch-size 2000000 \
  --seed 20260716 \
  --checkpoint joint_letters_checkpoint.json \
  --output joint_letters_results.json
```

### Full sequential pair: day-1 9/9 + day-2 9/9 + carryover

```bash
python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py \
  --mode gpu \
  --gpu-model sequential_letters \
  --trials 1000000000 \
  --batch-size 1000000 \
  --seed 20260717 \
  --checkpoint sequential_letters_checkpoint.json \
  --output sequential_letters_results.json
```

Numeric-support versions are available as:

- `fixed_numeric`
- `joint_numeric`
- `sequential_numeric`

## Resume a rented-server run

The checkpoint stores the next batch and accumulated counts. Re-run the identical command with `--resume`:

```bash
python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py \
  --mode gpu \
  --gpu-model sequential_letters \
  --trials 1000000000 \
  --batch-size 1000000 \
  --seed 20260717 \
  --checkpoint sequential_letters_checkpoint.json \
  --output sequential_letters_results.json \
  --resume
```

## Important statistical point

A zero-hit Monte Carlo run cannot by itself establish an arbitrarily large sigma. Its tail resolution is limited by the number of trials. That is why this package also performs **exact finite-space counting** under the declared discrete null models. The GPU run is a large-scale empirical cross-check of the same full gate.

## Files produced in this package

- `RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py` — complete engine
- `RTM_15_JULY_2026_EXACT_LEDGER.json` — exact 15 July strings and arithmetic ledger
- `RTM_14_15_FULL_JOINT_RESULTS_EXACT.json` — independently generated exact results
- `RTM_14_15_FULL_JOINT_RUN_LOG.txt` — human-readable run log
