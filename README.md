# RTM 14–15 July 2026 Reproducibility Package

This repository contains the reproducibility package for the RTM 14–15 July 2026 sequential validation.

It includes:

- the audited primary implementation;
- a second independent implementation;
- the exact 15 July 2026 Hebrew-string ledger;
- deterministic audit results;
- exhaustive Gregorian-date scans;
- exact finite-space null calculations;
- completed GPU Monte Carlo outputs;
- the additional deterministic semantic pathway;
- audit and reproduction documentation;
- and the validated research paper.

The 15 July paper should be read in direct continuity with the preceding RTM research note of 14 July 2026.

## Associated Zenodo records

### 14 July 2026 — prior timestamped record

Eran Harpaz,  
*Observer-Linked Arithmetic Convergence in a Same-Day Hebrew Date Query: An RTM Case Report from 14 July 2026.*

**DOI:** `10.5281/zenodo.21363194`

### 15 July 2026 — current validated record

Eran Harpaz,  
*Same-Day Observer-Linked Convergence and Cross-Day Numerical Carryover on 15 July 2026.*

**DOI:** `10.5281/zenodo.21384203`

---

## Main validated results

### Deterministic audit

- Exact Hebrew strings: **16/16 PASS**
- Arithmetic relations: **19/19 PASS**
- Total deterministic checks: **35/35 PASS**
- Recorded failures: **0**

### Exhaustive date scan

The frozen date generator was applied to every Gregorian date from:

`1900-01-01` through `2100-12-31`

Total dates scanned:

`73,414`

Results:

- Complete Day-1 9/9 match: **14 July 2026 only**
- Complete Day-2 9/9 match: **15 July 2026 only**
- Complete consecutive Day-1 → Day-2 pair: **14→15 July 2026 only**

Total consecutive date pairs scanned:

`73,413`

The full structure was evaluated as a joint gate. Individual equation probabilities were not multiplied as though they were independent events.

---

## Exact finite-space null results

### Complete 15 July date-and-observer structure

#### Uniform-letter joint null

- `p = 4.578159391028337 × 10^-13`
- `z = 7.142628σ`

#### Numeric joint null

- `p = 9.752961688128750 × 10^-14`
- `z = 7.352139σ`

### Complete sequential 14→15 July structure

#### Uniform-letter sequential null

- `p = 3.815184793939247 × 10^-15`
- `z = 7.773571σ`

#### Numeric sequential null

- `p = 8.127578782280660 × 10^-16`
- `z = 7.967016σ`

These values were obtained through direct finite-space counting of the complete frozen joint gates. They were not obtained by multiplying nominal probabilities assigned to individual equations.

---

## Independent implementation

A second implementation was written separately from the primary validation engine.

It independently reproduced:

- the 16 exact Hebrew-string values;
- the 19 arithmetic relations;
- the complete Day-1 gate;
- the complete Day-2 gate;
- the exhaustive date scan;
- the unique consecutive-pair result;
- and the exact uniform-letter null calculations.

The independent implementation returned the same central results:

- `5.398447σ` for the fixed-date observer structure;
- `7.142628σ` for the joint 15 July date-and-observer structure;
- and `7.773571σ` for the complete sequential 14→15 July structure.

---

## GPU Monte Carlo validation

The repository includes completed GPU-accelerated Monte Carlo outputs for three declared models.

A total of:

`300,000,000 trial evaluations`

were completed.

### Fixed-date observer model

- Trials: `100,000,000`
- Full hits: `4`
- Exact expected hits: approximately `3.361`
- Best score: `9/9`

The observed four full hits closely matched the exact finite-space expectation.

### Joint date-observer model

- Trials: `100,000,000`
- Full hits: `0`
- Best score: `4/9`

### Sequential 14→15 model

- Trials: `100,000,000`
- Full hits: `0`
- Best score: `8/9`

The GPU runs provide a large-scale computational cross-check of the implementation. The reported 7+ sigma values come from exact finite-space counting rather than from treating zero Monte Carlo hits as a direct measurement of probabilities near `10^-13` or `10^-15`.

---

## Repository structure

### `code/`

Executable validation and audit scripts:

- `RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION_AUDITED_V1_1.py`
- `RTM_14_15_JULY_2026_INDEPENDENT_AUDIT.py`
- `RTM_15_JULY_2026_SEMANTIC_PATHWAY_AUDIT.py`
- `RUN_RTM_FULL_JOINT_AUDITED_V1_1_ON_5090.sh`

### `data/`

Exact machine-readable inputs:

- `RTM_15_JULY_2026_EXACT_LEDGER.json`
- `RTM_15_JULY_2026_SEMANTIC_PATHWAY_SUPPLEMENT.json`

### `results/`

Exact-counting, independent-audit, semantic-pathway, and GPU outputs:

- `RTM_14_15_FULL_JOINT_RESULTS_AUDITED_V1_1.json`
- `RTM_14_15_INDEPENDENT_AUDIT_RESULTS.json`
- `RTM_15_JULY_2026_SEMANTIC_PATHWAY_AUDIT_OUTPUT.txt`
- `fixed_letters_audited_v1_1_results.json`
- `joint_letters_audited_v1_1_results.json`
- `sequential_letters_audited_v1_1_results.json`

### `docs/`

Audit and reproducibility documentation:

- `RTM_14_15_JULY_2026_CODE_AUDIT_REPORT.md`
- `RTM_14_15_JULY_2026_FULL_JOINT_README.md`

### `paper/`

The validated 15 July 2026 research paper associated with:

**DOI:** `10.5281/zenodo.21384203`

---

## Installation

Python 3.10 or later is recommended.

Install the required packages from the repository root:

```bash
python -m pip install -r RTM_14_15_JULY_2026_REQUIREMENTS.txt
