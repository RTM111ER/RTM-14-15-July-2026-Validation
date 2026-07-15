# RTM 14–15 July 2026 Reproducibility Package

This package contains the audited primary implementation, a second independent implementation, the exact 15 July ledger, exact finite-space result files, the semantic-path supplement, audit documentation, requirements, and the current prepublication paper.

## Important: add the server-generated GPU outputs
The following files were produced on the rented RTX 5090 server and are not present in this ChatGPT workspace. Add them to the repository after downloading them from the server:

- `fixed_letters_audited_v1_1_results.json`
- `joint_letters_audited_v1_1_results.json`
- `sequential_letters_audited_v1_1_results.json`
- `RTM_5090_100M_RUN.log`
- `RTM_5090_SMOKE_TEST.txt` (recommended)
- `RTM_ALL_EXACT_LOG.txt` (recommended)
- `RTM_INDEPENDENT_AUDIT_LOG.txt` (recommended)

## Suggested repository structure
- `code/` — executable validation and audit scripts
- `data/` — exact ledgers and semantic supplement
- `results/` — exact counting and independent audit outputs
- `docs/` — audit report and reproducibility notes
- `paper/` — current prepublication manuscript

Before public release, replace the pending DOI in the paper with the reserved Zenodo DOI.
