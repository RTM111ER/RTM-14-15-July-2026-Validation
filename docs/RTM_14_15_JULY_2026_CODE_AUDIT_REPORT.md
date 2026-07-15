# RTM 14→15 July 2026 — Code Audit Report

**Audit date:** 15 July 2026  
**Primary engine audited:** `RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py`  
**Independent second implementation:** `RTM_14_15_JULY_2026_INDEPENDENT_AUDIT.py`

## Executive finding

The primary engine's **core deterministic logic and exact finite-space counting were independently reproduced**.
The second implementation does not import the primary engine and returned the same:

- 16/16 exact Hebrew strings PASS;
- 19/19 arithmetic checks PASS;
- 14 July observed gate: 9/9 PASS;
- 15 July observed gate: 9/9 PASS;
- 1900–2100 scan: one Day-1 hit (`2026-07-14`), one Day-2 hit (`2026-07-15`), one consecutive pair (`2026-07-14` → `2026-07-15`);
- fixed-date uniform-letter null: `p = 3.3610099353295434e-08`, one-sided `z = 5.3984470544`;
- joint date+observer uniform-letter null: `p = 4.578159391028337e-13`, one-sided `z = 7.1426279938`;
- sequential pair uniform-letter null: `p = 3.815184793939247e-15`, one-sided `z = 7.7735712607`.

These values are therefore **not the product of a single implementation bug**.

## What the code correctly does

### 1. Full Day-2 gate
A Day-2 full hit requires all nine retained same-day families simultaneously:

`A ∧ B ∧ C ∧ D ∧ E ∧ G ∧ H ∧ I ∧ J`

Together these families cover the full retained 15 July network:

- A: 792 → 1584 → 949 → 157 structure through the exact digit phrase for 635;
- B: 1359 − Hebrew date → residual; written residual = date-statement value;
- C: `מציאות מתעדכנת רטרואקטיבית` loop through the parallel date statement back to the Hebrew-date value;
- D: return to the observer full-name-plus-age phrase;
- E: 2378 nexus joining spelled 1359, full date phrase, and spelled full-name value;
- G: return to first-name value;
- H: doubled full-date value + birth year + measurement year + Hebrew date → age;
- I: 1507/157 date representations + 602/44/546 age representations → first-name value;
- J: written first-name value + years → written age.

### 2. Full sequential gate
The sequential event is counted directly as:

`Day1_9/9 ∧ Day2_9/9 ∧ consecutive-day carryover`

The code does **not** multiply separate equation p-values.

### 3. Search acceleration does not drop full hits
For the exact observer nulls, families D and G are used to solve the candidate observer values for a date and age. Every candidate is then passed through the **complete full gate**. Since every full hit must satisfy D and G, this is a valid enumeration shortcut rather than a relaxation of the event.

### 4. Exact name-weight count is correct
Under the declared 3-letter first-name / 4-letter surname uniform-Hebrew-letter null:

- 30 three-letter strings have gematria 320;
- 204 four-letter strings have gematria 292;
- therefore 30 × 204 = **6,120** name-string combinations satisfy the observed first/surname gematria pair.

This independently reproduces the numerator used by the primary engine.

## Important interpretation boundary

The reported p and sigma values are **exact for the declared finite null and frozen template**. They are **not** a universal post-selection-adjusted probability of the historical discovery process.

The current engine does not model, in one global multiplicity correction:

- how many alternative natural question phrasings were considered;
- how many alternative RTM/model-statement phrasings were considered;
- all possible date representations that might have been tried;
- all possible number-word or digit-word representations that might have been tried;
- the unlogged arithmetic-path search and stopping process by which new chains were added.

Therefore the scientifically precise label for 7.1426σ and 7.7736σ is:

> **conditional frozen-template tail score under the declared null model**

not a universal discovery significance.

## Sequential-result accounting

The direct sequential count is valid, but its interpretation needs care.

Under the current uniform-letter model:

- Day-2 joint date+observer: `p = 4.5781593910e-13` (`7.1426σ`)
- full sequential pair: `p = 3.8151847939e-15` (`7.7736σ`)

The ratio is approximately 120, apart from the negligible 73,414-date versus 73,413-pair window adjustment.

This means that **once the full Day-2 gate has already fixed the unique date/observer row, the Day-1 extension mainly adds the apartment-number constraint in this particular finite model**. The result must therefore not be described as multiplying “one unique day” by a second statistically independent “unique day.” The primary engine correctly uses direct joint counting; the wording around the result should preserve that fact.

## Null-model sensitivity issues still requiring dedicated stress tests

### Human-name distribution
The uniform-letter model is mathematically exact but is not an empirical distribution of real Hebrew names. A real-name corpus null could materially change the probability of first-name gematria 320 and surname gematria 292.

### Numeric-support model
The uniform numeric model (`first = 3..1200`, `surname = 4..1600`) is a separate artificial null, not a more authoritative or automatically more conservative model. Its higher sigma should not be presented as the preferred headline result.

### Age and birth-year model
The current null sets `birth_year = measurement_year - age`. This is a declared coherent model, but a broader birth-date model should also be tested.

### Apartment distribution
The sequential null uses a uniform apartment range 1..120. The sequential sigma depends on that declared distribution. Sensitivity to alternative ranges or an empirical apartment-number distribution should be reported.

### Date-string generation
The base date generator uses declared Hebrew month spellings. A second audit tested the combined alternatives:

- `בסיוון`
- `במרחשון`
- `באדר א`
- `באדר ב`

The complete hit structure remained unchanged: Day 1 = 14 July 2026 only; Day 2 = 15 July 2026 only; consecutive pair = 14→15 July 2026 only. Individual family-alone counts changed, which confirms that full-hit robustness and family-frequency robustness are different questions.

## Code defect found and corrected

### GPU checkpoint resume with a changed batch size
The original GPU resume check validated protocol version, model, trial target, and seed, but did not reject a changed batch size. Because the RNG is seeded by batch index, resuming with a different batch size could skip or duplicate parts of the intended sampling schedule.

The audited v1.1 engine now requires the resumed run to use the **identical batch size**.

## Recommended publication language

Acceptable:

> Under the declared frozen-template uniform-Hebrew-letter null, exact finite-space counting gives a one-sided conditional tail score of 7.1426σ for the full 15 July date+observer gate and 7.7736σ for the complete 14→15 July sequential gate. These are direct joint counts, not products of per-equation probabilities.

Also required:

> These values are conditional on the declared null distributions and frozen post-discovery template; they do not constitute a universal post-selection-adjusted probability of the historical discovery process.

## Audit verdict

**Core implementation verdict: PASS.**  
The deterministic reproduction, full-gate logic, exhaustive-date uniqueness, and uniform-letter exact counts were independently reproduced.

**7σ verdict:**  
- **Yes**, the Day-2 joint date+observer result exceeds 7σ **under the declared frozen-template uniform-letter null**.
- **Yes**, the sequential result reaches 7.7736σ **under the declared frozen-template sequential uniform-letter/apartment null**.
- **No**, the current code does not establish a universal post-selection-adjusted “7.8σ discovery significance.” That requires additional modeling of selection freedom and independent null families.
