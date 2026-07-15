#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTM 14→15 July 2026 — Full Joint Validation Engine
===================================================

Purpose
-------
This program independently validates the exact 15 July 2026 ledger and then tests
FULL-NETWORK hits. A "full hit" is never defined as a sum of separate p-values.
It is a single Boolean AND gate over every locked closure family.

DAY-2 full same-day gate (15 July structure):
    A & B & C & D & E & G & H & I & J

Sequential 14→15 gate:
    DAY1_9_OF_9 & DAY2_FULL_9_OF_9 & CHRONOLOGICAL_CARRYOVER

The chronological carryover is reported explicitly but is not counted as an extra
independent equation or multiplied as a separate p-value.

The script provides four layers:
  1) Exact deterministic audit: 16 strings + 19 arithmetic checks = 35/35.
  2) Exhaustive 1900-2100 date scan under frozen date-construction rules.
  3) Exact combinatorial null calculations (no Monte Carlo approximation) for:
       - fixed observed date + random observer,
       - random date + random observer,
       - consecutive date pair + random observer + random apartment.
  4) Optional CUDA/PyTorch Monte Carlo cross-checks for very large trial counts.

Null-model note
---------------
Every probability/sigma produced here is CONDITIONAL on the declared null model.
The exact combinatorial models are fully enumerated over their finite sample spaces.
The GPU Monte Carlo is a cross-check and resolution-limited by the number of trials.

Default observer nulls
----------------------
UNIFORM_HEBREW_LETTERS:
  - first name: 3 independent letters, each uniform over the 22 ordinary Hebrew letters
  - surname:    4 independent letters, same distribution
  - age:        uniform integer 18..90
  - birth year: measurement_year - age (biographical coherence preserved)

NUMERIC_SUPPORT:
  - first-name gematria uniform on 3..1200 (support of 3 letters, 1..400 each)
  - surname gematria uniform on 4..1600 (support of 4 letters)
  - age uniform 18..90
  - birth year = measurement_year - age

Sequential-pair extension:
  - apartment number uniform integer 1..120
  - RTM anchor 116 remains fixed, matching the 14 July record

Dependencies
------------
Required: Python 3.10+, pyluach, numpy, scipy
Optional GPU: torch with CUDA support

Examples
--------
  python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py --mode all-exact
  python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py --mode gpu \
      --gpu-model fixed_letters --trials 1000000000 --batch-size 2000000
  python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py --mode gpu \
      --gpu-model joint_letters --trials 1000000000 --batch-size 2000000
  python RTM_14_15_JULY_2026_FULL_JOINT_VALIDATION.py --mode gpu \
      --gpu-model sequential_letters --trials 1000000000 --batch-size 1000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pyluach import dates as pyluach_dates
from scipy.stats import beta as beta_dist
from scipy.stats import norm

try:
    import torch
except Exception:  # GPU mode will fail with a clear message later.
    torch = None


# -----------------------------------------------------------------------------
# Frozen protocol constants
# -----------------------------------------------------------------------------

PROTOCOL_VERSION = "RTM-14-15-JULY-2026-FULL-JOINT-v1.1-AUDITED"

STATISTICAL_SCOPE = {
    "claim_type": "conditional frozen-template null probability",
    "discovery_process_modeled": False,
    "phrase_search_multiplicity_modeled": False,
    "representation_search_multiplicity_modeled": False,
    "arithmetic_path_search_multiplicity_modeled": False,
    "warning": (
        "Reported p and sigma values are exact only for the declared finite null model and frozen "
        "template. They are not a universal post-selection-adjusted probability of the historical "
        "discovery process."
    ),
}
WINDOW_START = date(1900, 1, 1)
WINDOW_END = date(2100, 12, 31)
OBSERVED_DAY1 = date(2026, 7, 14)
OBSERVED_DAY2 = date(2026, 7, 15)

AGE_MIN = 18
AGE_MAX = 90
APARTMENT_MIN = 1
APARTMENT_MAX = 120
FIRST_NUMERIC_MIN = 3
FIRST_NUMERIC_MAX = 1200
SURNAME_NUMERIC_MIN = 4
SURNAME_NUMERIC_MAX = 1600

OBS_FIRST = 320
OBS_SURNAME = 292
OBS_FULL = 612
OBS_AGE = 44
OBS_BIRTH = 1982
OBS_APARTMENT = 58
RTM_FIXED_ANCHOR = 116

LETTER_VALUES: Dict[str, int] = {
    "א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9,
    "י": 10, "כ": 20, "ך": 20, "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50,
    "ס": 60, "ע": 70, "פ": 80, "ף": 80, "צ": 90, "ץ": 90, "ק": 100,
    "ר": 200, "ש": 300, "ת": 400,
}

HEBREW_LETTER_VALUE_SUPPORT = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400],
    dtype=np.int64,
)

QUESTION_TEXT = "מה התאריך העברי של היום"
MODEL_STATEMENT_TEXT = "מציאות מתעדכנת רטרואקטיבית"
BEN_TEXT = "בן"
APARTMENT_LABEL_TEXT = "דירה"

# Exact Hebrew month token used inside the unpointed Hebrew-date calculation string.
# Pyluach month numbering: 1=Nisan, ..., 7=Tishrei, 12=Adar/Adar I, 13=Adar II.
HEBREW_MONTH_TOKENS = {
    1: "בניסן",
    2: "באייר",
    3: "בסיון",
    4: "בתמוז",
    5: "באב",
    6: "באלול",
    7: "בתשרי",
    8: "בחשון",
    9: "בכסלו",
    10: "בטבת",
    11: "בשבט",
    12: "באדר",
    13: "באדר ב",
}

GREGORIAN_MONTH_TOKENS = {
    1: "בינואר",
    2: "בפברואר",
    3: "במרץ",
    4: "באפריל",
    5: "במאי",
    6: "ביוני",
    7: "ביולי",
    8: "באוגוסט",
    9: "בספטמבר",
    10: "באוקטובר",
    11: "בנובמבר",
    12: "בדצמבר",
}

ONES_FEM = {
    0: "",
    1: "אחת",
    2: "שתיים",
    3: "שלוש",
    4: "ארבע",
    5: "חמש",
    6: "שש",
    7: "שבע",
    8: "שמונה",
    9: "תשע",
}
TEENS_FEM = {
    10: "עשר",
    11: "אחת עשרה",
    12: "שתים עשרה",
    13: "שלוש עשרה",
    14: "ארבע עשרה",
    15: "חמש עשרה",
    16: "שש עשרה",
    17: "שבע עשרה",
    18: "שמונה עשרה",
    19: "תשע עשרה",
}
TENS = {
    20: "עשרים",
    30: "שלושים",
    40: "ארבעים",
    50: "חמישים",
    60: "שישים",
    70: "שבעים",
    80: "שמונים",
    90: "תשעים",
}
HUNDREDS = {
    100: "מאה",
    200: "מאתיים",
    300: "שלוש מאות",
    400: "ארבע מאות",
    500: "חמש מאות",
    600: "שש מאות",
    700: "שבע מאות",
    800: "שמונה מאות",
    900: "תשע מאות",
}
DIGIT_WORDS_MASC = {
    0: "אפס",
    1: "אחד",
    2: "שתיים",
    3: "שלוש",
    4: "ארבע",
    5: "חמש",
    6: "שש",
    7: "שבע",
    8: "שמונה",
    9: "תשע",
}


def gematria(text: str) -> int:
    """Standard Mispar Hechrachi; spaces/punctuation contribute zero."""
    return sum(LETTER_VALUES.get(ch, 0) for ch in text)


def _under_100(n: int) -> str:
    if not 0 <= n < 100:
        raise ValueError(f"_under_100 out of range: {n}")
    if n < 10:
        return ONES_FEM[n]
    if n < 20:
        return TEENS_FEM[n]
    tens = (n // 10) * 10
    ones = n % 10
    if ones == 0:
        return TENS[tens]
    return f"{TENS[tens]} ו{ONES_FEM[ones]}"


def _under_1000(n: int) -> str:
    if not 0 <= n < 1000:
        raise ValueError(f"_under_1000 out of range: {n}")
    if n < 100:
        return _under_100(n)
    hundreds = (n // 100) * 100
    rem = n % 100
    if rem == 0:
        return HUNDREDS[hundreds]
    # Locked convention matching the record:
    # 612 -> "שש מאות ושתים עשרה"
    # 567 -> "חמש מאות שישים ושבע"
    # 320 -> "שלוש מאות עשרים"
    connector = " ו" if rem < 20 else " "
    return f"{HUNDREDS[hundreds]}{connector}{_under_100(rem)}"


def number_words(n: int) -> str:
    """
    Locked feminine/general number-word convention needed by the record and scan.
    Supports 0..9999, with exact forms required by the 14/15 July ledgers.
    """
    if not 0 <= n <= 9999:
        raise ValueError(f"number_words supports 0..9999, got {n}")
    if n < 1000:
        return _under_1000(n)
    thousands = n // 1000
    rem = n % 1000
    if thousands == 1:
        prefix = "אלף"
    elif thousands == 2:
        prefix = "אלפיים"
    else:
        # This branch is not used by the locked 1900-2100 date phrases, but is
        # defined deterministically for lookup-table completeness.
        special = {
            3: "שלושת אלפים",
            4: "ארבעת אלפים",
            5: "חמשת אלפים",
            6: "ששת אלפים",
            7: "שבעת אלפים",
            8: "שמונת אלפים",
            9: "תשעת אלפים",
        }
        prefix = special[thousands]
    return prefix if rem == 0 else f"{prefix} {_under_1000(rem)}"


def digit_words(n: int, one_form: str = "אחד") -> str:
    if n < 0:
        raise ValueError("digit_words requires non-negative integer")
    out: List[str] = []
    for ch in str(n):
        d = int(ch)
        if d == 1:
            out.append(one_form)
        else:
            out.append(DIGIT_WORDS_MASC[d])
    return " ".join(out)


def compact_day_month(d: date) -> int:
    """Locked compact convention: day followed by non-zero-padded month; 15/7 -> 157."""
    return int(f"{d.day}{d.month}")


def ddmm_zero_padded(d: date) -> int:
    """Locked zero-padded DDMM convention: 15/07 -> 1507."""
    return int(f"{d.day:02d}{d.month:02d}")


def hebrew_date_value(d: date) -> int:
    """
    Gematria value of the locked unpointed Hebrew-date calculation string.

    The string is represented structurally as:
        Hebrew day numeral + exact ב<month> token + Hebrew year without the 5000s.

    Gematria is therefore exactly:
        Hebrew day number + gematria(month token) + (Hebrew year mod 1000).

    This reproduces:
        2026-07-14 -> כט בתמוז תשפו -> 1270
        2026-07-15 -> א באב תשפו   -> 792
    """
    h = pyluach_dates.GregorianDate(d.year, d.month, d.day).to_heb()
    month_token = HEBREW_MONTH_TOKENS[h.month]
    return h.day + gematria(month_token) + (h.year % 1000)


def full_gregorian_date_phrase(d: date) -> str:
    return f"{number_words(d.day)} {GREGORIAN_MONTH_TOKENS[d.month]} {number_words(d.year)}"


def date_statement_phrase(d: date, one_form: str) -> str:
    return f"תאריך היום הוא {digit_words(compact_day_month(d), one_form=one_form)}"


Q = gematria(QUESTION_TEXT)
MODEL_STATEMENT = gematria(MODEL_STATEMENT_TEXT)
BEN = gematria(BEN_TEXT)
APARTMENT_LABEL = gematria(APARTMENT_LABEL_TEXT)
Q_WORDS = gematria(number_words(Q))


# -----------------------------------------------------------------------------
# Exact ledger audit
# -----------------------------------------------------------------------------

DEFAULT_LEDGER_NAME = "RTM_15_JULY_2026_EXACT_LEDGER.json"


def safe_arithmetic_eval(expr: str) -> int:
    allowed = set("0123456789+- ()")
    if any(ch not in allowed for ch in expr):
        raise ValueError(f"Unsafe expression: {expr}")
    return int(eval(expr, {"__builtins__": {}}, {}))


def find_ledger(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
    else:
        p = Path(__file__).with_name(DEFAULT_LEDGER_NAME)
    if not p.exists():
        raise FileNotFoundError(
            f"Ledger not found: {p}. Place {DEFAULT_LEDGER_NAME} next to this script or pass --ledger."
        )
    return p


def run_deterministic_audit(ledger_path: Path, verbose: bool = True) -> Dict[str, object]:
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    failures: List[str] = []
    string_rows = []
    arithmetic_rows = []

    h = pyluach_dates.GregorianDate(2026, 7, 15).to_heb()
    calendar_ok = (h.year, h.month, h.day) == (5786, 5, 1)
    if not calendar_ok:
        failures.append("calendar")

    for item in data["strings"]:
        got = gematria(item["hebrew"])
        ok = got == int(item["value"])
        string_rows.append({"id": item["id"], "got": got, "expected": int(item["value"]), "ok": ok})
        if not ok:
            failures.append(item["id"])

    for item in data["arithmetic_checks"]:
        got = safe_arithmetic_eval(item["expression"])
        ok = got == int(item["expected"])
        arithmetic_rows.append({"id": item["id"], "got": got, "expected": int(item["expected"]), "ok": ok})
        if not ok:
            failures.append(item["id"])

    result = {
        "calendar_ok": calendar_ok,
        "strings_passed": sum(r["ok"] for r in string_rows),
        "strings_total": len(string_rows),
        "arithmetic_passed": sum(r["ok"] for r in arithmetic_rows),
        "arithmetic_total": len(arithmetic_rows),
        "total_passed": sum(r["ok"] for r in string_rows) + sum(r["ok"] for r in arithmetic_rows),
        "total_checks": len(string_rows) + len(arithmetic_rows),
        "failures": failures,
        "pass": calendar_ok and not failures,
    }

    if verbose:
        print("=" * 88)
        print("DETERMINISTIC AUDIT")
        print("=" * 88)
        print(f"Calendar 2026-07-15 -> 1 Av 5786: {'PASS' if calendar_ok else 'FAIL'}")
        print(f"Exact strings: {result['strings_passed']}/{result['strings_total']}")
        print(f"Arithmetic:    {result['arithmetic_passed']}/{result['arithmetic_total']}")
        print(f"TOTAL:         {result['total_passed']}/{result['total_checks']} | {'PASS' if result['pass'] else 'FAIL'}")
        if failures:
            print("Failures:", ", ".join(failures))
    return result


# -----------------------------------------------------------------------------
# Lookup helpers
# -----------------------------------------------------------------------------

LOOKUP_MAX = 9999


def build_lookup_tables(max_n: int = LOOKUP_MAX) -> Dict[str, np.ndarray]:
    number_g = np.full(max_n + 1, -1, dtype=np.int64)
    digit_m = np.full(max_n + 1, -1, dtype=np.int64)
    digit_f = np.full(max_n + 1, -1, dtype=np.int64)
    for n in range(max_n + 1):
        number_g[n] = gematria(number_words(n))
        digit_m[n] = gematria(digit_words(n, "אחד"))
        digit_f[n] = gematria(digit_words(n, "אחת"))
    return {"number": number_g, "digit_m": digit_m, "digit_f": digit_f}


LOOKUPS = build_lookup_tables()


def lookup(arr: np.ndarray, n: int) -> Optional[int]:
    if n < 0 or n >= len(arr):
        return None
    v = int(arr[n])
    return None if v < 0 else v


# -----------------------------------------------------------------------------
# Day-2 full-network evaluator: exact AND gate over A,B,C,D,E,G,H,I,J
# -----------------------------------------------------------------------------

DAY2_FAMILIES = ("A", "B", "C", "D", "E", "G", "H", "I", "J")
DAY1_FAMILIES = ("A", "B", "C", "D", "E", "F", "G", "H", "I")


@dataclass(frozen=True)
class Observer:
    first: int
    surname: int
    age: int
    birth_year: int
    apartment: int = OBS_APARTMENT

    @property
    def full(self) -> int:
        return self.first + self.surname


REAL_OBSERVER = Observer(
    first=OBS_FIRST,
    surname=OBS_SURNAME,
    age=OBS_AGE,
    birth_year=OBS_BIRTH,
    apartment=OBS_APARTMENT,
)


def day2_families(d: date, obs: Observer) -> Dict[str, bool]:
    H = hebrew_date_value(d)
    compact = compact_day_month(d)
    ddmm = ddmm_zero_padded(d)
    year = d.year

    residual = Q - H
    residual_words = lookup(LOOKUPS["number"], residual)
    date_stmt_f = gematria(date_statement_phrase(d, "אחת"))
    date_stmt_m = gematria(date_statement_phrase(d, "אחד"))
    full_date_g = gematria(full_gregorian_date_phrase(d))

    x = H - compact
    x_digits = lookup(LOOKUPS["digit_m"], x)
    A = (x_digits is not None) and (x_digits == 2 * H) and ((2 * H - x - H) == compact)

    B = (residual_words is not None) and (residual_words == date_stmt_f)

    loop_residual = MODEL_STATEMENT - date_stmt_f
    C = (date_stmt_m - loop_residual) == H

    age_words = lookup(LOOKUPS["number"], obs.age)
    if age_words is None:
        return {k: False for k in DAY2_FAMILIES}
    observer_age_phrase = obs.full + BEN + age_words

    D = (residual_words is not None) and ((date_stmt_f - residual) == observer_age_phrase)

    full_name_words = lookup(LOOKUPS["number"], obs.full)
    E = (Q_WORDS == full_date_g) and (full_name_words == full_date_g)

    G = (full_date_g - observer_age_phrase - H) == obs.first

    h_temp = 2 * full_date_g - obs.birth_year - year
    H_family = (H - h_temp) == obs.age

    age_digits = lookup(LOOKUPS["digit_m"], obs.age)
    if age_digits is None:
        I = False
    else:
        i1 = ddmm - H
        i2 = i1 - age_words
        i3 = i2 - obs.age
        i4 = age_digits - i3
        I = (i4 - compact) == obs.first

    first_words = lookup(LOOKUPS["number"], obs.first)
    J = (first_words is not None) and (year - (2 * first_words - obs.birth_year) == age_words)

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "G": G, "H": H_family, "I": I, "J": J}


def day2_full_hit(d: date, obs: Observer) -> bool:
    fam = day2_families(d, obs)
    return all(fam[k] for k in DAY2_FAMILIES)


# -----------------------------------------------------------------------------
# Day-1 9/9 evaluator: exact locked 14 July structure
# -----------------------------------------------------------------------------


def day1_families(d: date, obs: Observer, rtm_anchor: int = RTM_FIXED_ANCHOR) -> Dict[str, bool]:
    H = hebrew_date_value(d)
    compact = compact_day_month(d)
    year = d.year

    h_words = lookup(LOOKUPS["number"], H)
    apartment_words = lookup(LOOKUPS["number"], obs.apartment)
    if h_words is None or apartment_words is None:
        return {k: False for k in DAY1_FAMILIES}
    apartment_phrase = APARTMENT_LABEL + apartment_words

    h_digits = lookup(LOOKUPS["digit_m"], H)
    full_words = lookup(LOOKUPS["number"], obs.full)
    full_digits_f = lookup(LOOKUPS["digit_f"], obs.full)
    full_digits_m = lookup(LOOKUPS["digit_m"], obs.full)
    if h_digits is None or full_words is None or full_digits_f is None or full_digits_m is None:
        return {k: False for k in DAY1_FAMILIES}

    gap = Q - H
    gap_words = lookup(LOOKUPS["number"], gap)
    gap_digits = lookup(LOOKUPS["digit_m"], gap)
    if gap_words is None or gap_digits is None:
        return {k: False for k in DAY1_FAMILIES}

    full_date_g = gematria(full_gregorian_date_phrase(d))

    A = h_words == apartment_phrase
    B = Q_WORDS == full_words
    C = (Q_WORDS - h_digits - h_words) == obs.apartment
    D = (compact - gap) == obs.apartment
    E = (gap_words + gap_digits - Q) == apartment_phrase
    F = (full_digits_f - Q - rtm_anchor) == (2 * compact)

    residual = full_digits_m - H
    G = (compact - residual) == obs.age

    H_family = (
        (Q_WORDS - Q)
        + (full_digits_f - Q)
        + (full_digits_m - Q)
        - H
        + compact
        == obs.first
    )

    I = (full_date_g - obs.birth_year - compact - rtm_anchor) == obs.apartment

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "H": H_family, "I": I}


def day1_full_hit(d: date, obs: Observer, rtm_anchor: int = RTM_FIXED_ANCHOR) -> bool:
    fam = day1_families(d, obs, rtm_anchor=rtm_anchor)
    return all(fam[k] for k in DAY1_FAMILIES)


def chronological_carryover(day1: date, day2: date) -> bool:
    """
    The prior-day network value carried forward is Q_WORDS = 2378.
    The day-2 full Gregorian date phrase must equal that already-fixed value.
    This is chronology, not a separately multiplied equation probability.
    """
    return day2 == day1 + timedelta(days=1) and gematria(full_gregorian_date_phrase(day2)) == Q_WORDS


def sequential_full_hit(day1: date, day2: date, obs: Observer) -> bool:
    return (
        day2 == day1 + timedelta(days=1)
        and day1_full_hit(day1, obs)
        and day2_full_hit(day2, obs)
        and chronological_carryover(day1, day2)
    )


# -----------------------------------------------------------------------------
# Date table and exhaustive scans
# -----------------------------------------------------------------------------


def iter_dates(start: date = WINDOW_START, end: date = WINDOW_END) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_date_rows(start: date = WINDOW_START, end: date = WINDOW_END) -> List[Dict[str, int]]:
    rows = []
    for d in iter_dates(start, end):
        rows.append(
            {
                "ordinal": d.toordinal(),
                "year": d.year,
                "month": d.month,
                "day": d.day,
                "H": hebrew_date_value(d),
                "compact": compact_day_month(d),
                "ddmm": ddmm_zero_padded(d),
                "date_stmt_f": gematria(date_statement_phrase(d, "אחת")),
                "date_stmt_m": gematria(date_statement_phrase(d, "אחד")),
                "full_date_g": gematria(full_gregorian_date_phrase(d)),
            }
        )
    return rows


def exhaustive_scan(verbose: bool = True) -> Dict[str, object]:
    day1_hits: List[str] = []
    day2_hits: List[str] = []
    pair_hits: List[Tuple[str, str]] = []
    day2_family_counts = {k: 0 for k in DAY2_FAMILIES}

    dates_list = list(iter_dates())
    for d in dates_list:
        f2 = day2_families(d, REAL_OBSERVER)
        for k in DAY2_FAMILIES:
            day2_family_counts[k] += int(f2[k])
        if all(f2[k] for k in DAY2_FAMILIES):
            day2_hits.append(d.isoformat())
        if day1_full_hit(d, REAL_OBSERVER):
            day1_hits.append(d.isoformat())

    for d1, d2 in zip(dates_list[:-1], dates_list[1:]):
        if sequential_full_hit(d1, d2, REAL_OBSERVER):
            pair_hits.append((d1.isoformat(), d2.isoformat()))

    result = {
        "date_window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "dates_scanned": len(dates_list),
        "pairs_scanned": len(dates_list) - 1,
        "day1_9_of_9_hits": day1_hits,
        "day2_full_9_of_9_hits": day2_hits,
        "sequential_full_hits": pair_hits,
        "day2_family_counts": day2_family_counts,
    }

    if verbose:
        print("\n" + "=" * 88)
        print("EXHAUSTIVE DATE SCAN — FULL AND GATES")
        print("=" * 88)
        print(f"Dates scanned: {len(dates_list):,}")
        print(f"Day-1 exact 9/9 hits: {len(day1_hits)} -> {day1_hits}")
        print(f"Day-2 exact full 9/9 hits: {len(day2_hits)} -> {day2_hits}")
        print(f"Consecutive full pair hits: {len(pair_hits)} -> {pair_hits}")
        print("Day-2 family-alone counts under this frozen generator:")
        for k in DAY2_FAMILIES:
            print(f"  {k}: {day2_family_counts[k]:,}")
    return result


# -----------------------------------------------------------------------------
# Exact combinatorial nulls (finite-space counting, no simulation approximation)
# -----------------------------------------------------------------------------


def sum_distribution(letter_values: Sequence[int], length: int) -> Counter:
    dist = Counter({0: 1})
    for _ in range(length):
        nxt = Counter()
        for subtotal, count in dist.items():
            for v in letter_values:
                nxt[subtotal + int(v)] += count
        dist = nxt
    return dist


FIRST_LETTER_DIST = sum_distribution(HEBREW_LETTER_VALUE_SUPPORT.tolist(), 3)
SURNAME_LETTER_DIST = sum_distribution(HEBREW_LETTER_VALUE_SUPPORT.tolist(), 4)
TOTAL_LETTER_NAMES = 22 ** 7
AGE_COUNT = AGE_MAX - AGE_MIN + 1
APARTMENT_COUNT = APARTMENT_MAX - APARTMENT_MIN + 1
NUMERIC_FIRST_COUNT = FIRST_NUMERIC_MAX - FIRST_NUMERIC_MIN + 1
NUMERIC_SURNAME_COUNT = SURNAME_NUMERIC_MAX - SURNAME_NUMERIC_MIN + 1


@dataclass
class ExactNullResult:
    model: str
    numerator_weight: int
    denominator_weight: int
    p_exact: float
    z_one_sided: float
    matching_parameter_rows: List[Dict[str, int]]




def day2_date_core_pass(d: date) -> bool:
    """Fast observer-independent prefilter for A, B, C and the date side of E."""
    H = hebrew_date_value(d)
    compact = compact_day_month(d)
    residual = Q - H
    date_stmt_f = gematria(date_statement_phrase(d, "אחת"))
    date_stmt_m = gematria(date_statement_phrase(d, "אחד"))
    full_date_g = gematria(full_gregorian_date_phrase(d))

    x = H - compact
    x_digits = lookup(LOOKUPS["digit_m"], x)
    A = (x_digits is not None) and (x_digits == 2 * H) and ((2 * H - x - H) == compact)

    residual_words = lookup(LOOKUPS["number"], residual)
    B = (residual_words is not None) and (residual_words == date_stmt_f)

    loop_residual = MODEL_STATEMENT - date_stmt_f
    C = (date_stmt_m - loop_residual) == H

    E_date = Q_WORDS == full_date_g
    return A and B and C and E_date

def candidate_observer_from_D_G(d: date, age: int) -> Optional[Observer]:
    """
    For a FULL day-2 hit, families D and G fix the required full-name and first-name
    values once date and age are chosen. This is only a search acceleration; the
    returned candidate is still passed through the complete A&B&C&D&E&G&H&I&J gate.
    """
    H = hebrew_date_value(d)
    residual = Q - H
    if residual < 0:
        return None
    date_stmt_f = gematria(date_statement_phrase(d, "אחת"))
    full_date_g = gematria(full_gregorian_date_phrase(d))
    age_words = lookup(LOOKUPS["number"], age)
    if age_words is None:
        return None

    required_observer_age_phrase = date_stmt_f - residual
    full = required_observer_age_phrase - BEN - age_words
    first = full_date_g - required_observer_age_phrase - H
    surname = full - first
    if first < 0 or surname < 0:
        return None
    return Observer(first=first, surname=surname, age=age, birth_year=d.year - age)


def exact_fixed_date_null(model: str = "letters", d: date = OBSERVED_DAY2) -> ExactNullResult:
    numerator = 0
    matches: List[Dict[str, int]] = []

    for age in range(AGE_MIN, AGE_MAX + 1):
        obs = candidate_observer_from_D_G(d, age)
        if obs is None or not day2_full_hit(d, obs):
            continue
        if model == "letters":
            weight = FIRST_LETTER_DIST.get(obs.first, 0) * SURNAME_LETTER_DIST.get(obs.surname, 0)
            denominator = AGE_COUNT * TOTAL_LETTER_NAMES
        elif model == "numeric":
            weight = int(
                FIRST_NUMERIC_MIN <= obs.first <= FIRST_NUMERIC_MAX
                and SURNAME_NUMERIC_MIN <= obs.surname <= SURNAME_NUMERIC_MAX
            )
            denominator = AGE_COUNT * NUMERIC_FIRST_COUNT * NUMERIC_SURNAME_COUNT
        else:
            raise ValueError("model must be 'letters' or 'numeric'")
        if weight:
            numerator += int(weight)
            matches.append({**asdict(obs), "full": obs.full, "weight": int(weight), "date_ordinal": d.toordinal()})

    p = numerator / denominator
    z = float(norm.isf(p)) if p > 0 else math.inf
    return ExactNullResult(
        model=f"fixed_date_{model}",
        numerator_weight=numerator,
        denominator_weight=denominator,
        p_exact=p,
        z_one_sided=z,
        matching_parameter_rows=matches,
    )


def exact_joint_date_observer_null(model: str = "letters") -> ExactNullResult:
    numerator = 0
    matches: List[Dict[str, int]] = []
    n_dates = (WINDOW_END - WINDOW_START).days + 1

    for d in iter_dates():
        if not day2_date_core_pass(d):
            continue
        for age in range(AGE_MIN, AGE_MAX + 1):
            obs = candidate_observer_from_D_G(d, age)
            if obs is None or not day2_full_hit(d, obs):
                continue
            if model == "letters":
                weight = FIRST_LETTER_DIST.get(obs.first, 0) * SURNAME_LETTER_DIST.get(obs.surname, 0)
                denominator = n_dates * AGE_COUNT * TOTAL_LETTER_NAMES
            elif model == "numeric":
                weight = int(
                    FIRST_NUMERIC_MIN <= obs.first <= FIRST_NUMERIC_MAX
                    and SURNAME_NUMERIC_MIN <= obs.surname <= SURNAME_NUMERIC_MAX
                )
                denominator = n_dates * AGE_COUNT * NUMERIC_FIRST_COUNT * NUMERIC_SURNAME_COUNT
            else:
                raise ValueError("model must be 'letters' or 'numeric'")
            if weight:
                numerator += int(weight)
                matches.append(
                    {
                        "date_ordinal": d.toordinal(),
                        "year": d.year,
                        "month": d.month,
                        "day": d.day,
                        **asdict(obs),
                        "full": obs.full,
                        "weight": int(weight),
                    }
                )

    p = numerator / denominator
    z = float(norm.isf(p)) if p > 0 else math.inf
    return ExactNullResult(
        model=f"joint_date_observer_{model}",
        numerator_weight=numerator,
        denominator_weight=denominator,
        p_exact=p,
        z_one_sided=z,
        matching_parameter_rows=matches,
    )


def exact_sequential_pair_null(model: str = "letters") -> ExactNullResult:
    """
    Direct finite-space count of the FULL sequential event:
      previous-day 9/9 AND current-day 9/9 AND carryover,
    under one shared observer and an apartment uniformly drawn from 1..120.

    This routine does not multiply separate line probabilities. It enumerates the
    finite parameter space using the day-2 D/G constraints only as a search shortcut,
    then evaluates the complete joint gate directly.
    """
    numerator = 0
    matches: List[Dict[str, int]] = []
    n_pairs = (WINDOW_END - WINDOW_START).days

    d2 = WINDOW_START + timedelta(days=1)
    while d2 <= WINDOW_END:
        d1 = d2 - timedelta(days=1)
        if not day2_date_core_pass(d2):
            d2 += timedelta(days=1)
            continue
        for age in range(AGE_MIN, AGE_MAX + 1):
            base_obs = candidate_observer_from_D_G(d2, age)
            if base_obs is None or not day2_full_hit(d2, base_obs):
                continue

            if model == "letters":
                name_weight = FIRST_LETTER_DIST.get(base_obs.first, 0) * SURNAME_LETTER_DIST.get(base_obs.surname, 0)
                denominator = n_pairs * AGE_COUNT * TOTAL_LETTER_NAMES * APARTMENT_COUNT
            elif model == "numeric":
                name_weight = int(
                    FIRST_NUMERIC_MIN <= base_obs.first <= FIRST_NUMERIC_MAX
                    and SURNAME_NUMERIC_MIN <= base_obs.surname <= SURNAME_NUMERIC_MAX
                )
                denominator = n_pairs * AGE_COUNT * NUMERIC_FIRST_COUNT * NUMERIC_SURNAME_COUNT * APARTMENT_COUNT
            else:
                raise ValueError("model must be 'letters' or 'numeric'")

            if not name_weight:
                continue

            for apartment in range(APARTMENT_MIN, APARTMENT_MAX + 1):
                obs = Observer(
                    first=base_obs.first,
                    surname=base_obs.surname,
                    age=base_obs.age,
                    birth_year=base_obs.birth_year,
                    apartment=apartment,
                )
                if sequential_full_hit(d1, d2, obs):
                    numerator += int(name_weight)
                    matches.append(
                        {
                            "day1_ordinal": d1.toordinal(),
                            "day2_ordinal": d2.toordinal(),
                            "day1_year": d1.year,
                            "day1_month": d1.month,
                            "day1_day": d1.day,
                            "day2_year": d2.year,
                            "day2_month": d2.month,
                            "day2_day": d2.day,
                            **asdict(obs),
                            "full": obs.full,
                            "weight": int(name_weight),
                        }
                    )
        d2 += timedelta(days=1)

    p = numerator / denominator
    z = float(norm.isf(p)) if p > 0 else math.inf
    return ExactNullResult(
        model=f"sequential_pair_{model}",
        numerator_weight=numerator,
        denominator_weight=denominator,
        p_exact=p,
        z_one_sided=z,
        matching_parameter_rows=matches,
    )


def run_all_exact_nulls(verbose: bool = True) -> Dict[str, object]:
    results = [
        exact_fixed_date_null("letters"),
        exact_fixed_date_null("numeric"),
        exact_joint_date_observer_null("letters"),
        exact_joint_date_observer_null("numeric"),
        exact_sequential_pair_null("letters"),
        exact_sequential_pair_null("numeric"),
    ]
    if verbose:
        print("\n" + "=" * 88)
        print("EXACT FINITE-SPACE NULL RESULTS — FULL JOINT GATES")
        print("=" * 88)
        for r in results:
            print(f"{r.model}")
            print(f"  numerator   = {r.numerator_weight:,}")
            print(f"  denominator = {r.denominator_weight:,}")
            print(f"  p_exact     = {r.p_exact:.16e}")
            print(f"  z_one_sided = {r.z_one_sided:.6f} sigma")
            print(f"  matching parameter rows = {len(r.matching_parameter_rows)}")
    packed = {r.model: asdict(r) for r in results}
    # The sequential model is counted directly, not obtained by multiplying independent day p-values.
    # This ratio is diagnostic only: under the current finite model, once the unique Day-2 date/observer
    # row is fixed, the previous-day gate further selects the apartment value (with a tiny window-size
    # adjustment because there are 73,413 pairs versus 73,414 single dates).
    if packed["joint_date_observer_letters"]["p_exact"] > 0 and packed["sequential_pair_letters"]["p_exact"] > 0:
        packed["sequential_increment_diagnostic"] = {
            "joint_day2_p": packed["joint_date_observer_letters"]["p_exact"],
            "sequential_p": packed["sequential_pair_letters"]["p_exact"],
            "ratio_joint_to_sequential": (
                packed["joint_date_observer_letters"]["p_exact"]
                / packed["sequential_pair_letters"]["p_exact"]
            ),
            "interpretation": (
                "Direct joint counting is authoritative. Do not interpret the two daily uniqueness results "
                "as independent p-values. In this frozen model the sequential extension mainly adds the "
                "apartment constraint after the Day-2 full gate has already fixed the date and observer row."
            ),
        }
    return packed


# -----------------------------------------------------------------------------
# GPU Monte Carlo cross-check
# -----------------------------------------------------------------------------


def date_table_numpy() -> Dict[str, np.ndarray]:
    rows = build_date_rows()
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=np.int64) for k in keys}


DATE_TABLE_NP = None


def _torch_lookup(table, idx, valid):
    safe = idx.clamp(0, table.numel() - 1)
    out = table[safe]
    return out, valid & (idx >= 0) & (idx < table.numel())


def _gpu_day2_masks(date_fields, first, surname, age, birth, lookup_tensors):
    H = date_fields["H"]
    compact = date_fields["compact"]
    ddmm = date_fields["ddmm"]
    year = date_fields["year"]
    dsf = date_fields["date_stmt_f"]
    dsm = date_fields["date_stmt_m"]
    PH = date_fields["full_date_g"]
    full = first + surname

    numg = lookup_tensors["number"]
    digm = lookup_tensors["digit_m"]

    x = H - compact
    xg, vx = _torch_lookup(digm, x, torch.ones_like(x, dtype=torch.bool))
    A = vx & (xg == 2 * H) & ((2 * H - x - H) == compact)

    residual = Q - H
    rw, vr = _torch_lookup(numg, residual, torch.ones_like(residual, dtype=torch.bool))
    B = vr & (rw == dsf)

    loop_residual = MODEL_STATEMENT - dsf
    C = (dsm - loop_residual) == H

    agew, va = _torch_lookup(numg, age, torch.ones_like(age, dtype=torch.bool))
    phrase = full + BEN + agew
    D = va & vr & ((dsf - residual) == phrase)

    fullw, vf = _torch_lookup(numg, full, torch.ones_like(full, dtype=torch.bool))
    E = vf & (Q_WORDS == PH) & (fullw == PH)

    G = (PH - phrase - H) == first

    temp = 2 * PH - birth - year
    Hm = (H - temp) == age

    aged, vad = _torch_lookup(digm, age, torch.ones_like(age, dtype=torch.bool))
    i1 = ddmm - H
    i2 = i1 - agew
    i3 = i2 - age
    i4 = aged - i3
    I = va & vad & ((i4 - compact) == first)

    firstw, vfirst = _torch_lookup(numg, first, torch.ones_like(first, dtype=torch.bool))
    J = vfirst & va & (year - (2 * firstw - birth) == agew)

    masks = [A, B, C, D, E, G, Hm, I, J]
    full_hit = masks[0]
    for m in masks[1:]:
        full_hit = full_hit & m
    score = sum(m.to(torch.int16) for m in masks)
    return full_hit, score


def _gpu_day1_masks(date_fields, first, surname, age, birth, apartment, lookup_tensors):
    H = date_fields["H"]
    compact = date_fields["compact"]
    year = date_fields["year"]
    PH = date_fields["full_date_g"]
    full = first + surname

    numg = lookup_tensors["number"]
    digm = lookup_tensors["digit_m"]
    digf = lookup_tensors["digit_f"]

    hw, vhw = _torch_lookup(numg, H, torch.ones_like(H, dtype=torch.bool))
    aptw, vapt = _torch_lookup(numg, apartment, torch.ones_like(apartment, dtype=torch.bool))
    apt_phrase = APARTMENT_LABEL + aptw

    hd, vhd = _torch_lookup(digm, H, torch.ones_like(H, dtype=torch.bool))
    fw, vfw = _torch_lookup(numg, full, torch.ones_like(full, dtype=torch.bool))
    fdf, vfdf = _torch_lookup(digf, full, torch.ones_like(full, dtype=torch.bool))
    fdm, vfdm = _torch_lookup(digm, full, torch.ones_like(full, dtype=torch.bool))

    gap = Q - H
    gapw, vgw = _torch_lookup(numg, gap, torch.ones_like(gap, dtype=torch.bool))
    gapd, vgd = _torch_lookup(digm, gap, torch.ones_like(gap, dtype=torch.bool))

    A = vhw & vapt & (hw == apt_phrase)
    B = vfw & (Q_WORDS == fw)
    C = vhd & vhw & ((Q_WORDS - hd - hw) == apartment)
    D = (compact - gap) == apartment
    E = vgw & vgd & vapt & ((gapw + gapd - Q) == apt_phrase)
    F = vfdf & ((fdf - Q - RTM_FIXED_ANCHOR) == 2 * compact)
    residual = fdm - H
    G = vfdm & ((compact - residual) == age)
    Hm = vfw & vfdf & vfdm & (((Q_WORDS - Q) + (fdf - Q) + (fdm - Q) - H + compact) == first)
    I = (PH - birth - compact - RTM_FIXED_ANCHOR) == apartment

    masks = [A, B, C, D, E, F, G, Hm, I]
    full_hit = masks[0]
    for m in masks[1:]:
        full_hit = full_hit & m
    score = sum(m.to(torch.int16) for m in masks)
    return full_hit, score


def _sample_names(batch: int, model: str, device: str, gen):
    if model.endswith("letters"):
        vals = torch.tensor(HEBREW_LETTER_VALUE_SUPPORT, device=device, dtype=torch.int64)
        first = torch.zeros(batch, device=device, dtype=torch.int64)
        surname = torch.zeros(batch, device=device, dtype=torch.int64)
        for _ in range(3):
            idx = torch.randint(0, 22, (batch,), generator=gen, device=device)
            first += vals[idx]
        for _ in range(4):
            idx = torch.randint(0, 22, (batch,), generator=gen, device=device)
            surname += vals[idx]
        return first, surname
    if model.endswith("numeric"):
        first = torch.randint(FIRST_NUMERIC_MIN, FIRST_NUMERIC_MAX + 1, (batch,), generator=gen, device=device)
        surname = torch.randint(SURNAME_NUMERIC_MIN, SURNAME_NUMERIC_MAX + 1, (batch,), generator=gen, device=device)
        return first, surname
    raise ValueError(f"Unknown GPU model: {model}")


def monte_carlo_interval(hits: int, trials: int, alpha: float = 0.05) -> Dict[str, float]:
    p_addone = (hits + 1) / (trials + 1)
    if hits == 0:
        lower = 0.0
        upper = 1.0 - alpha ** (1.0 / trials)
    else:
        lower = float(beta_dist.ppf(alpha / 2, hits, trials - hits + 1))
        upper = float(beta_dist.ppf(1 - alpha / 2, hits + 1, trials - hits))
    return {
        "p_addone": p_addone,
        "z_addone_one_sided": float(norm.isf(p_addone)),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "z_lower_bound_from_ci95_upper": float(norm.isf(upper)) if upper > 0 else math.inf,
    }


def run_gpu_monte_carlo(
    gpu_model: str,
    trials: int,
    batch_size: int,
    seed: int,
    checkpoint_path: Path,
    resume: bool,
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install a CUDA-enabled PyTorch build for GPU mode.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")
    if gpu_model not in {
        "fixed_letters", "fixed_numeric",
        "joint_letters", "joint_numeric",
        "sequential_letters", "sequential_numeric",
    }:
        raise ValueError(f"Unsupported --gpu-model: {gpu_model}")

    device = "cuda"
    table_np = date_table_numpy()
    date_tensors = {k: torch.tensor(v, device=device, dtype=torch.int64) for k, v in table_np.items()}
    lookup_tensors = {k: torch.tensor(v, device=device, dtype=torch.int64) for k, v in LOOKUPS.items()}
    n_dates = len(table_np["year"])

    state = {
        "protocol_version": PROTOCOL_VERSION,
        "statistical_scope": STATISTICAL_SCOPE,
        "gpu_model": gpu_model,
        "target_trials": int(trials),
        "batch_size_requested": int(batch_size),
        "seed": int(seed),
        "next_batch": 0,
        "trials_completed": 0,
        "full_hits": 0,
        "best_score": 0,
        "score_histogram": {},
        "device_name": torch.cuda.get_device_name(0),
        "started_unix": time.time(),
    }
    if resume and checkpoint_path.exists():
        old = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key in ("protocol_version", "gpu_model", "target_trials", "batch_size_requested", "seed"):
            if old.get(key) != state.get(key):
                raise RuntimeError(
                    f"Checkpoint mismatch for {key}: {old.get(key)} != {state.get(key)}. "
                    "Resume requires the identical batch size because batch-index seeding is used."
                )
        state.update(old)

    total_batches = math.ceil(trials / batch_size)
    start_batch = int(state["next_batch"])
    print("\n" + "=" * 88)
    print("GPU MONTE CARLO — FULL JOINT GATE")
    print("=" * 88)
    print(f"Device: {state['device_name']}")
    print(f"Model: {gpu_model}")
    print(f"Trials target: {trials:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Resume batch: {start_batch}/{total_batches}")

    observed_idx = (OBSERVED_DAY2 - WINDOW_START).days
    t0 = time.time()

    for batch_idx in range(start_batch, total_batches):
        remaining = trials - batch_idx * batch_size
        b = min(batch_size, remaining)
        if b <= 0:
            break

        # Deterministic batch-level seed makes checkpoints resumable.
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + batch_idx * 1_000_003)

        first, surname = _sample_names(b, gpu_model, device, gen)
        age = torch.randint(AGE_MIN, AGE_MAX + 1, (b,), generator=gen, device=device)

        if gpu_model.startswith("fixed_"):
            idx = torch.full((b,), observed_idx, device=device, dtype=torch.int64)
            fields2 = {k: v[idx] for k, v in date_tensors.items()}
            birth = fields2["year"] - age
            full_hit, score = _gpu_day2_masks(fields2, first, surname, age, birth, lookup_tensors)

        elif gpu_model.startswith("joint_"):
            idx = torch.randint(0, n_dates, (b,), generator=gen, device=device)
            fields2 = {k: v[idx] for k, v in date_tensors.items()}
            birth = fields2["year"] - age
            full_hit, score = _gpu_day2_masks(fields2, first, surname, age, birth, lookup_tensors)

        else:  # sequential_*
            idx1 = torch.randint(0, n_dates - 1, (b,), generator=gen, device=device)
            idx2 = idx1 + 1
            fields1 = {k: v[idx1] for k, v in date_tensors.items()}
            fields2 = {k: v[idx2] for k, v in date_tensors.items()}
            birth = fields2["year"] - age
            apartment = torch.randint(APARTMENT_MIN, APARTMENT_MAX + 1, (b,), generator=gen, device=device)
            hit1, score1 = _gpu_day1_masks(fields1, first, surname, age, birth, apartment, lookup_tensors)
            hit2, score2 = _gpu_day2_masks(fields2, first, surname, age, birth, lookup_tensors)
            carry = fields2["full_date_g"] == Q_WORDS
            full_hit = hit1 & hit2 & carry
            score = score1 + score2 + carry.to(torch.int16)

        hits = int(full_hit.sum().item())
        max_score = int(score.max().item())
        uniq, cnt = torch.unique(score, return_counts=True)
        hist_update = {int(k): int(v) for k, v in zip(uniq.cpu().tolist(), cnt.cpu().tolist())}

        state["full_hits"] = int(state["full_hits"]) + hits
        state["best_score"] = max(int(state["best_score"]), max_score)
        hist = {int(k): int(v) for k, v in state.get("score_histogram", {}).items()}
        for k, v in hist_update.items():
            hist[k] = hist.get(k, 0) + v
        state["score_histogram"] = {str(k): v for k, v in sorted(hist.items())}
        state["trials_completed"] = int(state["trials_completed"]) + b
        state["next_batch"] = batch_idx + 1
        state["elapsed_seconds"] = float(state.get("elapsed_seconds", 0.0)) + (time.time() - t0)
        t0 = time.time()

        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
            interval = monte_carlo_interval(int(state["full_hits"]), int(state["trials_completed"]))
            state["monte_carlo_interval"] = interval
            checkpoint_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"batch {batch_idx + 1:,}/{total_batches:,} | "
                f"trials {state['trials_completed']:,} | hits {state['full_hits']:,} | "
                f"best score {state['best_score']} | p_addone {interval['p_addone']:.3e} | "
                f"z_addone {interval['z_addone_one_sided']:.3f}"
            )

    state["finished_unix"] = time.time()
    state["monte_carlo_interval"] = monte_carlo_interval(int(state["full_hits"]), int(state["trials_completed"]))
    checkpoint_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


# -----------------------------------------------------------------------------
# Protocol fingerprint and main
# -----------------------------------------------------------------------------


def protocol_manifest(ledger_path: Path) -> Dict[str, object]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "observed_day1": OBSERVED_DAY1.isoformat(),
        "observed_day2": OBSERVED_DAY2.isoformat(),
        "day2_full_gate": list(DAY2_FAMILIES),
        "day1_full_gate": list(DAY1_FAMILIES),
        "age_range": [AGE_MIN, AGE_MAX],
        "apartment_range": [APARTMENT_MIN, APARTMENT_MAX],
        "numeric_first_range": [FIRST_NUMERIC_MIN, FIRST_NUMERIC_MAX],
        "numeric_surname_range": [SURNAME_NUMERIC_MIN, SURNAME_NUMERIC_MAX],
        "uniform_letter_support": HEBREW_LETTER_VALUE_SUPPORT.tolist(),
        "question_text": QUESTION_TEXT,
        "model_statement_text": MODEL_STATEMENT_TEXT,
        "rtm_fixed_anchor": RTM_FIXED_ANCHOR,
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def verify_observed_joint_gates() -> Dict[str, object]:
    d1 = day1_families(OBSERVED_DAY1, REAL_OBSERVER)
    d2 = day2_families(OBSERVED_DAY2, REAL_OBSERVER)
    carry = chronological_carryover(OBSERVED_DAY1, OBSERVED_DAY2)
    seq = sequential_full_hit(OBSERVED_DAY1, OBSERVED_DAY2, REAL_OBSERVER)
    result = {
        "day1_families": d1,
        "day1_9_of_9": all(d1.values()),
        "day2_families": d2,
        "day2_9_of_9": all(d2.values()),
        "chronological_carryover": carry,
        "sequential_full_joint_gate": seq,
    }
    print("\n" + "=" * 88)
    print("OBSERVED FULL JOINT GATES")
    print("=" * 88)
    print("Day 1 families:", d1, "->", "PASS" if result["day1_9_of_9"] else "FAIL")
    print("Day 2 families:", d2, "->", "PASS" if result["day2_9_of_9"] else "FAIL")
    print("Chronological carryover:", "PASS" if carry else "FAIL")
    print("SEQUENTIAL FULL JOINT GATE:", "PASS" if seq else "FAIL")
    return result


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTM 14→15 July 2026 full-joint validation engine")
    p.add_argument(
        "--mode",
        choices=["audit", "exhaustive", "exact-nulls", "all-exact", "gpu"],
        default="all-exact",
    )
    p.add_argument("--ledger", default=None, help="Path to RTM_15_JULY_2026_EXACT_LEDGER.json")
    p.add_argument("--output", default="RTM_14_15_FULL_JOINT_RESULTS.json")
    p.add_argument(
        "--gpu-model",
        choices=[
            "fixed_letters", "fixed_numeric",
            "joint_letters", "joint_numeric",
            "sequential_letters", "sequential_numeric",
        ],
        default="fixed_letters",
    )
    p.add_argument("--trials", type=int, default=100_000_000)
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--checkpoint", default="RTM_GPU_CHECKPOINT.json")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ledger_path = find_ledger(args.ledger)
    out: Dict[str, object] = {
        "manifest": protocol_manifest(ledger_path),
        "mode": args.mode,
        "statistical_scope": STATISTICAL_SCOPE,
    }

    if args.mode in {"audit", "all-exact"}:
        out["deterministic_audit"] = run_deterministic_audit(ledger_path)
        out["observed_joint_gates"] = verify_observed_joint_gates()

    if args.mode in {"exhaustive", "all-exact"}:
        out["exhaustive_scan"] = exhaustive_scan()

    if args.mode in {"exact-nulls", "all-exact"}:
        out["exact_nulls"] = run_all_exact_nulls()

    if args.mode == "gpu":
        out["deterministic_audit"] = run_deterministic_audit(ledger_path)
        out["observed_joint_gates"] = verify_observed_joint_gates()
        if not out["deterministic_audit"]["pass"] or not out["observed_joint_gates"]["sequential_full_joint_gate"]:
            raise RuntimeError("Refusing GPU run because deterministic self-test failed.")
        out["gpu_monte_carlo"] = run_gpu_monte_carlo(
            gpu_model=args.gpu_model,
            trials=args.trials,
            batch_size=args.batch_size,
            seed=args.seed,
            checkpoint_path=Path(args.checkpoint),
            resume=args.resume,
        )

    write_json(Path(args.output), out)
    print(f"\nResults written to: {Path(args.output).resolve()}")
    print(f"Protocol manifest SHA-256: {out['manifest']['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
