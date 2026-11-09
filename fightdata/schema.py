"""Normalized bout schema shared by every promotion, plus parsing helpers.

Every data source returns one row per bout with the columns in ``BOUT_COLUMNS``.
Fighter-specific columns are prefixed with ``r_`` (red corner) and ``b_``
(blue corner), matching the convention of the original UFC dataset.

Units are metric everywhere: weights in kg, heights and reaches in cm.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from datetime import date
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

SPORTS = ("mma", "kickboxing", "muay_thai", "grappling")

# Per-fighter attributes known before the bout.
FIGHTER_ATTRS = ("dob", "height_cm", "reach_cm", "stance", "weight_kg")
# Per-fighter statistics recorded *during* the bout (UFC only). These are
# never used directly as features; they feed career-to-date averages.
FIGHT_STATS = ("kd", "sig_str_landed", "sig_str_att", "td_landed", "td_att", "sub_att", "rev", "ctrl_sec")

BOUT_COLUMNS = [
    "bout_id", "promotion", "organization", "sport", "status",
    "event_id", "event_name", "event_date", "bout_order",
    "weight_class", "weight_class_kg", "gender", "is_title_bout", "scheduled_rounds",
    # winner is "Red", "Blue", "Draw", "NC", or missing (upcoming / unknown)
    "winner", "method", "method_detail", "finish_round", "finish_time_sec", "fight_minutes",
    "r_id", "r_name", "b_id", "b_name",
    *[f"{corner}_{attr}" for corner in ("r", "b") for attr in FIGHTER_ATTRS],
    *[f"{corner}_{stat}" for corner in ("r", "b") for stat in FIGHT_STATS],
]

_NUMERIC_COLUMNS = [
    "bout_order", "weight_class_kg", "scheduled_rounds", "finish_round", "finish_time_sec", "fight_minutes",
    *[f"{c}_{a}" for c in ("r", "b") for a in ("height_cm", "reach_cm", "weight_kg")],
    *[f"{c}_{s}" for c in ("r", "b") for s in FIGHT_STATS],
]

LB_TO_KG = 0.45359237
IN_TO_CM = 2.54

# Normalized method labels.
KO_TKO, SUBMISSION, DECISION, DQ, DRAW, NO_CONTEST, OTHER = "KO/TKO", "SUB", "DEC", "DQ", "DRAW", "NC", "OTHER"

_METHOD_RULES = (
    (NO_CONTEST, re.compile(r"no contest|\bnc\b|overturned", re.I)),
    (DRAW, re.compile(r"\bdraw\b", re.I)),
    (DQ, re.compile(r"\bdq\b|disqualif", re.I)),
    (DECISION, re.compile(r"technical decision", re.I)),
    (SUBMISSION, re.compile(r"submission|\bsub\b|tap.?out", re.I)),
    (KO_TKO, re.compile(r"\bt?ko\b|knock.?out|knockdowns|doctor|corner|retire|stoppage|could not continue|injury", re.I)),
    (DECISION, re.compile(r"decision|\bdec\b|points", re.I)),
)

_SPORT_RULES = (
    ("muay_thai", re.compile(r"muay\s*thai", re.I)),
    ("kickboxing", re.compile(r"kick\s*boxing", re.I)),
    ("grappling", re.compile(r"grappling", re.I)),
    ("mma", re.compile(r"\bmma\b|mixed martial arts", re.I)),
)

_EXPLICIT_WEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|lbs?|pounds?)\b", re.I)
_WOMEN = re.compile(r"\bwom[ae]n(?:['’]s)?\b|\bfemale\b", re.I)


def is_missing(value) -> bool:
    """True for None, NaN/NaT and empty strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_method(text) -> Optional[str]:
    """Map a free-text result method ("TKO (punches)", "Decision - Split", ...) to a label."""
    if is_missing(text):
        return None
    text = str(text)
    for label, pattern in _METHOD_RULES:
        if pattern.search(text):
            return label
    return OTHER


def detect_sport(*texts, default: str) -> str:
    """Return the first sport mentioned in ``texts`` (checked in priority order)."""
    for text in texts:
        if is_missing(text):
            continue
        for sport, pattern in _SPORT_RULES:
            if pattern.search(str(text)):
                return sport
    return default


def detect_gender(text) -> str:
    return "Women" if not is_missing(text) and _WOMEN.search(str(text)) else "Men"


def weight_limit_kg(text, divisions_kg: Optional[Mapping[str, float]] = None) -> float:
    """Weight-class limit in kg from text such as "Welterweight 77 kg" or "Catchweight (130 lbs)".

    Falls back to a promotion-specific division table ("flyweight" -> 61.2).
    Open-ended classes ("Heavyweight +95 kg") return their lower bound.
    """
    if is_missing(text):
        return math.nan
    text = str(text)
    match = _EXPLICIT_WEIGHT.search(text)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return round(value * LB_TO_KG, 1) if unit.startswith(("lb", "pound")) else value
    if divisions_kg:
        lowered = text.lower()
        for name in sorted(divisions_kg, key=len, reverse=True):  # "light heavyweight" before "heavyweight"
            if name in lowered:
                return float(divisions_kg[name])
    return math.nan


def parse_clock(text) -> float:
    """"2:19" -> 139.0 seconds; NaN when unparseable."""
    if is_missing(text):
        return math.nan
    match = re.search(r"(\d+):(\d{1,2})", str(text))
    if not match:
        return math.nan
    return int(match.group(1)) * 60.0 + int(match.group(2))


def parse_int(text) -> Optional[int]:
    if is_missing(text):
        return None
    if isinstance(text, (int, np.integer)):
        return int(text)
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else None


def to_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def normalize_name(name) -> str:
    """Accent-, case- and punctuation-insensitive key for matching fighter names."""
    if is_missing(name):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def slugify(text) -> str:
    return normalize_name(text).replace(" ", "-") or "unknown"


def stable_coin(*parts) -> bool:
    """Deterministic, outcome-independent coin flip used to assign corners.

    Wikipedia lists the winner first, so keeping that order would leak the
    result into the corner. The flip only depends on the bout's identity.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return bool(digest[0] & 1)


def chunks(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def bouts_frame(rows: Iterable[Mapping]) -> pd.DataFrame:
    """Build a DataFrame with exactly ``BOUT_COLUMNS`` (extra keys are dropped)."""
    df = pd.DataFrame(list(rows)).reindex(columns=BOUT_COLUMNS)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    for col in ("r_dob", "b_dob"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_title_bout"] = df["is_title_bout"].eq(True)
    return df


def today() -> date:
    return date.today()
