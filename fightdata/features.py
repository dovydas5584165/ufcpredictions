"""Leak-free pre-fight features.

Every feature of a bout is computed from each fighter's *earlier* bouts only —
strictly earlier dates, so two bouts on the same night (tournaments) never see
each other's results. Per-fighter values are stored as ``r_pre_<feature>`` /
``b_pre_<feature>`` and the model inputs are the red-minus-blue differences
``<feature>_diff`` (``reach_diff``, ``age_diff``, ``sig_str_acc_diff``, ... as in
the original model).

The in-fight statistics of the bout itself (``r_kd``, ``r_sig_str_landed``, ...)
stay in the table for reference but are never model inputs: they are only known
after the fight.

Feature groups
--------------
record   (every promotion): fights, win_rate, streak, ko_win_rate, sub_win_rate,
         finish_rate, chin, heart, days_since_last
physical (where known): age, height, reach
stats    (UFC per-fight statistics): sig_str_acc, str_def, slpm, sapm, td_acc,
         td_def, sub_attempt, rev, kd

"Chin" is durability: 1 - share of fights lost by KO/TKO. "Heart" is
resilience: the win rate in fights that went past the first round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RECORD_FEATURES = ["fights", "win_rate", "streak", "ko_win_rate", "sub_win_rate",
                   "finish_rate", "chin", "heart", "days_since_last"]
PHYSICAL_FEATURES = ["age", "height", "reach"]
STAT_FEATURES = ["sig_str_acc", "str_def", "slpm", "sapm", "td_acc", "td_def", "sub_attempt", "rev", "kd"]
ALL_FEATURES = RECORD_FEATURES + PHYSICAL_FEATURES + STAT_FEATURES

_STRIKING_SPORTS = {"kickboxing", "muay_thai"}
# In-fight stat columns used to build career averages: long-format name -> bout column suffix.
_OWN_STATS = {"kd": "kd", "sig_landed": "sig_str_landed", "sig_att": "sig_str_att",
              "td_landed": "td_landed", "td_att": "td_att", "sub_att": "sub_att", "rev": "rev"}
_OPP_STATS = {"opp_sig_landed": "sig_str_landed", "opp_sig_att": "sig_str_att",
              "opp_td_landed": "td_landed", "opp_td_att": "td_att"}


def candidate_features(sport: str) -> list:
    """Model inputs (diff columns) that make sense for ``sport``.

    Striking sports have no submissions; grappling has no strikes. Training
    later drops candidates that are mostly missing in a segment (e.g. stats for
    promotions without per-fight statistics).
    """
    if sport in _STRIKING_SPORTS:
        names = [f for f in RECORD_FEATURES if f != "sub_win_rate"] + PHYSICAL_FEATURES
    elif sport == "grappling":
        names = [f for f in RECORD_FEATURES if f not in ("ko_win_rate", "chin")] + PHYSICAL_FEATURES
    else:
        names = RECORD_FEATURES + PHYSICAL_FEATURES + STAT_FEATURES
    return [f"{name}_diff" for name in names]


def build_features(bouts: pd.DataFrame) -> pd.DataFrame:
    """Return ``bouts`` plus pre-fight features, their diffs, ``target`` and ``r_weight``/``b_weight``.

    ``target`` is 1 when red won, 0 when blue won, NaN for draws, no contests
    and upcoming bouts. Upcoming bouts still get features (from all completed
    bouts before their date), which is what prediction uses.
    """
    long = _long_format(bouts)
    pre = _pre_fight(long, _history(long))
    out = bouts
    for corner in ("r", "b"):
        side = pre[pre["corner"] == corner].drop(columns="corner").set_index("bout_id").add_prefix(f"{corner}_pre_")
        out = out.merge(side, left_on="bout_id", right_index=True, how="left")
    extra = {f"{name}_diff": out[f"r_pre_{name}"] - out[f"b_pre_{name}"] for name in ALL_FEATURES}
    extra["target"] = out["winner"].map({"Red": 1.0, "Blue": 0.0})
    # Names used by the cohort queries in predictions.py (kg).
    extra["r_weight"] = out["r_weight_kg"]
    extra["b_weight"] = out["b_weight_kg"]
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(extra).reset_index(drop=True)], axis=1)


def _long_format(bouts: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter per bout."""
    winner = bouts["winner"]
    parts = []
    for corner, opponent, mine, theirs in (("r", "b", "Red", "Blue"), ("b", "r", "Blue", "Red")):
        result = np.select(
            [winner.eq(mine).to_numpy(), winner.eq(theirs).to_numpy(),
             winner.eq("Draw").to_numpy(), winner.eq("NC").to_numpy()],
            ["W", "L", "D", "NC"], default="",
        )
        part = {
            "bout_id": bouts["bout_id"].to_numpy(),
            "fighter_id": bouts[f"{corner}_id"].to_numpy(),
            "corner": corner,
            "date": bouts["event_date"].to_numpy(),
            "result": result,
            "method": bouts["method"].to_numpy(),
            "finish_round": bouts["finish_round"].to_numpy(dtype=float),
            "minutes": bouts["fight_minutes"].to_numpy(dtype=float),
            "dob": bouts[f"{corner}_dob"].to_numpy(),
            "height_cm": bouts[f"{corner}_height_cm"].to_numpy(dtype=float),
            "reach_cm": bouts[f"{corner}_reach_cm"].to_numpy(dtype=float),
        }
        part.update({name: bouts[f"{corner}_{col}"].to_numpy(dtype=float) for name, col in _OWN_STATS.items()})
        part.update({name: bouts[f"{opponent}_{col}"].to_numpy(dtype=float) for name, col in _OPP_STATS.items()})
        parts.append(pd.DataFrame(part))
    long = pd.concat(parts, ignore_index=True)
    long["result"] = long["result"].replace("", np.nan)
    return long[long["fighter_id"].notna() & long["date"].notna()].reset_index(drop=True)


def _history(long: pd.DataFrame) -> pd.DataFrame:
    """Cumulative career totals per fighter *through* each fight date (inclusive)."""
    done = long[long["result"].notna()]
    win, loss, draw = done["result"].eq("W"), done["result"].eq("L"), done["result"].eq("D")
    decided = win | loss | draw  # no contests don't count towards the record
    went_long = decided & (done["method"].eq("DEC") | (done["finish_round"] > 1))
    has_stats = done["sig_att"].notna()
    timed = has_stats & (done["minutes"] > 0)
    per_fight = pd.DataFrame({
        "fighter_id": done["fighter_id"],
        "date": done["date"],
        "n": decided.astype(float),
        "wins": win.astype(float),
        "losses": loss.astype(float),
        "draws": draw.astype(float),
        "ko_wins": (win & done["method"].eq("KO/TKO")).astype(float),
        "sub_wins": (win & done["method"].eq("SUB")).astype(float),
        "ko_losses": (loss & done["method"].eq("KO/TKO")).astype(float),
        "long_fights": went_long.astype(float),
        "long_wins": (went_long & win).astype(float),
        "stat_fights": has_stats.astype(float),
        "minutes": done["minutes"].where(timed),
        "sig_landed_timed": done["sig_landed"].where(timed),
        "opp_sig_landed_timed": done["opp_sig_landed"].where(timed),
        **{name: done[name].where(has_stats) for name in [*_OWN_STATS, *_OPP_STATS]},
    })
    daily = per_fight.groupby(["fighter_id", "date"], sort=True).sum()  # NaN-skipping sums
    streak = _streaks(daily)
    totals = daily.groupby(level="fighter_id").cumsum().add_prefix("cum_")
    totals["streak"] = streak
    return totals.reset_index().rename(columns={"date": "last_date"})


def _streaks(daily: pd.DataFrame) -> np.ndarray:
    """Win (+) / loss (-) streak after each fight day; rows sorted by fighter, then date."""
    out = np.zeros(len(daily))
    previous, streak = None, 0.0
    fighters = daily.index.get_level_values("fighter_id")
    for i, (fighter, wins, losses, draws) in enumerate(
            zip(fighters, daily["wins"].to_numpy(), daily["losses"].to_numpy(), daily["draws"].to_numpy())):
        if fighter != previous:
            previous, streak = fighter, 0.0
        if losses > 0:  # a loss ends the night (e.g. won the semi, lost the final)
            streak = -1.0 if wins > 0 else min(streak, 0.0) - losses
        elif wins > 0:
            streak = max(streak, 0.0) + wins
        elif draws > 0:
            streak = 0.0
        out[i] = streak
    return out


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.where(denominator > 0) / denominator.where(denominator > 0)


def _pre_fight(long: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Join each bout to the fighter's totals from the latest *earlier* fight day."""
    left = long.sort_values("date", kind="stable")
    right = history.sort_values("last_date", kind="stable")
    right = right.astype({"last_date": left["date"].dtype})
    joined = pd.merge_asof(left, right, left_on="date", right_on="last_date", by="fighter_id",
                           allow_exact_matches=False, direction="backward")
    n = joined["cum_n"].fillna(0.0)
    age = (joined["date"] - pd.to_datetime(joined["dob"], errors="coerce")).dt.days / 365.25
    features = pd.DataFrame({
        "bout_id": joined["bout_id"],
        "corner": joined["corner"],
        "fights": n,
        "win_rate": _ratio(joined["cum_wins"], n),
        "streak": joined["streak"].fillna(0.0),
        "ko_win_rate": _ratio(joined["cum_ko_wins"], n),
        "sub_win_rate": _ratio(joined["cum_sub_wins"], n),
        "finish_rate": _ratio(joined["cum_ko_wins"] + joined["cum_sub_wins"], joined["cum_wins"]),
        "chin": 1.0 - _ratio(joined["cum_ko_losses"], n),
        "heart": _ratio(joined["cum_long_wins"], joined["cum_long_fights"]),
        "days_since_last": (joined["date"] - joined["last_date"]).dt.days.astype(float),
        "age": age.where((age >= 14) & (age <= 65)),  # guard against bad DOBs
        "height": joined["height_cm"],
        "reach": joined["reach_cm"],
        "sig_str_acc": _ratio(joined["cum_sig_landed"], joined["cum_sig_att"]),
        "str_def": 1.0 - _ratio(joined["cum_opp_sig_landed"], joined["cum_opp_sig_att"]),
        "slpm": _ratio(joined["cum_sig_landed_timed"], joined["cum_minutes"]),
        "sapm": _ratio(joined["cum_opp_sig_landed_timed"], joined["cum_minutes"]),
        "td_acc": _ratio(joined["cum_td_landed"], joined["cum_td_att"]),
        "td_def": 1.0 - _ratio(joined["cum_opp_td_landed"], joined["cum_opp_td_att"]),
        "sub_attempt": _ratio(joined["cum_sub_att"], joined["cum_stat_fights"]),
        "rev": _ratio(joined["cum_rev"], joined["cum_stat_fights"]),
        "kd": _ratio(joined["cum_kd"], joined["cum_stat_fights"]),
    })
    return features
