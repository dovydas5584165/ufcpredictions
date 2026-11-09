"""UFC data from the official UFC API (the JSON feed behind ufc.com's fight tracker).

Endpoints (no key required):

* ``/api/v3/event/live/{event_id}.json`` – event metadata and the full fight
  card: corners, outcome, method, round/time, weight class, title belts, and
  fighter bio data (DOB, height, reach, stance, weigh-in).
* ``/api/v3/fight/live/{fight_id}.json`` – official per-fight statistics
  (knockdowns, significant strikes, takedowns, submission attempts, reversals,
  control time) for each fighter.

Event IDs are small integers (1 … ~1350 as of late 2026) but they are not in
chronological order, and some belong to other organizations in the UFC
database (e.g. K-1 MMA), so we enumerate the ID space and filter on
``Organization``. Unused IDs return ``{"LiveEventDetail": {}}``; enumeration
stops after a run of those past the highest known event.

Note: ``Record`` and ``Age`` in the API are *current* values, not values at
fight time, so they are deliberately ignored. Age is recomputed from DOB and
records are rebuilt from fight history in :mod:`fightdata.features`.
"""
from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

from ..http import HttpClient
from ..schema import (
    IN_TO_CM,
    LB_TO_KG,
    bouts_frame,
    detect_gender,
    is_missing,
    normalize_method,
    parse_clock,
    parse_int,
    to_float,
)

log = logging.getLogger(__name__)

API_ROOT = "https://d29dxerjsp82wz.cloudfront.net/api/v3"
EVENT_URL = API_ROOT + "/event/live/{event_id}.json"
FIGHT_URL = API_ROOT + "/fight/live/{fight_id}.json"

UFC = "Ultimate Fighting Championship"
DWCS = "Dana White's Contender Series"
DEFAULT_ORGANIZATIONS = (UFC, DWCS)

# Division limits in lbs, used only when the API omits the weight range.
UFC_DIVISIONS_LBS = {
    "strawweight": 115, "flyweight": 125, "bantamweight": 135, "featherweight": 145,
    "lightweight": 155, "welterweight": 170, "middleweight": 185,
    "light heavyweight": 205, "heavyweight": 265,
}

_STAT_FIELDS = {
    "kd": "Knockdowns",
    "sig_str_landed": "SigStrikesLanded",
    "sig_str_att": "SigStrikesAttempted",
    "td_landed": "TakedownsLanded",
    "td_att": "TakedownsAttempted",
    "sub_att": "SubmissionsAttempted",
    "rev": "Reversals",
}


class UFCApiSource:
    """Fetch and normalize UFC bouts.

    Parameters
    ----------
    organizations:
        Organization names to keep (``None`` keeps everything in the API).
    include_stats:
        Also download per-fight statistics (one request per completed fight,
        ~9k on a cold cache; cached forever afterwards).
    first_event_id / last_event_id:
        Restrict the ID range (handy for quick experiments).
    mutable_ttl:
        Seconds before re-checking events that aren't final yet (upcoming
        cards change) and unused IDs past the newest event.
    """

    promotion = "ufc"

    def __init__(
        self,
        client: HttpClient,
        organizations: Optional[Iterable[str]] = DEFAULT_ORGANIZATIONS,
        include_stats: bool = True,
        first_event_id: int = 1,
        last_event_id: Optional[int] = None,
        stop_after_missing: int = 40,
        max_workers: int = 4,
        mutable_ttl: float = 6 * 3600,
    ):
        self.client = client
        self.organizations = None if organizations is None else frozenset(organizations)
        self.include_stats = include_stats
        self.first_event_id = first_event_id
        self.last_event_id = last_event_id
        self.stop_after_missing = stop_after_missing
        self.max_workers = max_workers
        self.mutable_ttl = mutable_ttl

    # ------------------------------------------------------------------ public
    def fetch_bouts(self):
        events = self.fetch_events()
        kept = [e for e in events if self._wanted(e)]
        log.info("UFC API: %d events found, %d from %s", len(events), len(kept),
                 "all organizations" if self.organizations is None else ", ".join(sorted(self.organizations)))
        rows = [row for event in kept for row in parse_event(event)]
        if self.include_stats:
            self._attach_stats(rows)
        return bouts_frame(rows)

    def fetch_events(self) -> list:
        """Event details for every used ID in the configured range."""
        events = []
        missing_run = 0
        next_id = self.first_event_id
        batch_size = max(self.max_workers * 10, 20)
        # Offline, uncached IDs look "missing"; scan further so a partial cache still works.
        stop_after = self.stop_after_missing if not self.client.offline else max(self.stop_after_missing, 1000)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for batch_number in range(1, 1_000_000):
                if self.last_event_id is not None and next_id > self.last_event_id:
                    break
                stop = next_id + batch_size - 1
                if self.last_event_id is not None:
                    stop = min(stop, self.last_event_id)
                for detail in pool.map(self._fetch_event, range(next_id, stop + 1)):
                    if detail:
                        events.append(detail)
                        missing_run = 0
                    else:
                        missing_run += 1
                next_id = stop + 1
                if self.last_event_id is None and missing_run >= stop_after:
                    break
                if batch_number % 5 == 0:
                    log.info("UFC API: scanned event IDs up to %d (%d events so far)", stop, len(events))
        return events

    # ----------------------------------------------------------------- helpers
    def _wanted(self, event: dict) -> bool:
        if self.organizations is None:
            return True
        return (event.get("Organization") or {}).get("Name") in self.organizations

    def _fetch_event(self, event_id: int) -> Optional[dict]:
        url = EVENT_URL.format(event_id=event_id)
        try:
            detail = (self.client.get_json(url) or {}).get("LiveEventDetail") or {}
            if detail.get("Status") != "Final":
                # Upcoming/live cards change and new IDs get filled in: re-check when stale.
                detail = (self.client.get_json(url, max_age=self.mutable_ttl) or {}).get("LiveEventDetail") or {}
        except LookupError:  # offline and not cached
            return None
        return detail or None

    def _fetch_fight(self, fight_id) -> Optional[dict]:
        try:
            return self.client.get_json(FIGHT_URL.format(fight_id=fight_id))
        except LookupError:
            return None
        except RuntimeError as exc:
            log.warning("UFC API: no stats for fight %s (%s)", fight_id, exc)
            return None

    def _attach_stats(self, rows: list) -> None:
        targets = [row for row in rows if row["status"] == "final"]
        log.info("UFC API: fetching per-fight statistics for %d fights (cached after the first run)", len(targets))
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for done, (row, payload) in enumerate(
                zip(targets, pool.map(self._fetch_fight, [row["_fight_id"] for row in targets])), start=1
            ):
                stats = parse_fight_stats(payload) if payload else {}
                for corner in ("r", "b"):
                    for key, value in stats.get(row[f"_{corner}_fighter_id"], {}).items():
                        row[f"{corner}_{key}"] = value
                if done % 1000 == 0:
                    log.info("UFC API: statistics for %d/%d fights", done, len(targets))


# --------------------------------------------------------------------- parsing
def parse_event(detail: dict) -> list:
    """Normalize one ``LiveEventDetail`` payload into bout rows."""
    rows = []
    organization = (detail.get("Organization") or {}).get("Name")
    event_date = _date_part(detail.get("StartTime"))
    for fight in detail.get("FightCard") or []:
        red, blue = _corners(fight.get("Fighters") or [])
        if red is None or blue is None:
            continue
        result = fight.get("Result") or {}
        weight_class = fight.get("WeightClass") or {}
        rules = fight.get("RuleSet") or {}
        class_kg = _weight_limit_kg(weight_class)
        winner = _winner(red, blue)
        status = "final" if (fight.get("Status") or detail.get("Status")) == "Final" else "upcoming"
        method_detail = result.get("Method")
        finish_round = parse_int(result.get("EndingRound"))
        row = {
            "bout_id": f"ufc:{fight.get('FightId')}",
            "promotion": "ufc",
            "organization": organization,
            "sport": "mma",
            "status": status,
            "event_id": f"ufc:{detail.get('EventId')}",
            "event_name": detail.get("Name"),
            "event_date": event_date,
            "bout_order": fight.get("FightOrder"),
            "weight_class": weight_class.get("Description"),
            "weight_class_kg": class_kg,
            "gender": detect_gender(weight_class.get("Description")),
            "is_title_bout": any((a or {}).get("Type") == "Belt" for a in fight.get("Accolades") or []),
            "scheduled_rounds": parse_int(rules.get("PossibleRounds")),
            "winner": winner,
            "method": normalize_method(method_detail) if status == "final" else None,
            "method_detail": method_detail,
            "finish_round": finish_round,
            "finish_time_sec": parse_clock(result.get("EndingTime")),
            "fight_minutes": _fight_minutes(rules.get("Description"), finish_round, result.get("EndingTime")),
            "_fight_id": fight.get("FightId"),
        }
        if row["method"] is None and winner in ("Draw", "NC") and status == "final":
            row["method"] = "DRAW" if winner == "Draw" else "NC"
        for corner, fighter in (("r", red), ("b", blue)):
            row.update(_fighter_fields(corner, fighter, class_kg))
        rows.append(row)
    return rows


def parse_fight_stats(payload: dict) -> dict:
    """``{FighterId: {kd, sig_str_landed, ...}}`` from a ``LiveFightDetail`` payload."""
    detail = payload.get("LiveFightDetail", payload) if isinstance(payload, dict) else {}
    raw = [s for s in (detail or {}).get("FightStats") or [] if s.get("FighterId") is not None]
    # Placeholder blocks (nothing attempted by either fighter) carry no information.
    if not raw or all(not to_float(s.get("TotalStrikesAttempted")) and not to_float(s.get("TakedownsAttempted"))
                      for s in raw):
        return {}
    out = {}
    for s in raw:
        record = {key: to_float(s.get(field)) for key, field in _STAT_FIELDS.items()}
        record["ctrl_sec"] = parse_clock(s.get("ControlTime"))
        out[s["FighterId"]] = record
    return out


def _corners(fighters: list):
    red = next((f for f in fighters if f.get("Corner") == "Red"), None)
    blue = next((f for f in fighters if f.get("Corner") == "Blue"), None)
    if (red is None or blue is None) and len(fighters) == 2:
        red, blue = fighters
    return red, blue


def _outcome(fighter: dict) -> str:
    return str((fighter.get("Outcome") or {}).get("Outcome") or "").strip().lower()


def _winner(red: dict, blue: dict) -> Optional[str]:
    r, b = _outcome(red), _outcome(blue)
    if r == "win":
        return "Red"
    if b == "win":
        return "Blue"
    if "draw" in (r, b):
        return "Draw"
    if any(o in ("nc", "no contest", "no-contest") or "contest" in o for o in (r, b) if o):
        return "NC"
    return None


def _fighter_fields(corner: str, fighter: dict, class_kg: float) -> dict:
    name = fighter.get("Name") or {}
    weigh_in = to_float(fighter.get("WeighIn"))
    return {
        f"{corner}_id": f"ufc:{fighter.get('FighterId')}",
        f"_{corner}_fighter_id": fighter.get("FighterId"),
        f"{corner}_name": " ".join(p for p in (name.get("FirstName"), name.get("LastName")) if p) or None,
        f"{corner}_dob": fighter.get("DOB") or None,
        f"{corner}_height_cm": _scaled(fighter.get("Height"), IN_TO_CM),
        f"{corner}_reach_cm": _scaled(fighter.get("Reach"), IN_TO_CM),
        f"{corner}_stance": fighter.get("Stance") or None,
        # Weigh-in weight is fight-time data; the listed "Weight" is the fighter's current one.
        f"{corner}_weight_kg": round(weigh_in * LB_TO_KG, 1) if weigh_in > 0 else class_kg,
    }


def _scaled(value, factor: float) -> float:
    number = to_float(value)
    return round(number * factor, 1) if number > 0 else math.nan


def _weight_limit_kg(weight_class: dict) -> float:
    catch = to_float(weight_class.get("CatchWeight"))
    if catch > 0:
        return round(catch * LB_TO_KG, 1)
    numbers = re.findall(r"\d+(?:\.\d+)?", str(weight_class.get("Weight") or ""))
    if numbers:  # "171-185" -> 185, "Over 205" -> 205, "200 and over" -> 200
        return round(float(numbers[-1]) * LB_TO_KG, 1)
    description = str(weight_class.get("Description") or "").lower()
    for division in sorted(UFC_DIVISIONS_LBS, key=len, reverse=True):
        if division in description:
            return round(UFC_DIVISIONS_LBS[division] * LB_TO_KG, 1)
    return math.nan


def _fight_minutes(rules_description, finish_round, ending_time) -> float:
    """Elapsed fight time, using the round lengths in e.g. "3 Rnd (5-5-5)" or "1 Rnd + OT (12-3)"."""
    seconds = parse_clock(ending_time)
    if finish_round is None or math.isnan(seconds):
        return math.nan
    match = re.search(r"\(([\d\s\-]+)\)", str(rules_description or ""))
    lengths = [int(x) for x in re.findall(r"\d+", match.group(1))] if match else []
    lengths = lengths or [5]
    completed = sum(lengths[min(i, len(lengths) - 1)] for i in range(max(finish_round - 1, 0)))
    return completed + seconds / 60.0


def _date_part(value) -> Optional[str]:
    if is_missing(value):
        return None
    match = re.match(r"\d{4}-\d{2}-\d{2}", str(value))
    return match.group() if match else None
