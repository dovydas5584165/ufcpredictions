"""Registry of supported promotions and one entry point to load their bouts.

Adding a promotion that has "<year> in <Promotion>" results pages on Wikipedia
is one more ``Promotion(...)`` entry using :func:`_wikipedia_factory`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

import pandas as pd

from .http import HttpClient
from .schema import bouts_frame
from .sources.ufc_api import DEFAULT_ORGANIZATIONS, UFCApiSource
from .sources.wikipedia import WIKIMEDIA_HOST_INTERVALS, WikipediaResultsSource

log = logging.getLogger(__name__)

# ONE Championship's own limits (115/125/135/145/155/170/185/205/225/265 lbs).
ONE_DIVISIONS_KG = {
    "atomweight": 52.2, "strawweight": 56.7, "flyweight": 61.2, "bantamweight": 65.8,
    "featherweight": 70.3, "lightweight": 77.1, "welterweight": 83.9, "middleweight": 93.0,
    "light heavyweight": 102.1, "cruiserweight": 102.1, "heavyweight": 120.2,
}
# GLORY's divisions in kg (heavyweight is open-ended above 95 kg). Most result
# rows state the limit explicitly ("Welterweight 77 kg"); this covers the rest.
GLORY_DIVISIONS_KG = {
    "featherweight": 65.0, "lightweight": 70.0, "welterweight": 77.0, "middleweight": 85.0,
    "light heavyweight": 95.0, "heavyweight": 95.0,
}


@dataclass(frozen=True)
class Promotion:
    key: str
    name: str
    sports: tuple
    source: str
    factory: Callable[..., object]
    aliases: tuple = ()


def _ufc_factory(client: HttpClient, ufc_stats: bool = True, ufc_event_range: Optional[tuple] = None,
                 ufc_organizations: Optional[Iterable[str]] = DEFAULT_ORGANIZATIONS, **_):
    first, last = ufc_event_range or (1, None)
    return UFCApiSource(client, organizations=ufc_organizations, include_stats=ufc_stats,
                        first_event_id=first, last_event_id=last)


def _wikipedia_factory(promotion: str, organization: str, title_template: str, first_year: int,
                       default_sport: str, divisions_kg: Mapping[str, float]):
    def factory(client: HttpClient, wiki_enrich: bool = True, **_):
        return WikipediaResultsSource(
            client, promotion=promotion, organization=organization, title_template=title_template,
            first_year=first_year, default_sport=default_sport, divisions_kg=divisions_kg, enrich=wiki_enrich,
        )
    return factory


PROMOTIONS = {p.key: p for p in (
    Promotion(
        key="ufc", name="UFC", sports=("mma",),
        source="official UFC API (events, fight cards, per-fight stats)",
        factory=_ufc_factory,
    ),
    Promotion(
        key="glory", name="GLORY", sports=("kickboxing",),
        source="Wikipedia '<year> in Glory' results pages + Wikidata",
        factory=_wikipedia_factory("glory", "Glory", "{year} in Glory", 2012, "kickboxing", GLORY_DIVISIONS_KG),
        aliases=("glory_kickboxing", "glorykickboxing"),
    ),
    Promotion(
        key="one", name="ONE Championship", sports=("mma", "muay_thai", "kickboxing", "grappling"),
        source="Wikipedia '<year> in ONE Championship' results pages + Wikidata",
        factory=_wikipedia_factory("one", "ONE Championship", "{year} in ONE Championship", 2011, "mma",
                                   ONE_DIVISIONS_KG),
        aliases=("onefc", "one_fc", "one-fc", "one_championship", "onechampionship"),
    ),
)}

_ALIASES = {alias: p.key for p in PROMOTIONS.values() for alias in (p.key, *p.aliases)}


def resolve_promotions(names: Iterable[str]) -> list:
    """Normalize user input ("ONEFC", "glory") to registry keys, preserving order."""
    keys = []
    for name in names:
        key = _ALIASES.get(str(name).strip().lower())
        if key is None:
            raise ValueError(f"unknown promotion {name!r}; choose from: {', '.join(PROMOTIONS)}")
        if key not in keys:
            keys.append(key)
    return keys


def default_client(**kwargs) -> HttpClient:
    intervals = {**WIKIMEDIA_HOST_INTERVALS, **kwargs.pop("host_intervals", {})}
    return HttpClient(host_intervals=intervals, **kwargs)


def load_bouts(promotions: Iterable[str] = ("ufc", "glory", "one"), client: Optional[HttpClient] = None,
               **options) -> pd.DataFrame:
    """Fetch (or read from cache) and normalize bouts for ``promotions``.

    Options: ``ufc_stats`` (bool), ``ufc_event_range`` ((first, last) IDs),
    ``ufc_organizations`` (names or None for all), ``wiki_enrich`` (bool).
    """
    client = client or default_client()
    frames = []
    for key in resolve_promotions(promotions):
        promotion = PROMOTIONS[key]
        log.info("Loading %s from the %s", promotion.name, promotion.source)
        frame = promotion.factory(client, **options).fetch_bouts()
        log.info("%s: %d bouts (%d upcoming)", promotion.name, len(frame), int((frame["status"] == "upcoming").sum()))
        frames.append(frame)
    frames = [f for f in frames if not f.empty]
    bouts = pd.concat(frames, ignore_index=True) if frames else bouts_frame([])
    return bouts.sort_values(["event_date", "promotion", "event_id", "bout_order"], kind="stable").reset_index(drop=True)
