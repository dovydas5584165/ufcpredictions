"""GLORY, ONE Championship (and similar promotions) from Wikipedia results pages.

Wikipedia keeps one page per promotion per year ("2025 in Glory",
"2025 in ONE Championship"). Each event is a ``<h2>`` section with an infobox
(holding the date) and a results table::

    Weight Class | (winner) | def. | (loser) | Method | Round | Time | Notes

Pages are fetched through the MediaWiki API (``action=parse``), so we parse
the same HTML the site renders. Fighter links are then resolved to canonical
article titles (following redirects), and date of birth, height and sex are
looked up on Wikidata for fighters who have an article.

Things the parser has to cope with (all seen on real pages):

* the sport is in the weight class ("Flyweight Muay Thai") on recent ONE
  pages, but only in the notes ("Kickboxing") on older ones;
* older tables use another CSS class (``toccolours``) and carry footnote
  markers inside fighter cells;
* rows with ``vs.`` are draws / no contests, or not yet fought (no method);
* award and "title fights" tables also contain "def." rows; they are skipped
  because they are not inside an event section with a date.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Optional
from urllib.parse import unquote

from bs4 import BeautifulSoup, FeatureNotFound

from ..http import HttpClient
from ..schema import (
    DRAW,
    NO_CONTEST,
    bouts_frame,
    chunks,
    detect_gender,
    detect_sport,
    is_missing,
    normalize_method,
    normalize_name,
    parse_clock,
    parse_int,
    slugify,
    stable_coin,
    to_float,
    weight_limit_kg,
)

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
# Wikimedia asks for serial, unhurried requests from bots.
WIKIMEDIA_HOST_INTERVALS = {"en.wikipedia.org": 1.0, "www.wikidata.org": 1.0}

_DAY = 86400.0
_SEPARATORS = {
    "def.": "def", "def": "def", "defeated": "def",
    "vs.": "vs", "vs": "vs", "v.": "vs",
    "draw.": "draw", "draw": "draw", "drew": "draw", "drew with": "draw",
}
_MONTHS = {name: number for number, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), start=1)}
_MONTH_PATTERN = "|".join(_MONTHS)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY_YEAR = re.compile(rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:\s*[–\-]\s*\d{{1,2}})?,?\s+(\d{{4}})", re.I)
_DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})(?:\s*[–\-]\s*\d{{1,2}})?\s+({_MONTH_PATTERN})\.?,?\s+(\d{{4}})", re.I)
_CHAMPION_MARK = re.compile(r"\((?:c|ic|tc|uc)\)", re.I)
_TITLE_NOTE = re.compile(r"\bfor the\b.*\b(?:championship|title)\b", re.I)

_WIKIDATA_FEMALE = {"Q6581072", "Q1052281"}
_WIKIDATA_MALE = {"Q6581097", "Q2449503"}
_UNIT_TO_CM = {"Q174728": 1.0, "Q11573": 100.0, "Q218593": 2.54, "Q3710": 30.48}  # cm, m, inch, foot


@dataclass
class _Section:
    name: Optional[str]
    anchor: Optional[str]
    date: Optional[date]


@dataclass
class _Fighter:
    name: str
    link: Optional[str]
    champion: bool


class WikipediaResultsSource:
    """Bouts for one promotion from its "<year> in <Promotion>" pages.

    Parameters
    ----------
    title_template:
        Page title pattern, e.g. ``"{year} in Glory"``.
    default_sport:
        Sport assumed when neither the weight class nor the notes name one.
    divisions_kg:
        Division name -> weight limit (kg), for weight classes without a number.
    enrich:
        Resolve fighter links to canonical titles and fetch DOB / height / sex
        from Wikidata.
    """

    def __init__(
        self,
        client: HttpClient,
        promotion: str,
        organization: str,
        title_template: str,
        first_year: int,
        default_sport: str,
        divisions_kg: Optional[Mapping[str, float]] = None,
        last_year: Optional[int] = None,
        enrich: bool = True,
        current_ttl: float = 6 * 3600,
        recent_ttl: float = 7 * _DAY,
        reference_ttl: float = 30 * _DAY,
    ):
        self.client = client
        self.promotion = promotion
        self.organization = organization
        self.title_template = title_template
        self.first_year = first_year
        self.last_year = last_year
        self.default_sport = default_sport
        self.divisions_kg = dict(divisions_kg or {})
        self.enrich = enrich
        self.current_ttl = current_ttl
        self.recent_ttl = recent_ttl
        self.reference_ttl = reference_ttl

    def page_titles(self) -> list:
        last = self.last_year or date.today().year
        return [(year, self.title_template.format(year=year)) for year in range(self.first_year, last + 1)]

    def fetch_bouts(self):
        rows = []
        this_year = date.today().year
        for year, title in self.page_titles():
            # Old seasons are settled; the current one gains events every week.
            max_age = self.current_ttl if year >= this_year else self.recent_ttl if year == this_year - 1 else None
            html = self._page_html(title, max_age)
            if html is None:
                log.warning("Wikipedia: no page %r; skipping", title)
                continue
            page_rows = parse_results_page(html, self.promotion, self.organization, self.default_sport, self.divisions_kg)
            log.info("Wikipedia: %-30s %4d bouts", title, len(page_rows))
            rows.extend(page_rows)
        rows = _dedupe(rows)
        if self.enrich and rows:
            try:
                self._enrich(rows)
            except LookupError:
                log.warning("Wikipedia: offline and fighter metadata not cached; skipping Wikidata enrichment")
        return bouts_frame(rows)

    # ----------------------------------------------------------------- helpers
    def _page_html(self, title: str, max_age: Optional[float]) -> Optional[str]:
        params = {"action": "parse", "page": title, "prop": "text", "redirects": "1",
                  "format": "json", "formatversion": "2", "maxlag": "5"}
        try:
            data = self.client.get_json(WIKIPEDIA_API, params=params, max_age=max_age,
                                        retry_if=_mediawiki_lagged, cacheable=_mediawiki_ok)
        except LookupError:
            return None
        if "error" in data:
            if data["error"].get("code") == "missingtitle":
                return None
            raise RuntimeError(f"MediaWiki error for {title!r}: {data['error']}")
        return data["parse"]["text"]

    def _enrich(self, rows: list) -> None:
        links = sorted({row[f"_{c}_link"] for row in rows for c in "rb" if row.get(f"_{c}_link")})
        resolved = resolve_titles(self.client, links, max_age=self.reference_ttl)
        qid_of = {title: qid for title, qid in resolved.values() if qid}

        # 1) Linked mentions -> canonical article title; remember names for step 2.
        titles_by_name = defaultdict(set)
        for row in rows:
            for c in "rb":
                link = row.get(f"_{c}_link")
                if link in resolved:
                    title = resolved[link][0]
                    row[f"{c}_id"] = f"wiki:{title}"
                    titles_by_name[normalize_name(row[f"{c}_name"])].add(title)
                    titles_by_name[normalize_name(re.sub(r"\s*\([^)]*\)$", "", title))].add(title)
                else:  # red link, missing or disambiguation page: identify by name
                    row[f"{c}_id"] = name_id(self.promotion, row[f"{c}_name"])
        # 2) Unlinked mentions of a fighter who is linked elsewhere -> same identity.
        for row in rows:
            for c in "rb":
                if row.get(f"_{c}_link") in resolved:
                    continue
                titles = titles_by_name.get(normalize_name(row[f"{c}_name"]))
                if titles and len(titles) == 1:
                    row[f"{c}_id"] = f"wiki:{next(iter(titles))}"
        # 3) Wikidata facts for fighters with an article.
        people = fetch_people(self.client, sorted(set(qid_of.values())), max_age=self.reference_ttl)
        for row in rows:
            female = False
            for c in "rb":
                fighter_id = row[f"{c}_id"]
                person = people.get(qid_of.get(fighter_id[5:])) if fighter_id.startswith("wiki:") else None
                if person:
                    row[f"{c}_dob"] = person["dob"]
                    row[f"{c}_height_cm"] = person["height_cm"]
                    female = female or person["gender"] == "Women"
            if female:  # weight class didn't say "Women's"
                row["gender"] = "Women"
        log.info("Wikipedia: %d linked fighters, %d with Wikidata facts", len(qid_of), len(people))


# --------------------------------------------------------------------- parsing
def parse_results_page(html: str, promotion: str, organization: str, default_sport: str,
                       divisions_kg: Optional[Mapping[str, float]] = None) -> list:
    """All bouts on one rendered results page (pure function: no network)."""
    soup = _soup(html)
    for tag in soup.find_all(["sup", "style", "script"]):
        tag.decompose()  # footnote markers like "[1]" would end up inside fighter names
    for flag in soup.select("span.flagicon"):
        flag.decompose()  # country flags sit next to fighter links
    listed = _listed_event_dates(soup)
    rows = []
    section = None
    next_order = defaultdict(int)
    for element in soup.find_all(["h2", "table"]):
        if element.name == "h2":
            section = _Section(_text(element), element.get("id"), None)
            continue
        if element.find_parent("table") is not None:
            continue
        if "infobox" in (element.get("class") or []):
            title, when = _infobox_event(element)
            if when is not None:
                section = _Section(title or (section.name if section else None),
                                   section.anchor if section else None, when)
            continue
        if not _is_results_table(element):
            continue
        event_name, event_date = _event_for_table(element, section, listed)
        if event_date is None or not event_name:
            continue  # awards / title-history tables, or an event we can't date
        key = (event_date, event_name)
        table_rows = _parse_results_table(element, event_name, event_date, promotion, organization,
                                          default_sport, divisions_kg, next_order[key])
        next_order[key] += len(table_rows)
        rows.extend(table_rows)
    return rows


def parse_event_date(text) -> Optional[date]:
    """Dates as written in infoboxes: "February 22, 2025", "8 March 2014", "(2025-01-10)"."""
    if is_missing(text):
        return None
    text = str(text)
    candidates = []
    match = _ISO_DATE.search(text)
    if match:
        candidates.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    match = _MONTH_DAY_YEAR.search(text)
    if match:
        candidates.append((int(match.group(3)), _MONTHS[match.group(1).lower()], int(match.group(2))))
    match = _DAY_MONTH_YEAR.search(text)
    if match:
        candidates.append((int(match.group(3)), _MONTHS[match.group(2).lower()], int(match.group(1))))
    for year, month, day in candidates:
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return None


def name_id(promotion: str, name) -> str:
    """Identity for fighters without an article (scoped to the promotion)."""
    return f"name:{promotion}:{normalize_name(name)}"


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def _text(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _cells(row) -> list:
    return row.find_all(["td", "th"], recursive=False)


def _event_key(text) -> str:
    if is_missing(text):
        return ""
    return normalize_name(re.sub(r"\([^)]*\)", " ", str(text).replace("_", " ")))


def _infobox_event(table):
    caption = table.find("caption") or table.find("th")
    title = _text(caption) or None
    for row in table.find_all("tr"):
        header, value = row.find("th", recursive=False), row.find("td", recursive=False)
        if header is not None and value is not None and _text(header).lower() == "date":
            return title, parse_event_date(_text(value))
    return title, None


def _is_results_table(table) -> bool:
    for row in table.find_all("tr", limit=4):
        labels = [_text(cell).lower() for cell in _cells(row)]
        if labels and "method" in labels and "weight" in labels[0]:
            return True
    return False


def _table_title(table) -> Optional[str]:
    first = table.find("tr")
    cells = _cells(first) if first is not None else []
    return _text(cells[0]) if len(cells) == 1 else None


def _listed_event_dates(soup) -> dict:
    """Event name / section anchor -> date, from the page's "List of events" table."""
    listed = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header = [_text(cell).lower() for cell in _cells(rows[0])]
        if "date" not in header:
            continue
        event_col = next((i for i, label in enumerate(header) if "event" in label), None)
        if event_col is None:
            continue
        date_col = header.index("date")
        for row in rows[1:]:
            cells = _cells(row)
            if len(cells) <= max(event_col, date_col):
                continue
            when = parse_event_date(_text(cells[date_col]))
            if when is None:
                continue
            keys = {_event_key(_text(cells[event_col]))}
            for link in cells[event_col].find_all("a", href=True):
                if "#" in link["href"]:
                    keys.add(_event_key(unquote(link["href"].split("#", 1)[1])))
                keys.add(_event_key(link.get("title")))
            for key in keys - {""}:
                listed.setdefault(key, when)
    return listed


def _event_for_table(table, section: Optional[_Section], listed: dict):
    if section is not None and section.date is not None:
        return section.name, section.date
    candidates = [_table_title(table)]
    if section is not None:
        candidates += [section.anchor, section.name]
    for candidate in candidates:
        key = _event_key(candidate)
        if key and key in listed:
            return (section.name if section is not None and section.name else candidate), listed[key]
    return (section.name if section is not None else None), None


def _parse_fighter(cell) -> _Fighter:
    text = _text(cell)
    champion = bool(_CHAMPION_MARK.search(text))
    name = re.sub(r"\[[^\]]*\]", " ", _CHAMPION_MARK.sub(" ", text))
    name = re.sub(r"\s+", " ", name).strip(" *†‡#")
    link = None
    for anchor in cell.find_all("a", href=True):
        if anchor.find_parent(class_="flagicon") is not None:
            continue  # the country flag
        if "new" in (anchor.get("class") or []):
            break  # red link: the fighter has no article
        href = anchor["href"]
        if href.startswith("/wiki/"):
            target = unquote(href[len("/wiki/"):].split("#", 1)[0]).replace("_", " ")
            if target and ":" not in target:
                link = target
        break
    return _Fighter(name, link, champion)


def _parse_results_table(table, event_name, event_date, promotion, organization, default_sport,
                         divisions_kg, start_order: int) -> list:
    rows = []
    segment = None  # "Main card", "Prelims", ...
    carried_weight = None  # weight-class cells can span several rows
    order = start_order
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = _cells(tr)
        if not cells:
            continue
        texts = [_text(cell) for cell in cells]
        if len(cells) == 1:
            segment = texts[0]
            continue
        lowered = [t.lower() for t in texts]
        if "method" in lowered:
            continue  # header row
        sep = next((i for i, t in enumerate(lowered) if t in _SEPARATORS), None)
        if sep is None or sep < 1 or sep + 1 >= len(cells):
            continue
        if sep >= 2:
            carried_weight = texts[sep - 2]
        first, second = _parse_fighter(cells[sep - 1]), _parse_fighter(cells[sep + 1])
        if not first.name or not second.name:
            continue
        tail = texts[sep + 2:]
        method_text, round_text, time_text = (tail + ["", "", ""])[:3]
        notes = " ".join(t for t in tail[3:] if t)
        order += 1
        rows.append(_bout_row(
            first, second, _SEPARATORS[lowered[sep]], carried_weight, method_text, round_text, time_text,
            notes, segment, event_name, event_date, promotion, organization, default_sport, divisions_kg, order,
        ))
    return rows


def _bout_row(first, second, separator, weight_text, method_text, round_text, time_text, notes, segment,
              event_name, event_date, promotion, organization, default_sport, divisions_kg, order) -> dict:
    method = normalize_method(method_text)
    if method == NO_CONTEST:
        outcome = "nc"
    elif method == DRAW or separator == "draw":
        outcome = "draw"
    elif separator == "def":
        outcome = "first"
    elif not method_text:
        outcome = "upcoming"
    else:
        outcome = "unknown"

    first_id = f"wiki:{first.link}" if first.link else name_id(promotion, first.name)
    second_id = f"wiki:{second.link}" if second.link else name_id(promotion, second.name)
    # Wikipedia lists the winner first. Keeping that order would leak the result
    # into the corner, so corners come from a coin flip on the bout's identity.
    first_is_red = stable_coin(promotion, event_date.isoformat(), *sorted((first_id, second_id)))
    red, red_id, blue, blue_id = ((first, first_id, second, second_id) if first_is_red
                                  else (second, second_id, first, first_id))
    class_kg = weight_limit_kg(weight_text, divisions_kg)
    event_slug = slugify(event_name)
    row = {
        "bout_id": f"{promotion}:{event_date.isoformat()}:{event_slug}:{order}",
        "promotion": promotion,
        "organization": organization,
        "sport": detect_sport(weight_text, notes, segment, default=default_sport),
        "status": "upcoming" if outcome == "upcoming" else "final",
        "event_id": f"{promotion}:{event_date.isoformat()}:{event_slug}",
        "event_name": event_name,
        "event_date": event_date.isoformat(),
        "bout_order": order,
        "weight_class": weight_text,
        "weight_class_kg": class_kg,
        "gender": detect_gender(weight_text),
        "is_title_bout": first.champion or second.champion or bool(_TITLE_NOTE.search(notes)),
        "winner": {"first": "Red" if first_is_red else "Blue", "draw": "Draw", "nc": "NC"}.get(outcome),
        "method": method,
        "method_detail": method_text or None,
        "finish_round": parse_int(round_text),
        "finish_time_sec": parse_clock(time_text),
    }
    for corner, fighter, fighter_id in (("r", red, red_id), ("b", blue, blue_id)):
        row[f"{corner}_id"] = fighter_id
        row[f"{corner}_name"] = fighter.name
        row[f"_{corner}_link"] = fighter.link
        row[f"{corner}_weight_kg"] = class_kg
    return row


def _dedupe(rows: list) -> list:
    seen, out = set(), []
    for row in rows:
        key = (row["event_date"], tuple(sorted((row["r_id"], row["b_id"]))))
        if row["bout_id"] in seen or key in seen:
            continue
        seen.update((row["bout_id"], key))
        out.append(row)
    return out


# -------------------------------------------------------------- enrichment API
def _mediawiki_lagged(body) -> bool:
    return isinstance(body, dict) and (body.get("error") or {}).get("code") == "maxlag"


def _mediawiki_ok(body) -> bool:
    return isinstance(body, dict) and "error" not in body


def resolve_titles(client: HttpClient, titles: Iterable[str], max_age: Optional[float] = None) -> dict:
    """Link target -> (canonical title, Wikidata QID or None).

    Follows redirects ("Rodtang" -> "Rodtang Jitmuangnon"); missing pages and
    disambiguation pages are left out so they fall back to name identities.
    """
    resolved = {}
    for batch in chunks(titles, 50):
        params = {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item|disambiguation",
                  "titles": "|".join(batch), "redirects": "1", "format": "json", "formatversion": "2",
                  "maxlag": "5"}
        data = client.get_json(WIKIPEDIA_API, params=params, max_age=max_age,
                               retry_if=_mediawiki_lagged, cacheable=_mediawiki_ok)
        query = data.get("query") or {}
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        pages = {p.get("title"): p for p in query.get("pages", [])}
        for title in batch:
            final = normalized.get(title, title)
            for _ in range(5):
                if final not in redirects:
                    break
                final = redirects[final]
            page = pages.get(final)
            if not page or page.get("missing") or page.get("invalid"):
                continue
            props = page.get("pageprops") or {}
            if "disambiguation" in props:
                continue
            resolved[title] = (final, props.get("wikibase_item"))
    return resolved


def fetch_people(client: HttpClient, qids: Iterable[str], max_age: Optional[float] = None) -> dict:
    """QID -> {"dob": ISO date or None, "height_cm": float, "gender": "Men"/"Women"/None}."""
    people = {}
    for batch in chunks(sorted(set(qids)), 50):
        params = {"action": "wbgetentities", "ids": "|".join(batch), "props": "claims",
                  "format": "json", "maxlag": "5"}
        data = client.get_json(WIKIDATA_API, params=params, max_age=max_age,
                               retry_if=_mediawiki_lagged, cacheable=_mediawiki_ok)
        for qid, entity in (data.get("entities") or {}).items():
            claims = entity.get("claims") or {}
            people[qid] = {
                "dob": _wikidata_date(claims.get("P569")),
                "height_cm": _wikidata_height_cm(claims.get("P2048")),
                "gender": _wikidata_gender(claims.get("P21")),
            }
    return people


def _ranked(statements) -> list:
    order = {"preferred": 0, "normal": 1}
    usable = [s for s in statements or [] if s.get("rank") in order]
    return sorted(usable, key=lambda s: order[s["rank"]])


def _snak_value(statement):
    return ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")


def _wikidata_date(statements) -> Optional[str]:
    for statement in _ranked(statements):
        value = _snak_value(statement) or {}
        match = re.match(r"\+(\d{4})-(\d{2})-(\d{2})", str(value.get("time") or ""))
        precision = value.get("precision", 11)
        if not match or precision < 9:  # BCE dates or precision coarser than a year
            continue
        year, month, day = (int(g) for g in match.groups())
        if precision == 9 or month == 0:
            month, day = 7, 1  # year only: assume mid-year
        elif precision == 10 or day == 0:
            day = 15  # month only
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


def _wikidata_height_cm(statements) -> float:
    for statement in _ranked(statements):
        value = _snak_value(statement) or {}
        amount = to_float(str(value.get("amount", "")).lstrip("+"))
        factor = _UNIT_TO_CM.get(str(value.get("unit", "")).rsplit("/", 1)[-1])
        if factor and amount > 0 and 140 <= amount * factor <= 230:
            return round(amount * factor, 1)
    return math.nan


def _wikidata_gender(statements) -> Optional[str]:
    for statement in _ranked(statements):
        qid = (_snak_value(statement) or {}).get("id")
        if qid in _WIKIDATA_FEMALE:
            return "Women"
        if qid in _WIKIDATA_MALE:
            return "Men"
    return None
