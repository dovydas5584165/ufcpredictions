"""Multi-promotion fight data for outcome prediction.

* UFC: the official UFC data API (events, fight cards, per-fight statistics).
* GLORY and ONE Championship: event results from Wikipedia's yearly results
  pages via the MediaWiki API, enriched with date of birth / height from Wikidata.

Typical use::

    from fightdata import HttpClient, load_bouts, build_features
    bouts = load_bouts(["ufc", "glory", "one"], client=HttpClient())
    table = build_features(bouts)   # leak-free pre-fight features + red-blue diffs
"""
__version__ = "0.2.0"

from .http import HttpClient  # noqa: E402
from .promotions import PROMOTIONS, load_bouts, resolve_promotions  # noqa: E402
from .features import build_features, candidate_features  # noqa: E402

__all__ = [
    "HttpClient",
    "PROMOTIONS",
    "load_bouts",
    "resolve_promotions",
    "build_features",
    "candidate_features",
    "__version__",
]
