"""Data sources. Each exposes ``fetch_bouts() -> DataFrame`` in the ``fightdata.schema`` layout."""
from .ufc_api import UFCApiSource
from .wikipedia import WikipediaResultsSource

__all__ = ["UFCApiSource", "WikipediaResultsSource"]
