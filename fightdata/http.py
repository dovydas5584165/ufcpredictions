"""GET-only JSON client with an on-disk cache, polite throttling and retries.

All data sources go through :class:`HttpClient` so that

* repeated runs don't re-download history: responses are stored as gzip files
  under ``cache_dir`` and reused (callers pass ``max_age`` for data that can
  still change, such as upcoming events or the current season's page);
* we stay polite to public APIs: a per-host minimum interval between requests,
  and ``Retry-After`` is honoured on HTTP 429/503;
* transient failures (429, 5xx, connection resets, truncated JSON) are retried
  with exponential backoff.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlencode, urlsplit

import requests

from . import __version__

log = logging.getLogger(__name__)

USER_AGENT_ENV = "FIGHTDATA_USER_AGENT"
CACHE_DIR_ENV = "FIGHTDATA_CACHE_DIR"
DEFAULT_CACHE_DIR = os.path.join("data", "cache")
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 120.0
_TRANSIENT_ERRORS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    getattr(requests.exceptions, "JSONDecodeError", json.JSONDecodeError),  # truncated / non-JSON body
    json.JSONDecodeError,
)


def default_user_agent() -> str:
    """Descriptive User-Agent (Wikimedia asks API clients to identify themselves).

    Set ``FIGHTDATA_USER_AGENT`` to something like
    ``"my-fight-model/1.0 (me@example.com)"`` to include your contact details.
    """
    return os.environ.get(USER_AGENT_ENV) or (
        f"fightdata/{__version__} (fight outcome research; set {USER_AGENT_ENV} to add contact info) "
        f"python-requests/{requests.__version__}"
    )


class _RetryableResponse(Exception):
    def __init__(self, message: str, wait: Optional[float] = None):
        super().__init__(message)
        self.wait = wait


class HttpClient:
    """Cached, throttled, retrying JSON client.

    Parameters
    ----------
    cache_dir:
        Where responses are cached (default ``data/cache`` or ``$FIGHTDATA_CACHE_DIR``).
    host_intervals:
        Minimum seconds between requests per host, e.g. ``{"en.wikipedia.org": 1.0}``.
        Slots are reserved under a lock, so the limit also holds across threads.
    offline:
        Never touch the network; serve whatever is cached (stale or not) and
        raise :class:`LookupError` for anything that isn't.
    """

    def __init__(
        self,
        cache_dir: Optional[os.PathLike] = None,
        user_agent: Optional[str] = None,
        host_intervals: Optional[Mapping[str, float]] = None,
        default_interval: float = 0.0,
        max_retries: int = 5,
        timeout: float = 30.0,
        offline: bool = False,
    ):
        self.cache_dir = Path(cache_dir or os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)
        self.user_agent = user_agent or default_user_agent()
        self.host_intervals = dict(host_intervals or {})
        self.default_interval = default_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.offline = offline
        self._local = threading.local()
        self._lock = threading.Lock()
        self._next_slot: dict[str, float] = {}

    # ------------------------------------------------------------------ public
    def get_json(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        max_age: Optional[float] = None,
        retry_if: Optional[Callable[[Any], bool]] = None,
        cacheable: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        """GET ``url`` and return the decoded JSON body.

        ``max_age`` (seconds): reuse a cached copy only if it is at most this old;
        ``None`` reuses any cached copy (use it for data that never changes).
        ``retry_if(body)``: treat a 200 response as transient (e.g. MediaWiki maxlag).
        ``cacheable(body)``: return False to avoid caching error payloads.
        """
        key = cache_key(url, params)
        path = self._cache_path(url, key)
        cached = _read_cache(path)
        if cached is not None:
            fresh = max_age is None or (time.time() - cached.get("fetched_at", 0)) <= max_age
            if fresh or self.offline:
                return cached.get("body")
        elif self.offline:
            raise LookupError(f"offline mode and not cached: {key}")

        body = self._fetch(url, params, retry_if)
        if cacheable is None or cacheable(body):
            _write_cache(path, {"url": key, "fetched_at": time.time(), "body": body})
        return body

    # ----------------------------------------------------------------- helpers
    @property
    def session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.user_agent, "Accept": "application/json"})
            self._local.session = session
        return session

    def _cache_path(self, url: str, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        host = urlsplit(url).netloc.replace(":", "_") or "local"
        return self.cache_dir / host / digest[:2] / f"{digest}.json.gz"

    def _throttle(self, host: str) -> None:
        interval = self.host_intervals.get(host, self.default_interval)
        if interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, 0.0))
            self._next_slot[host] = slot + interval
        if slot > now:
            time.sleep(slot - now)

    def _fetch(self, url: str, params, retry_if) -> Any:
        host = urlsplit(url).netloc
        backoff = 1.0
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            wait = backoff
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in RETRYABLE_STATUS:
                    raise _RetryableResponse(f"HTTP {response.status_code}", _retry_after_seconds(response))
                response.raise_for_status()
                body = response.json()
                if retry_if is not None and retry_if(body):
                    raise _RetryableResponse("server asked to retry later", _retry_after_seconds(response))
                return body
            except requests.exceptions.ProxyError:
                raise  # a proxy refusing the host won't change on retry
            except (_RetryableResponse, *_TRANSIENT_ERRORS) as exc:
                if isinstance(exc, _RetryableResponse) and exc.wait is not None:
                    wait = exc.wait
                if attempt >= self.max_retries:
                    raise RuntimeError(f"GET {url} failed after {attempt + 1} attempts: {exc}") from exc
                log.warning("GET %s failed (%s); retry %d/%d in %.0fs", url, exc, attempt + 1, self.max_retries, wait)
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
        raise AssertionError("unreachable")


def cache_key(url: str, params: Optional[Mapping[str, Any]]) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(sorted((str(k), str(v)) for k, v in params.items()))}"


def _retry_after_seconds(response: requests.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return min(max(float(value), 0.0), MAX_BACKOFF_SECONDS)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return min(max((when - datetime.now(timezone.utc)).total_seconds(), 0.0), MAX_BACKOFF_SECONDS)


def _read_cache(path: Path) -> Optional[dict]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, EOFError, ValueError):
        log.warning("ignoring unreadable cache file %s", path)
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    os.replace(tmp, path)  # atomic: an interrupted run never leaves a half-written file
