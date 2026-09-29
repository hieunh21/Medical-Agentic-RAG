"""Session, fetch-with-retry và slug helper dùng chung cho discover/fetch."""
from __future__ import annotations

import random
import time
from urllib.parse import urlparse

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.8",
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    """GET với retry + exponential backoff. Trả về response đã theo redirect."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.encoding = r.encoding or "utf-8"
            if r.status_code == 200:
                return r
            if 500 <= r.status_code < 600:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def polite_sleep(min_s: float = 0.5, max_s: float = 1.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1]
    return slug or "index"
