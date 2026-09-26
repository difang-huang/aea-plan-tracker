"""Shared helpers: HTTP with retries, an on-disk cache, text and id utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import requests

from . import config

log = logging.getLogger("tracker")


class NotFound(Exception):
    """A definitive 4xx answer. Never worth retrying."""


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": config.USER_AGENT})
    return _session


def http_get(
    url: str,
    *,
    params: dict | None = None,
    timeout: int | None = None,
    accept: str | None = None,
    stream: bool = False,
) -> requests.Response:
    """GET with bounded exponential backoff. Raises on final failure."""
    headers = {"Accept": accept} if accept else {}
    last: Exception | None = None
    for attempt in range(config.HTTP_RETRIES):
        try:
            r = session().get(
                url,
                params=params,
                timeout=timeout or config.HTTP_TIMEOUT,
                headers=headers,
                stream=stream,
            )
            # 429 and 5xx are worth retrying; other 4xx never are.
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} for {url}")
            if 400 <= r.status_code < 500:
                # A 404 from a DOI lookup is an answer, not a failure.
                # Retrying it four times with backoff is how a matching run
                # burns its whole budget on nothing.
                raise NotFound(f"{r.status_code} for {url}")
            r.raise_for_status()
            return r
        except NotFound:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == config.HTTP_RETRIES - 1:
                break
            wait = config.HTTP_BACKOFF * (2**attempt)
            log.warning("GET failed (%s), retrying in %.0fs: %s", exc, wait, url)
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {config.HTTP_RETRIES} attempts: {url}") from last


def get_json(url: str, *, params: dict | None = None, cache_key: str | None = None,
             ttl_days: int | None = None) -> Any:
    """GET JSON, optionally served from / written to the on-disk cache."""
    if cache_key:
        hit = cache_read(cache_key, ttl_days=ttl_days)
        if hit is not None:
            return hit
    r = http_get(url, params=params, accept="application/json")
    data = r.json()
    if cache_key:
        cache_write(cache_key, data)
    return data


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    d = config.CACHE / h[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def cache_read(key: str, ttl_days: int | None = None) -> Any | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if ttl_days is not None:
        stamped = blob.get("_cached_at")
        if stamped:
            age = (datetime.utcnow() - datetime.fromisoformat(stamped)).days
            if age > ttl_days:
                return None
    return blob.get("value")


def cache_write(key: str, value: Any) -> None:
    p = _cache_path(key)
    p.write_text(
        json.dumps({"_cached_at": datetime.utcnow().isoformat(), "value": value}),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%m/%d/%Y",
    "%d %B %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_date(value: Any) -> date | None:
    """Parse the several date spellings the registry export uses."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "n/a", ""}:
        return None
    s = s.split("T")[0] if re.match(r"^\d{4}-\d{2}-\d{2}T", s) else s
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def iso(d: date | None) -> str:
    return d.isoformat() if d else ""


def today() -> date:
    return datetime.utcnow().date()


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    if s.lower() in {"nan", "none"}:
        return ""
    s = s.replace(" ", " ").replace("\r", " ")
    return _WS.sub(" ", s).strip()


def split_multi(value: Any, extra_seps: Iterable[str] = ()) -> list[str]:
    """Split a registry cell that packs several values into one string.

    The export is inconsistent: some fields use ';', some '|', some newlines,
    some ', '. We split on the strong separators first and only fall back to
    commas when no strong separator is present, so that names like
    "Duflo, Esther" are not shredded.
    """
    s = clean_text(value)
    if not s:
        return []
    for sep in ("|", ";", "\n"):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    for sep in extra_seps:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def first_number(value: Any) -> float | None:
    """Pull the first number out of a free-text numeric field."""
    s = clean_text(value)
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    """Aggressive normalisation used for title equality checks."""
    s = clean_text(title).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return _WS.sub(" ", s).strip()


def normalize_person(name: str) -> str:
    """'Duflo, Esther' and 'Esther Duflo' both -> 'esther duflo'."""
    s = clean_text(name)
    s = re.sub(r"\b(dr|prof|professor|phd|ph\.d\.?|mr|ms|mrs)\.?\b", " ", s, flags=re.I)
    if "," in s and s.count(",") == 1:
        last, first = [p.strip() for p in s.split(",")]
        if first and last:
            s = f"{first} {last}"
    s = re.sub(r"[^A-Za-z \-']", " ", s)
    return _WS.sub(" ", s).strip().lower()


def surname(name: str) -> str:
    parts = normalize_person(name).split()
    return parts[-1] if parts else ""


# --------------------------------------------------------------------------
# Identifier extraction
# --------------------------------------------------------------------------
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
ARXIV_RE = re.compile(r"\barxiv[:\s/]*((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))", re.I)
NBER_RE = re.compile(r"\b(?:nber\D{0,25}?|working\s+paper\s+(?:no\.?\s*)?)w?(\d{4,5})\b", re.I)
SSRN_RE = re.compile(r"ssrn(?:\.com)?[^\d]{0,30}(\d{6,8})", re.I)
RCTID_RE = re.compile(r"\bAEARCTR[-\s]?(\d{7})\b", re.I)
URL_RE = re.compile(r"https?://[^\s,;|)\]<>\"']+")


# File extensions that get swept up when a DOI is pulled out of a URL such as
# ".../10.1186/s40172-015-0022-8.pdf". Left on, they turn every such link into
# a guaranteed 404.
_DOI_SUFFIX = re.compile(
    r"(?:\.(?:pdf|html?|xml|json|full|abstract|epub|txt|supp|s\d+))+$", re.I)


def clean_doi(raw: str) -> str:
    """Strip resolver prefix, trailing punctuation and file extensions."""
    s = clean_text(raw).lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip(".,;)]>\"'")
    s = _DOI_SUFFIX.sub("", s)
    return s.rstrip(".,;)]>\"'")


def extract_dois(text: str) -> list[str]:
    out, seen = [], set()
    for m in DOI_RE.finditer(text or ""):
        d = clean_doi(m.group(0))
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def extract_urls(text: str) -> list[str]:
    out, seen = [], set()
    for m in URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;)]>\"'")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_rct_ids(text: str) -> list[str]:
    return sorted({f"AEARCTR-{m.group(1)}" for m in RCTID_RE.finditer(text or "")})


def rct_id_variants(rct_id: str) -> list[str]:
    """Every spelling of a registry id that a paper might use."""
    m = re.search(r"(\d{7})", rct_id or "")
    if not m:
        return []
    n = m.group(1)
    short = str(int(n))
    return [f"AEARCTR-{n}", f"AEARCTR{n}", f"AEARCTR-{short}"]


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = True) -> int:
    mode = "a" if append and path.exists() else "w"
    n = 0
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
