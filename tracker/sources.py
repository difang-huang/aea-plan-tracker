"""Thin clients for the bibliographic sources used to find a plan's paper."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Iterable

from . import config
from .util import clean_doi, clean_text, http_get, get_json, throttle_sleep

log = logging.getLogger("tracker.sources")


# --------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------
OA_SELECT = ",".join([
    "id", "doi", "title", "display_name", "publication_year", "publication_date",
    "type", "type_crossref", "cited_by_count", "authorships", "primary_location",
    "best_oa_location", "open_access", "abstract_inverted_index", "referenced_works_count",
])


def _oa_params(extra: dict) -> dict:
    p = {"per-page": 25, "select": OA_SELECT}
    if config.CONTACT_EMAIL:
        p["mailto"] = config.CONTACT_EMAIL
    p.update(extra)
    return p


def oa_search(filters: str, *, search: str | None = None, per_page: int = 25,
              cache_key: str | None = None, ttl_days: int = 30) -> list[dict]:
    params = _oa_params({"filter": filters, "per-page": per_page})
    if search:
        params["search"] = search
    try:
        data = get_json(f"{config.OPENALEX_BASE}/works", params=params,
                        cache_key=cache_key, ttl_days=ttl_days)
    except Exception as exc:  # noqa: BLE001
        log.warning("OpenAlex query failed (%s): %s", exc, filters)
        return []
    throttle_sleep(config.OPENALEX_SLEEP)
    return data.get("results", []) or []


def oa_by_doi(doi: str) -> dict | None:
    doi = clean_doi(doi)
    if not doi:
        return None
    params = _oa_params({})
    params.pop("per-page", None)
    try:
        return get_json(f"{config.OPENALEX_BASE}/works/doi:{doi}", params=params,
                        cache_key=f"oa:doi:{doi}", ttl_days=45)
    except Exception:  # noqa: BLE001
        return None


def oa_fulltext(phrase: str) -> list[dict]:
    """Papers whose indexed full text contains this exact-ish phrase.

    This is how a trial registry id ("AEARCTR-0001234") finds its paper: the
    paper cites the id, so the match is near-certain when it hits.
    """
    return oa_search(f"fulltext.search:{phrase}", per_page=10,
                     cache_key=f"oa:ft:{phrase}", ttl_days=21)


def oa_title_abstract(query: str, *, from_year: int | None = None,
                      per_page: int = 25) -> list[dict]:
    filt = f"title_and_abstract.search:{query}"
    if from_year:
        filt += f",from_publication_date:{from_year}-01-01"
    return oa_search(filt, per_page=per_page,
                     cache_key=f"oa:ta:{from_year}:{query}", ttl_days=21)


def oa_abstract(work: dict) -> str:
    """Rebuild plain-text abstract from OpenAlex's inverted index."""
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return clean_text(" ".join(w for _, w in positions))[:4000]


def oa_authors(work: dict) -> list[str]:
    return [clean_text((a.get("author") or {}).get("display_name", ""))
            for a in (work.get("authorships") or [])][:25]


def oa_venue(work: dict) -> tuple[str, str]:
    """Return (venue name, venue type) from the primary location."""
    loc = work.get("primary_location") or work.get("best_oa_location") or {}
    src = loc.get("source") or {}
    return clean_text(src.get("display_name", "")), clean_text(src.get("type", ""))


WORKING_PAPER_HINTS = (
    "working paper", "nber", "cepr", "iza", "discussion paper", "bread",
    "policy research working paper", "ssrn", "arxiv", "preprint", "mimeo",
    "research paper series", "cesifo", "bank of", "federal reserve",
)


def classify_work(work: dict) -> str:
    """'published' | 'working_paper' | 'other'."""
    wtype = (work.get("type") or "").lower()
    venue, vtype = oa_venue(work)
    v = venue.lower()
    if wtype in {"preprint", "posted-content"}:
        return "working_paper"
    if vtype in {"repository"}:
        return "working_paper"
    if any(h in v for h in WORKING_PAPER_HINTS):
        return "working_paper"
    if wtype in {"article", "journal-article", "review", "book-chapter", "book"}:
        return "published"
    return "other"


# --------------------------------------------------------------------------
# Crossref
# --------------------------------------------------------------------------
def crossref_by_doi(doi: str) -> dict | None:
    doi = clean_doi(doi)
    if not doi:
        return None
    params = {"mailto": config.CONTACT_EMAIL} if config.CONTACT_EMAIL else None
    try:
        data = get_json(f"{config.CROSSREF_BASE}/works/{doi}", params=params,
                        cache_key=f"cr:doi:{doi}", ttl_days=45)
    except Exception:  # noqa: BLE001
        return None
    throttle_sleep(config.CROSSREF_SLEEP)
    return (data or {}).get("message")


def crossref_search_title(title: str, rows: int = 5) -> list[dict]:
    params = {"query.bibliographic": title[:250], "rows": rows,
              "select": "DOI,title,container-title,type,issued,author,is-referenced-by-count"}
    if config.CONTACT_EMAIL:
        params["mailto"] = config.CONTACT_EMAIL
    try:
        data = get_json(f"{config.CROSSREF_BASE}/works", params=params,
                        cache_key=f"cr:t:{title[:120]}", ttl_days=30)
    except Exception:  # noqa: BLE001
        return []
    throttle_sleep(config.CROSSREF_SLEEP)
    return ((data or {}).get("message") or {}).get("items", []) or []


def crossref_meta(msg: dict) -> dict:
    title = (msg.get("title") or [""])[0]
    venue = (msg.get("container-title") or [""])[0]
    issued = ((msg.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = issued[0] if issued else None
    authors = [
        clean_text(f"{a.get('given','')} {a.get('family','')}")
        for a in (msg.get("author") or [])
    ]
    return {
        "doi": clean_doi(msg.get("DOI", "")),
        "title": clean_text(title),
        "venue": clean_text(venue),
        "type": clean_text(msg.get("type", "")),
        "year": year,
        "authors": authors,
        "cited_by": msg.get("is-referenced-by-count"),
    }


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------
ATOM = "{http://www.w3.org/2005/Atom}"


def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    params = {"search_query": query, "max_results": max_results,
              "sortBy": "relevance"}
    try:
        r = http_get(config.ARXIV_BASE, params=params, timeout=60)
    except Exception as exc:  # noqa: BLE001
        log.warning("arXiv query failed: %s", exc)
        return []
    time.sleep(config.ARXIV_SLEEP)
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        def txt(tag: str) -> str:
            el = entry.find(f"{ATOM}{tag}")
            return clean_text(el.text if el is not None else "")
        out.append({
            "arxiv_id": txt("id").rsplit("/", 1)[-1],
            "title": txt("title"),
            "abstract": txt("summary"),
            "published": txt("published")[:10],
            "authors": [clean_text(a.findtext(f"{ATOM}name", ""))
                        for a in entry.findall(f"{ATOM}author")],
            "url": txt("id"),
        })
    return out
