"""Link each pre-analysis plan to the paper it became, and track that paper's status.

Evidence ladder, strongest first
-------------------------------
1. **Author-declared.** The registry's own `Relevant papers` / `Post trial
   documents` / `External Links` fields. If a DOI is in there, the link is
   as good as certain (0.97).
2. **Registry-ID citation.** Papers whose indexed full text contains
   "AEARCTR-0001234". A paper that cites the trial id *is* the trial's paper
   (0.93).
3. **Declared citation without a DOI.** Free-text reference resolved against
   Crossref/OpenAlex by title (0.85 on a strong title match).
4. **Author + content similarity.** Papers by the PI, published after
   registration, whose title/abstract is close to the plan's text. This is
   the only inferential rung, so it is capped below "confirmed" and always
   carries its similarity score as evidence.

Nothing below `CANDIDATE_MIN` is stored. Every stored link records which rung
it came from, so a user can filter to author-declared links alone if they
want a zero-inference dataset.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, sources
from .util import (
    ARXIV_RE, NBER_RE, SSRN_RE,
    clean_doi, clean_text, extract_dois, extract_urls, normalize_person,
    normalize_title, parse_date, rct_id_variants, sha1, surname, today,
)

log = logging.getLogger("tracker.match")

PAPER_COLUMNS = [
    "rct_id", "paper_key", "kind", "title", "authors", "venue", "venue_type",
    "year", "publication_date", "doi", "openalex_id", "arxiv_id", "url",
    "cited_by", "method", "confidence", "evidence", "first_seen", "last_seen",
]
STATE_COLUMNS = ["rct_id", "last_checked", "checked_hash", "n_papers",
                 "best_kind", "best_confidence"]


# ==========================================================================
# Declared references
# ==========================================================================
def declared_blob(plan: dict) -> str:
    parts = [
        plan.get("relevant_papers") or "",
        plan.get("post_trial_documents") or "",
        " ".join(plan.get("external_links") or []),
        plan.get("public_data_url") or "",
        plan.get("program_files_url") or "",
    ]
    return clean_text(" ".join(p for p in parts if p))


def declared_references(plan: dict) -> dict:
    """Pull structured identifiers out of the author-declared free text."""
    blob = declared_blob(plan)
    if not blob:
        return {"dois": [], "arxiv": [], "nber": [], "ssrn": [], "urls": [],
                "titles": [], "blob": ""}

    dois = [d for d in extract_dois(blob)
            if not d.startswith("10.1257/rct")]  # the registry's own DOI
    urls = extract_urls(blob)
    for u in urls:  # DOIs hidden inside links
        for d in extract_dois(u):
            if d not in dois and not d.startswith("10.1257/rct"):
                dois.append(d)

    arxiv = sorted({m.group(1) for m in ARXIV_RE.finditer(blob)})
    nber = sorted({m.group(1) for m in NBER_RE.finditer(blob)})
    ssrn = sorted({m.group(1) for m in SSRN_RE.finditer(blob)})

    # Candidate titles: text chunks with the citation furniture stripped off.
    titles = []
    for chunk in _split_citations(blob):
        t = _title_from_citation(chunk)
        if t and len(t) >= 12:
            titles.append(t)

    return {"dois": dois, "arxiv": arxiv, "nber": nber, "ssrn": ssrn,
            "urls": urls, "titles": titles[:5], "blob": blob}


def _split_citations(blob: str) -> list[str]:
    for sep in ("|", ";", "\n"):
        if sep in blob:
            return [c.strip() for c in blob.split(sep) if c.strip()]
    return [blob]


_QUOTED = re.compile(r'["“]([^"”]{8,300})["”]')
_YEAR_PREFIX = re.compile(r"^.{0,120}?\(\d{4}[a-z]?\)\.?\s*")
_URLS = re.compile(r"https?://\S+")


def _title_from_citation(chunk: str) -> str:
    """Grab the most title-looking span from a free-text citation.

    A quoted span wins outright — that is how economists write titles. Failing
    that, strip URLs and a leading "Author, A. (2021)." prefix and take what
    is left.
    """
    s = clean_text(chunk)
    m = _QUOTED.search(s)
    if m:
        return m.group(1).strip()
    s = _URLS.sub(" ", s)
    s = _YEAR_PREFIX.sub("", s)
    return clean_text(s)[:250]


# ==========================================================================
# Candidate construction
# ==========================================================================
def _paper_from_openalex(work: dict) -> dict:
    venue, vtype = sources.oa_venue(work)
    doi = clean_doi(work.get("doi") or "")
    oa_id = (work.get("id") or "").rsplit("/", 1)[-1]
    return {
        "kind": sources.classify_work(work),
        "title": clean_text(work.get("display_name") or work.get("title") or ""),
        "authors": sources.oa_authors(work),
        "venue": venue,
        "venue_type": vtype,
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date") or "",
        "doi": doi,
        "openalex_id": oa_id,
        "arxiv_id": "",
        "url": f"https://doi.org/{doi}" if doi else (work.get("id") or ""),
        "cited_by": work.get("cited_by_count"),
        "_abstract": sources.oa_abstract(work),
    }


def _paper_from_crossref(msg: dict) -> dict:
    meta = sources.crossref_meta(msg)
    kind = "working_paper" if meta["type"] in {"posted-content", "report"} else (
        "published" if meta["type"] in {"journal-article", "book-chapter",
                                        "proceedings-article"} else "other")
    venue_l = meta["venue"].lower()
    if any(h in venue_l for h in sources.WORKING_PAPER_HINTS):
        kind = "working_paper"
    return {
        "kind": kind,
        "title": meta["title"],
        "authors": meta["authors"],
        "venue": meta["venue"],
        "venue_type": meta["type"],
        "year": meta["year"],
        "publication_date": f"{meta['year']}-01-01" if meta["year"] else "",
        "doi": meta["doi"],
        "openalex_id": "",
        "arxiv_id": "",
        "url": f"https://doi.org/{meta['doi']}" if meta["doi"] else "",
        "cited_by": meta["cited_by"],
        "_abstract": "",
    }


def _paper_from_arxiv(entry: dict) -> dict:
    return {
        "kind": "working_paper",
        "title": entry["title"],
        "authors": entry["authors"],
        "venue": "arXiv",
        "venue_type": "repository",
        "year": int(entry["published"][:4]) if entry.get("published") else None,
        "publication_date": entry.get("published", ""),
        "doi": "",
        "openalex_id": "",
        "arxiv_id": entry["arxiv_id"],
        "url": entry.get("url", ""),
        "cited_by": None,
        "_abstract": entry.get("abstract", ""),
    }


def paper_key(paper: dict) -> str:
    for field in ("doi", "openalex_id", "arxiv_id"):
        if paper.get(field):
            return f"{field}:{paper[field]}"
    return "t:" + sha1(normalize_title(paper.get("title", "")))[:16]


# ==========================================================================
# Matching one plan
# ==========================================================================
def match_plan(plan: dict) -> list[dict]:
    """Return every paper link found for this plan, with confidences."""
    rid = plan.get("rct_id") or ""
    found: dict[str, dict] = {}

    def add(paper: dict, method: str, confidence: float, evidence: str) -> None:
        if not paper.get("title"):
            return
        key = paper_key(paper)
        prev = found.get(key)
        if prev and prev["confidence"] >= confidence:
            return
        rec = dict(paper)
        rec.update({"paper_key": key, "method": method,
                    "confidence": round(float(confidence), 3),
                    "evidence": evidence[:400]})
        found[key] = rec

    decl = declared_references(plan)

    # --- rung 1: declared DOIs ------------------------------------------
    for doi in decl["dois"][:8]:
        work = sources.oa_by_doi(doi)
        if work:
            add(_paper_from_openalex(work), "declared_doi", 0.97,
                f"DOI {doi} listed by the authors in the registry entry.")
            continue
        msg = sources.crossref_by_doi(doi)
        if msg:
            add(_paper_from_crossref(msg), "declared_doi", 0.97,
                f"DOI {doi} listed by the authors; resolved via Crossref.")

    # --- rung 1b: declared arXiv ids ------------------------------------
    for aid in decl["arxiv"][:4]:
        entries = sources.arxiv_search(f"id:{aid}", max_results=1)
        if entries:
            add(_paper_from_arxiv(entries[0]), "declared_arxiv", 0.95,
                f"arXiv:{aid} listed by the authors.")

    # --- rung 2: papers that cite the registry id ------------------------
    for variant in rct_id_variants(rid)[:2]:
        for work in sources.oa_fulltext(variant):
            add(_paper_from_openalex(work), "rct_id_citation", 0.93,
                f"Paper's full text cites the trial id {variant}.")

    # --- rung 3: declared citations without a DOI ------------------------
    if not found:
        for title in decl["titles"][:3]:
            for msg in sources.crossref_search_title(title, rows=3):
                meta = sources.crossref_meta(msg)
                sim = _title_similarity(title, meta["title"])
                if sim >= 0.75:
                    add(_paper_from_crossref(msg), "declared_citation",
                        0.70 + 0.18 * sim,
                        f"Author-listed reference matched by title "
                        f"(similarity {sim:.2f}).")

    # --- rung 4: author + content similarity -----------------------------
    if not found and _worth_inferring(plan):
        for paper, sim, why in _infer_by_author(plan):
            conf = min(0.79, config.CANDIDATE_MIN + 0.55 * sim)
            if conf >= config.CANDIDATE_MIN:
                add(paper, "author_similarity", conf, why)

    out = []
    stamp = today().isoformat()
    for rec in found.values():
        rec.pop("_abstract", None)
        rec["rct_id"] = rid
        rec["first_seen"] = stamp
        rec["last_seen"] = stamp
        out.append(rec)
    out.sort(key=lambda r: -r["confidence"])
    return out[:12]


def _worth_inferring(plan: dict) -> bool:
    """Only guess for trials old enough to plausibly have a paper."""
    reg = parse_date(plan.get("first_registered_on"))
    if not reg:
        return False
    age_years = (today() - reg).days / 365.25
    if age_years < 1.5:
        return False
    if plan.get("status") == "withdrawn":
        return False
    return True


def _infer_by_author(plan: dict) -> list[tuple[dict, float, str]]:
    """Search the PI's output, then score candidates against the plan text."""
    pis = [p for p in (plan.get("pis") or []) if p][:3]
    if not pis:
        return []
    reg = parse_date(plan.get("first_registered_on"))
    from_year = reg.year if reg else None

    title_terms = clean_text(plan.get("title", ""))[:120]
    if not title_terms:
        return []

    candidates: list[dict] = []
    seen: set[str] = set()
    for work in sources.oa_title_abstract(title_terms, from_year=from_year,
                                          per_page=25):
        oid = work.get("id") or ""
        if oid in seen:
            continue
        seen.add(oid)
        candidates.append(work)
    if not candidates:
        return []

    plan_surnames = {surname(p) for p in pis if surname(p)}
    plan_text = clean_text(
        f"{plan.get('title','')} {plan.get('abstract','')} "
        f"{plan.get('intervention','')} {plan.get('primary_outcomes','')}"
    )
    if len(plan_text) < 80:
        return []

    docs = [plan_text]
    kept = []
    for work in candidates:
        paper = _paper_from_openalex(work)
        author_hit = bool(plan_surnames & {surname(a) for a in paper["authors"]})
        if not author_hit:
            continue  # an author overlap is required; content alone is too weak
        docs.append(f"{paper['title']} {paper['_abstract']}")
        kept.append(paper)
    if not kept:
        return []

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              sublinear_tf=True, min_df=1)
        X = vec.fit_transform(docs)
        sims = cosine_similarity(X[0:1], X[1:]).ravel()
    except ValueError:
        return []

    out = []
    for paper, sim in zip(kept, sims):
        if sim < 0.18:
            continue
        shared = sorted(plan_surnames & {surname(a) for a in paper["authors"]})
        out.append((
            paper, float(sim),
            f"Shares author(s) {', '.join(s.title() for s in shared)} with the "
            f"plan and the text overlaps (similarity {sim:.2f}). Inferred, not "
            f"author-declared."
        ))
    out.sort(key=lambda t: -t[1])
    return out[:3]


def _title_similarity(a: str, b: str) -> float:
    ta, tb = set(normalize_title(a).split()), set(normalize_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ==========================================================================
# Status ladder
# ==========================================================================
STAGES = ["registered", "in_progress", "completed_no_paper",
          "candidate_paper", "working_paper", "published"]


def plan_stage(plan: dict, papers: list[dict]) -> str:
    confirmed = [p for p in papers if p["confidence"] >= config.MATCH_CONFIRMED]
    likely = [p for p in papers if p["confidence"] >= config.MATCH_CANDIDATE]
    if any(p["kind"] == "published" for p in confirmed):
        return "published"
    if any(p["kind"] == "working_paper" for p in confirmed):
        return "working_paper"
    if likely:
        return "candidate_paper"
    if plan.get("status") == "completed":
        return "completed_no_paper"
    if plan.get("status") == "in_development":
        return "registered"
    return "in_progress"


def results_overdue(plan: dict, stage: str) -> float | None:
    """Years since data collection should have ended, with still no paper.

    This is the publication-bias signal: a finished trial with no visible
    output. It is reported, never used to penalise a score.
    """
    if stage in {"published", "working_paper", "candidate_paper"}:
        return None
    end = parse_date(
        plan.get("data_collection_completion_date")
        or plan.get("intervention_end")
        or plan.get("end_date")
    )
    if not end:
        return None
    years = (today() - end).days / 365.25
    return round(years, 2) if years >= 2.0 else None


# ==========================================================================
# Persistence + orchestration
# ==========================================================================
def load_papers() -> pd.DataFrame:
    if config.PAPERS_CSV.exists():
        df = pd.read_csv(config.PAPERS_CSV, dtype=str, keep_default_na=False)
        if "confidence" in df:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        return df
    return pd.DataFrame(columns=PAPER_COLUMNS)


def load_state() -> pd.DataFrame:
    if config.MATCH_STATE_CSV.exists():
        return pd.read_csv(config.MATCH_STATE_CSV, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=STATE_COLUMNS)


def select_plans(plans: list[dict], state: pd.DataFrame,
                 budget: int) -> list[dict]:
    """Choose which plans to spend this run's API budget on.

    Priority: never checked > content changed since last check > matched but
    stale > unmatched and stale. Within a tier, trials likelier to have a
    paper (completed, older) go first.
    """
    st = {r["rct_id"]: r for r in state.to_dict("records")} if len(state) else {}
    now = today()

    def days_since(value: str) -> int:
        d = parse_date(value)
        return 10_000 if not d else (now - d).days

    tiers: list[tuple[int, float, dict]] = []
    for p in plans:
        rid = p["rct_id"]
        s = st.get(rid)
        reg = parse_date(p.get("first_registered_on"))
        age = (now - reg).days / 365.25 if reg else 0.0
        completed = 1.0 if p.get("status") == "completed" else 0.0
        interest = completed * 3 + min(age, 12) / 4.0

        if s is None:
            tiers.append((0, -interest, p))
            continue
        if s.get("checked_hash") != p.get("content_hash"):
            tiers.append((1, -interest, p))
            continue
        gap = days_since(s.get("last_checked", ""))
        n_papers = int(s.get("n_papers") or 0)
        if n_papers > 0 and gap >= config.REMATCH_AFTER_DAYS:
            tiers.append((2, -interest, p))
        elif n_papers == 0 and gap >= config.RETRY_UNMATCHED_AFTER_DAYS:
            tiers.append((3, -interest, p))

    tiers.sort(key=lambda t: (t[0], t[1]))
    chosen = [p for _, _, p in tiers[:budget]]
    log.info("Match queue: %d eligible, taking %d", len(tiers), len(chosen))
    return chosen


def run(plans: list[dict], budget: int | None = None) -> dict:
    """Matching step. Returns a summary and appends events."""
    budget = budget if budget is not None else config.MATCH_BUDGET_PER_RUN
    papers_df = load_papers()
    state_df = load_state()

    existing: dict[str, list[dict]] = {}
    for rec in papers_df.to_dict("records"):
        existing.setdefault(rec["rct_id"], []).append(rec)

    queue = select_plans(plans, state_df, budget)
    plan_by_id = {p["rct_id"]: p for p in plans}
    stamp = today().isoformat()
    events: list[dict] = []
    state = {r["rct_id"]: dict(r) for r in state_df.to_dict("records")} \
        if len(state_df) else {}

    new_links = 0
    for i, plan in enumerate(queue, 1):
        rid = plan["rct_id"]
        if i % 50 == 0:
            log.info("  matched %d/%d", i, len(queue))
        try:
            found = match_plan(plan)
        except Exception as exc:  # noqa: BLE001
            log.warning("match failed for %s: %s", rid, exc)
            continue

        before = {r["paper_key"]: r for r in existing.get(rid, [])}
        merged: dict[str, dict] = {}
        for rec in found:
            key = rec["paper_key"]
            prev = before.get(key)
            rec["first_seen"] = (prev or {}).get("first_seen") or stamp
            rec["last_seen"] = stamp
            merged[key] = rec
            if not prev:
                new_links += 1
                events.append({
                    "date": stamp, "type": "paper_found", "rct_id": rid,
                    "paper_key": key, "kind": rec["kind"],
                    "title": rec["title"][:300], "venue": rec["venue"],
                    "confidence": rec["confidence"], "method": rec["method"],
                })
            elif prev.get("kind") != rec["kind"] and rec["kind"] == "published":
                events.append({
                    "date": stamp, "type": "paper_published", "rct_id": rid,
                    "paper_key": key, "title": rec["title"][:300],
                    "venue": rec["venue"], "from": prev.get("kind"),
                })
        # Keep links we found before but did not re-find (search is noisy).
        for key, prev in before.items():
            if key not in merged:
                merged[key] = prev
        existing[rid] = list(merged.values())

        best = max((r["confidence"] for r in merged.values()), default=0.0)
        stage = plan_stage(plan, list(merged.values()))
        prev_stage = (state.get(rid) or {}).get("best_kind")
        if prev_stage and prev_stage != stage:
            events.append({"date": stamp, "type": "stage_changed",
                           "rct_id": rid, "from": prev_stage, "to": stage})
        state[rid] = {
            "rct_id": rid, "last_checked": stamp,
            "checked_hash": plan.get("content_hash", ""),
            "n_papers": str(len(merged)), "best_kind": stage,
            "best_confidence": f"{best:.3f}",
        }

    rows = [r for recs in existing.values() for r in recs]
    out = pd.DataFrame(rows, columns=PAPER_COLUMNS)
    if len(out):
        out["authors"] = out["authors"].apply(
            lambda a: a if isinstance(a, str) else "; ".join(a or []))
        out = out.sort_values(["rct_id", "confidence"], ascending=[True, False])
    out.to_csv(config.PAPERS_CSV, index=False)

    st_out = pd.DataFrame(list(state.values()), columns=STATE_COLUMNS)
    st_out.sort_values("rct_id").to_csv(config.MATCH_STATE_CSV, index=False)

    from .util import write_jsonl
    if events:
        write_jsonl(config.EVENTS_JSONL, events, append=True)

    summary = {
        "checked": len(queue),
        "new_links": new_links,
        "plans_with_papers": sum(1 for v in existing.values() if v),
        "total_links": len(rows),
    }
    log.info("Match summary: %s", summary)
    return summary
