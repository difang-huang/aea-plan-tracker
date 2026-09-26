"""Download the AEA RCT Registry export, normalise it, and record what changed.

Design notes
------------
* The registry publishes a CSV export of the *whole* registry from its
  advanced-search page. One pull a day is trivial load for them and is the
  only network call this module makes.
* Plans are stored sharded by registration year (`data/plans/plans_YYYY.csv`).
  Only the current year's shard changes on a normal day, so git diffs stay
  small even though the corpus is ~13k plans.
* Every add / change is appended to `data/events.jsonl`. That file is the
  project's memory: it is what makes "what happened this week" and
  "how long from registration to paper" answerable.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config
from .normalize import normalize_row
from .util import http_get, today, write_jsonl

log = logging.getLogger("tracker.collect")

# Fields whose change is worth an event (ignores cosmetic churn).
WATCHED_FIELDS = [
    "status", "title", "abstract", "intervention", "primary_outcomes",
    "primary_outcome_explanation", "experimental_design_details",
    "randomization_method", "randomization_unit", "treatment_arms",
    "n_clusters_planned", "n_obs_planned", "n_arms", "mde_text",
    "end_date", "intervention_end", "data_collection_complete",
    "analysis_plan_docs", "relevant_papers", "post_trial_documents",
    "public_data", "public_data_url", "external_links",
]

# Columns persisted to the sharded CSVs. Lists are stored JSON-encoded.
LIST_FIELDS = {"keywords", "countries", "jel", "sponsors", "partners",
               "external_links", "pis", "pi_keys"}
DROP_FIELDS = {"text_blob", "extra"}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_registry_csv(dest: Path | None = None) -> Path:
    """Fetch the whole registry to `build/registry_raw.csv`.

    `/site/csv` is the bulk export behind the advanced search page's
    "Download as CSV" button — the entire registry in one file. The server
    builds it on demand and can take a few minutes, so it gets its own long
    timeout.

    `/trials/search.csv` is *not* a bulk endpoint: it returns one page of 20
    results and ignores `per_page`. It is only used as a fallback, walked page
    by page, if the bulk export is unavailable.
    """
    dest = dest or (config.BUILD / "registry_raw.csv")
    log.info("Downloading bulk registry export from %s", config.REGISTRY_CSV_URL)
    try:
        r = http_get(config.REGISTRY_CSV_URL, accept="text/csv", stream=True,
                     timeout=config.BULK_TIMEOUT)
        total = 0
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
        log.info("Wrote %s (%.1f MB)", dest, total / 1e6)
        n = _count_rows(dest)
        log.info("Bulk export contains %d rows", n)
        if n >= config.MIN_EXPECTED_PLANS:
            return dest
        log.warning("Bulk export only had %d rows (expected >= %d); falling "
                    "back to paginated search", n, config.MIN_EXPECTED_PLANS)
    except Exception as exc:  # noqa: BLE001
        log.warning("Bulk export failed (%s); falling back to paginated search", exc)

    return download_paginated(dest)


def _count_rows(path: Path) -> int:
    """Count CSV records (the export has embedded newlines inside quotes)."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def download_paginated(dest: Path, max_pages: int = 2000) -> Path:
    """Walk `/trials/search.csv?page=N` until the pages stop yielding rows."""
    log.info("Paginating %s (20 rows per page)", config.REGISTRY_SEARCH_CSV_URL)
    header: list[str] | None = None
    rows: list[list[str]] = []
    seen_first: set[str] = set()

    for page in range(1, max_pages + 1):
        r = http_get(config.REGISTRY_SEARCH_CSV_URL, params={"page": page},
                     accept="text/csv", timeout=config.HTTP_TIMEOUT)
        reader = csv.reader(io.StringIO(r.text))
        try:
            page_header = next(reader)
        except StopIteration:
            break
        page_rows = [row for row in reader if any(c.strip() for c in row)]
        if header is None:
            header = page_header
        if not page_rows:
            log.info("Page %d empty; stopping", page)
            break
        # The registry repeats the last page forever rather than 404ing, so
        # stop as soon as a page's first row is one we have already taken.
        key = "|".join(page_rows[0][:3])
        if key in seen_first:
            log.info("Page %d repeats earlier content; stopping", page)
            break
        seen_first.add(key)
        rows.extend(page_rows)
        if page % 25 == 0:
            log.info("  %d pages, %d rows so far", page, len(rows))
        time.sleep(config.PAGE_SLEEP)

    if header is None or not rows:
        raise RuntimeError("Paginated fallback produced no rows")

    with dest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    log.info("Wrote %s (%d rows via pagination)", dest, len(rows))
    return dest


def read_raw_csv(path: Path) -> list[dict]:
    """Read the export defensively: it contains embedded newlines and quotes."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    log.info("Parsed %d raw rows from %s", len(rows), path.name)
    return rows


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
def _shard_path(year: int | None) -> Path:
    return config.PLANS_DIR / f"plans_{year or 'unknown'}.csv"


def _encode(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k in DROP_FIELDS:
            continue
        out[k] = json.dumps(v, ensure_ascii=False) if k in LIST_FIELDS else v
    return out


def _decode(rec: dict) -> dict:
    out = dict(rec)
    for k in LIST_FIELDS:
        v = out.get(k)
        if isinstance(v, str):
            try:
                out[k] = json.loads(v) if v else []
            except json.JSONDecodeError:
                out[k] = [v] if v else []
        elif v is None or (isinstance(v, float) and pd.isna(v)):
            out[k] = []
    for k, v in list(out.items()):
        if isinstance(v, float) and pd.isna(v):
            out[k] = None
    return out


def load_plans() -> dict[str, dict]:
    """Load every stored plan, keyed by RCT id."""
    plans: dict[str, dict] = {}
    for path in sorted(config.PLANS_DIR.glob("plans_*.csv")):
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        for rec in df.to_dict("records"):
            rec = _decode(rec)
            for numeric in ("n_clusters_planned", "n_obs_planned", "n_arms",
                            "n_clusters_actual", "n_obs_actual", "mde_value",
                            "trial_number", "registration_year", "n_pis",
                            "n_countries"):
                v = rec.get(numeric)
                if v not in (None, ""):
                    try:
                        rec[numeric] = float(v)
                    except (TypeError, ValueError):
                        rec[numeric] = None
                else:
                    rec[numeric] = None
            for boolean in ("has_analysis_plan", "has_irb", "prospective",
                            "public_data", "program_files",
                            "data_collection_complete"):
                v = rec.get(boolean)
                if isinstance(v, str):
                    rec[boolean] = {"True": True, "False": False}.get(v)
            if rec.get("rct_id"):
                plans[rec["rct_id"]] = rec
    log.info("Loaded %d stored plans", len(plans))
    return plans


def save_plans(plans: dict[str, dict]) -> None:
    """Write plans back out, sharded by registration year."""
    by_year: dict[object, list[dict]] = defaultdict(list)
    for rec in plans.values():
        year = rec.get("registration_year")
        try:
            year = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            year = None
        by_year[year].append(rec)

    written = set()
    for year, recs in by_year.items():
        recs.sort(key=lambda r: (r.get("trial_number") or 0))
        df = pd.DataFrame([_encode(r) for r in recs])
        # Stable column order keeps git diffs readable.
        df = df.reindex(sorted(df.columns), axis=1)
        path = _shard_path(year)
        df.to_csv(path, index=False)
        written.add(path)
        log.info("Wrote %s (%d plans)", path.name, len(recs))

    for stale in config.PLANS_DIR.glob("plans_*.csv"):
        if stale not in written:
            stale.unlink()


# --------------------------------------------------------------------------
# Diff
# --------------------------------------------------------------------------
def diff_plans(old: dict[str, dict], new: dict[str, dict]) -> list[dict]:
    """Compare two plan stores and return an event list."""
    stamp = today().isoformat()
    events: list[dict] = []

    for rct_id, rec in new.items():
        prev = old.get(rct_id)
        if prev is None:
            events.append({
                "date": stamp,
                "type": "plan_registered",
                "rct_id": rct_id,
                "title": rec.get("title", "")[:300],
                "registered_on": rec.get("first_registered_on", ""),
                "status": rec.get("status", ""),
            })
            continue
        if prev.get("content_hash") == rec.get("content_hash"):
            continue
        changed = [f for f in WATCHED_FIELDS
                   if _norm(prev.get(f)) != _norm(rec.get(f))]
        if not changed:
            continue
        ev = {
            "date": stamp,
            "type": "plan_updated",
            "rct_id": rct_id,
            "title": rec.get("title", "")[:300],
            "fields": changed,
        }
        if "status" in changed:
            ev["type"] = "status_changed"
            ev["from"] = prev.get("status")
            ev["to"] = rec.get("status")
        if "relevant_papers" in changed or "post_trial_documents" in changed:
            ev["papers_field_changed"] = True
        events.append(ev)

    for rct_id in set(old) - set(new):
        events.append({
            "date": stamp, "type": "plan_disappeared", "rct_id": rct_id,
            "title": old[rct_id].get("title", "")[:300],
        })
    return events


def _norm(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run(raw_csv: Path | None = None, *, offline: bool = False) -> dict:
    """Collect step. Returns a small summary dict."""
    if raw_csv is None and not offline:
        raw_csv = download_registry_csv()
    if raw_csv is None:
        raise RuntimeError("offline mode needs --raw-csv pointing at a snapshot")

    raw_rows = read_raw_csv(Path(raw_csv))
    if len(raw_rows) < 100:
        raise RuntimeError(
            f"Only {len(raw_rows)} rows in the export; that is not the full "
            "registry. Aborting rather than deleting stored plans."
        )

    new_plans: dict[str, dict] = {}
    skipped = 0
    for row in raw_rows:
        rec = normalize_row(row)
        if not rec.get("rct_id"):
            skipped += 1
            continue
        new_plans[rec["rct_id"]] = rec

    old_plans = load_plans()

    # Guard against a bad upstream day wiping the corpus.
    if old_plans and len(new_plans) < 0.9 * len(old_plans):
        raise RuntimeError(
            f"Export has {len(new_plans)} plans but we stored "
            f"{len(old_plans)}. That >10% drop looks like an upstream "
            "problem; keeping existing data."
        )

    events = diff_plans(old_plans, new_plans)
    save_plans(new_plans)
    if events:
        write_jsonl(config.EVENTS_JSONL, events, append=True)

    summary = {
        "plans_total": len(new_plans),
        "plans_new": sum(1 for e in events if e["type"] == "plan_registered"),
        "plans_updated": sum(1 for e in events
                             if e["type"] in {"plan_updated", "status_changed"}),
        "rows_skipped": skipped,
    }
    log.info("Collect summary: %s", summary)
    return summary
