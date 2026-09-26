"""Generate the static GitHub Pages site from the stored data.

Output shape
------------
docs/index.html          explorer + dashboard
docs/plan.html           per-plan detail (?id=AEARCTR-0000123)
docs/data/summary.json   headline counts, chart series, recent activity
docs/data/index.json     one slim row per plan (column-oriented, ~2 MB)
docs/data/detail/NN.json 64 shards of full per-plan detail, lazy-loaded
docs/data/events.json    the most recent activity events

The index is column-oriented (a `columns` list plus `rows` of arrays) rather
than a list of objects: same information, roughly a third of the bytes, and
the client rehydrates it in one pass.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .match import STAGES, plan_stage, results_overdue
from .util import parse_date, read_jsonl, today

log = logging.getLogger("tracker.site")

TEMPLATES = Path(__file__).resolve().parent / "templates"
N_SHARDS = 64

INDEX_COLUMNS = [
    "id", "title", "pi", "year", "status", "stage", "novelty", "feasibility",
    "novelty_band", "feasibility_band", "country", "jel", "n_papers",
    "paper_title", "paper_venue", "paper_year", "paper_url", "overdue",
    "prospective", "has_plan", "n_obs", "llm",
]

STAGE_LABELS = {
    "registered": "Registered",
    "in_progress": "Under way",
    "completed_no_paper": "Completed, no paper yet",
    "candidate_paper": "Candidate paper",
    "working_paper": "Working paper",
    "published": "Published",
}
# Five ordinal funnel steps (validated blue ramp; see README).
FUNNEL_STEPS = [
    ("under_way", "Registered / under way", ["registered", "in_progress"]),
    ("completed_no_paper", "Completed, no paper yet", ["completed_no_paper"]),
    ("candidate_paper", "Candidate paper", ["candidate_paper"]),
    ("working_paper", "Working paper", ["working_paper"]),
    ("published", "Published", ["published"]),
]


def _shard(rct_id: str) -> int:
    digits = "".join(ch for ch in rct_id if ch.isdigit()) or "0"
    return int(digits) % N_SHARDS


def _density(pairs: list[tuple[float, float]], bins: int = 10) -> list[dict]:
    """2-D counts over the novelty x feasibility plane.

    Percentile scores are uniform by construction, so a 1-D histogram of them
    is a flat bar chart carrying no information. The *joint* distribution does
    carry something: whether novel plans tend to be the less feasible ones.
    """
    grid: Counter = Counter()
    for x, y in pairs:
        ix = min(bins - 1, max(0, int(x / (100.0 / bins))))
        iy = min(bins - 1, max(0, int(y / (100.0 / bins))))
        grid[(ix, iy)] += 1
    step = 100 // bins
    return [{"nx": ix, "fy": iy, "n": grid.get((ix, iy), 0),
             "x0": ix * step, "y0": iy * step, "step": step}
            for iy in range(bins) for ix in range(bins)]


def _corr(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 30:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return round(num / (dx * dy), 3) if dx and dy else None


def build(plans: list[dict], scores: dict, papers_df: pd.DataFrame) -> dict:
    config.DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DATA / "detail").mkdir(parents=True, exist_ok=True)

    papers_by_plan: dict[str, list[dict]] = defaultdict(list)
    if len(papers_df):
        for rec in papers_df.to_dict("records"):
            try:
                rec["confidence"] = float(rec.get("confidence") or 0)
            except (TypeError, ValueError):
                rec["confidence"] = 0.0
            papers_by_plan[rec["rct_id"]].append(rec)
        for v in papers_by_plan.values():
            v.sort(key=lambda r: -r["confidence"])

    rows: list[list] = []
    shards: dict[int, dict] = defaultdict(dict)
    stage_counts: Counter = Counter()
    by_year: Counter = Counter()
    published_by_year: Counter = Counter()
    paper_by_year: Counter = Counter()
    plan_doc_by_year: Counter = Counter()
    prospective_by_year: Counter = Counter()
    score_pairs: list[tuple[float, float]] = []
    overdue_rows: list[dict] = []
    lag_days: list[int] = []

    for plan in plans:
        rid = plan["rct_id"]
        sc = scores.get(rid, {})
        pl = papers_by_plan.get(rid, [])
        stage = plan_stage(plan, pl)
        stage_counts[stage] += 1
        overdue = results_overdue(plan, stage)

        best = pl[0] if pl else {}
        nov = sc.get("novelty_blended", sc.get("novelty_score"))
        fea = sc.get("feasibility_blended", sc.get("feasibility_score"))
        if nov is not None and fea is not None:
            score_pairs.append((float(nov), float(fea)))

        year = plan.get("registration_year")
        try:
            year = int(year) if year not in (None, "") else None
        except (TypeError, ValueError):
            year = None
        if year:
            by_year[year] += 1
            if stage == "published":
                published_by_year[year] += 1
            if pl:
                paper_by_year[year] += 1
            if plan.get("has_analysis_plan"):
                plan_doc_by_year[year] += 1
            if plan.get("prospective") is True:
                prospective_by_year[year] += 1

        # Registration -> first paper lag.
        reg = parse_date(plan.get("first_registered_on"))
        if reg and pl:
            dates = [parse_date(p.get("publication_date")) for p in pl]
            dates = [d for d in dates if d and d >= reg]
            if dates:
                lag_days.append((min(dates) - reg).days)

        rows.append([
            rid,
            (plan.get("title") or "")[:220],
            (plan.get("pi_name") or "")[:80],
            year,
            plan.get("status"),
            stage,
            round(float(nov), 1) if nov is not None else None,
            round(float(fea), 1) if fea is not None else None,
            sc.get("novelty_band"),
            sc.get("feasibility_band"),
            (plan.get("countries") or [None])[0],
            (plan.get("jel") or [None])[0],
            len(pl),
            (best.get("title") or "")[:180] or None,
            (best.get("venue") or "")[:80] or None,
            best.get("year") or None,
            best.get("url") or None,
            overdue,
            plan.get("prospective"),
            bool(plan.get("has_analysis_plan")),
            int(plan["n_obs_planned"]) if plan.get("n_obs_planned") else None,
            1 if sc.get("llm") else 0,
        ])

        if overdue:
            overdue_rows.append({"id": rid, "title": (plan.get("title") or "")[:160],
                                 "years": overdue, "pi": plan.get("pi_name", "")})

        shards[_shard(rid)][rid] = {
            "rct_id": rid,
            "title": plan.get("title"),
            "abstract": plan.get("abstract"),
            "url": plan.get("url"),
            "doi": plan.get("doi"),
            "pi_name": plan.get("pi_name"),
            "pis": plan.get("pis"),
            "status": plan.get("status"),
            "stage": stage,
            "first_registered_on": plan.get("first_registered_on"),
            "start_date": plan.get("start_date"),
            "end_date": plan.get("end_date"),
            "intervention_start": plan.get("intervention_start"),
            "intervention_end": plan.get("intervention_end"),
            "countries": plan.get("countries"),
            "keywords": plan.get("keywords"),
            "jel": plan.get("jel"),
            "sponsors": plan.get("sponsors"),
            "partners": plan.get("partners"),
            "intervention": plan.get("intervention"),
            "primary_outcomes": plan.get("primary_outcomes"),
            "primary_outcome_explanation": plan.get("primary_outcome_explanation"),
            "experimental_design_details": plan.get("experimental_design_details"),
            "randomization_method": plan.get("randomization_method"),
            "randomization_unit": plan.get("randomization_unit"),
            "treatment_arms": plan.get("treatment_arms"),
            "n_obs_planned": plan.get("n_obs_planned"),
            "n_clusters_planned": plan.get("n_clusters_planned"),
            "n_arms": plan.get("n_arms"),
            "mde_text": plan.get("mde_text"),
            "has_analysis_plan": plan.get("has_analysis_plan"),
            "has_irb": plan.get("has_irb"),
            "prospective": plan.get("prospective"),
            "overdue_years": overdue,
            "scores": {
                "novelty": sc.get("novelty_score"),
                "feasibility": sc.get("feasibility_score"),
                "novelty_blended": sc.get("novelty_blended"),
                "feasibility_blended": sc.get("feasibility_blended"),
                "novelty_band": sc.get("novelty_band"),
                "feasibility_band": sc.get("feasibility_band"),
                "novelty_components": sc.get("novelty_components"),
                "feasibility_components": sc.get("feasibility_components"),
                "novelty_notes": sc.get("novelty_notes"),
                "feasibility_notes": sc.get("feasibility_notes"),
                "flags": sc.get("flags"),
                "methods": sc.get("methods"),
                "llm": sc.get("llm"),
            },
            "neighbours": sc.get("neighbours", []),
            "papers": [
                {k: p.get(k) for k in
                 ("title", "authors", "venue", "kind", "year", "doi", "url",
                  "cited_by", "method", "confidence", "evidence", "first_seen")}
                for p in pl
            ],
        }

    rows.sort(key=lambda r: (r[3] or 0, r[0]), reverse=True)

    (config.DOCS_DATA / "index.json").write_text(
        json.dumps({"columns": INDEX_COLUMNS, "rows": rows},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    for shard_id, payload in shards.items():
        (config.DOCS_DATA / "detail" / f"{shard_id:02d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    # ---- events -----------------------------------------------------------
    events = read_jsonl(config.EVENTS_JSONL)
    events.sort(key=lambda e: e.get("date", ""), reverse=True)
    recent = events[:400]
    (config.DOCS_DATA / "events.json").write_text(
        json.dumps(recent, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")

    # ---- summary ----------------------------------------------------------
    funnel = []
    for key, label, members in FUNNEL_STEPS:
        funnel.append({"key": key, "label": label,
                       "n": sum(stage_counts[m] for m in members)})

    n_with_paper = sum(1 for v in papers_by_plan.values() if v)
    declared = sum(
        1 for v in papers_by_plan.values()
        if any(str(p.get("method", "")).startswith("declared") for p in v)
    )
    lag_days.sort()
    median_lag = lag_days[len(lag_days) // 2] if lag_days else None

    years = sorted(by_year)
    summary = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "n_plans": len(plans),
        "n_scored": len(scores),
        "n_with_paper": n_with_paper,
        "n_declared": declared,
        "n_published": stage_counts["published"],
        "n_working_paper": stage_counts["working_paper"],
        "n_overdue": len(overdue_rows),
        "median_lag_days": median_lag,
        "median_lag_years": round(median_lag / 365.25, 1) if median_lag else None,
        "funnel": funnel,
        "stage_counts": {k: stage_counts[k] for k in STAGES},
        "stage_labels": STAGE_LABELS,
        "registrations_by_year": [{"year": y, "n": by_year[y],
                                   "published": published_by_year[y]}
                                  for y in years],
        "cohort_output": [
            {"year": y, "n": by_year[y], "with_paper": paper_by_year[y],
             "share": round(100.0 * paper_by_year[y] / by_year[y], 1)}
            for y in years if by_year[y] >= 5
        ],
        "plan_quality": [
            {"year": y, "n": by_year[y],
             "analysis_plan": round(100.0 * plan_doc_by_year[y] / by_year[y], 1),
             "prospective": round(100.0 * prospective_by_year[y] / by_year[y], 1)}
            for y in years if by_year[y] >= 5
        ],
        "score_density": _density(score_pairs),
        "score_corr": _corr(score_pairs),
        "overdue_top": sorted(overdue_rows, key=lambda r: -r["years"])[:25],
        "llm_enabled": bool(config.LLM_ENABLED),
        "n_events": len(events),
    }
    (config.DOCS_DATA / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")

    # ---- pages ------------------------------------------------------------
    css = (TEMPLATES / "style.css").read_text("utf-8")
    common = (TEMPLATES / "common.js").read_text("utf-8")
    for page in ("index.html", "plan.html"):
        html = (TEMPLATES / page).read_text("utf-8")
        html = html.replace("/*{{CSS}}*/", css).replace("/*{{COMMON_JS}}*/", common)
        html = html.replace("{{SITE_TITLE}}", config.SITE_TITLE)
        html = html.replace("{{SITE_TAGLINE}}", config.SITE_TAGLINE)
        html = html.replace("{{GENERATED_AT}}", summary["generated_at"])
        (config.DOCS / page).write_text(html, encoding="utf-8")

    (config.DOCS / ".nojekyll").write_text("", encoding="utf-8")

    # Ship the datasets next to the site so the data is one click away.
    dl = config.DOCS / "download"
    dl.mkdir(exist_ok=True)
    for src in (config.PAPERS_CSV, config.SCORES_CSV):
        if src.exists():
            shutil.copy2(src, dl / src.name)
    for src in sorted(config.PLANS_DIR.glob("plans_*.csv")):
        shutil.copy2(src, dl / src.name)

    log.info("Site built: %d plans, %d shards", len(plans), len(shards))
    return summary
