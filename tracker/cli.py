"""Command line entry point.

    python -m tracker.cli daily            # the whole pipeline (what CI runs)
    python -m tracker.cli collect          # download + diff only
    python -m tracker.cli score            # rescore from stored plans
    python -m tracker.cli match --budget 50
    python -m tracker.cli site             # rebuild docs/ from stored data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import build_site, collect, config, llm, match, score
from .util import setup_logging

log = logging.getLogger("tracker.cli")

SCORE_COLUMNS = [
    "rct_id", "novelty_score", "feasibility_score", "novelty_raw",
    "feasibility_raw", "novelty_band", "feasibility_band",
    "novelty_llm", "feasibility_llm", "novelty_blended", "feasibility_blended",
    "max_prior_similarity", "closest_prior_plan", "methods",
    "implied_mde_sd", "pi_prior_trials", "underpowered", "retrospective",
]


def save_scores(scores: dict) -> None:
    rows = []
    for rid, s in scores.items():
        flags = s.get("flags", {})
        nb = s.get("neighbours") or []
        rows.append({
            "rct_id": rid,
            "novelty_score": s.get("novelty_score"),
            "feasibility_score": s.get("feasibility_score"),
            "novelty_raw": s.get("novelty_raw"),
            "feasibility_raw": s.get("feasibility_raw"),
            "novelty_band": s.get("novelty_band"),
            "feasibility_band": s.get("feasibility_band"),
            "novelty_llm": s.get("novelty_llm"),
            "feasibility_llm": s.get("feasibility_llm"),
            "novelty_blended": s.get("novelty_blended"),
            "feasibility_blended": s.get("feasibility_blended"),
            "max_prior_similarity": s.get("max_prior_similarity"),
            "closest_prior_plan": nb[0]["rct_id"] if nb else "",
            "methods": ";".join(s.get("methods") or []),
            "implied_mde_sd": flags.get("implied_mde_sd"),
            "pi_prior_trials": flags.get("pi_prior_trials"),
            "underpowered": bool(flags.get("underpowered")),
            "retrospective": bool(flags.get("retrospective")),
        })
    df = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values("rct_id")
    df.to_csv(config.SCORES_CSV, index=False)
    log.info("Wrote %s (%d rows)", config.SCORES_CSV.name, len(df))


def paper_counts_by_pi(plans: list[dict], papers_df: pd.DataFrame) -> dict[str, int]:
    """How many of each PI's earlier plans produced a paper (a track-record input)."""
    if not len(papers_df):
        return {}
    with_paper = set(papers_df["rct_id"].unique())
    out: dict[str, int] = {}
    for p in plans:
        if p["rct_id"] in with_paper:
            for key in p.get("pi_keys") or []:
                out[key] = out.get(key, 0) + 1
    return out


def save_corpus_stats() -> dict:
    """Persist the similarity diagnostics produced by the last scoring pass.

    These are measurements, not scores: they say what a latent/embedding
    similarity space would have changed, so the decision to adopt one can be
    made from registry-scale evidence instead of intuition.
    """
    diag = dict(score.SIMILARITY_DIAGNOSTICS)
    if not diag:
        return {}
    config.CORPUS_STATS_JSON.write_text(
        json.dumps({"similarity_probe": diag}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log.info("Wrote %s (latent probe: %s)", config.CORPUS_STATS_JSON.name,
             diag.get("status"))
    return {k: v for k, v in diag.items() if k != "examples"}


def do_score(plans: list[dict], use_llm: bool = True) -> dict:
    papers_df = match.load_papers()
    scores = score.score_all(plans, paper_counts_by_pi(plans, papers_df))
    if use_llm:
        llm.run(plans, scores)
    save_scores(scores)
    return scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tracker")
    ap.add_argument("command",
                    choices=["daily", "collect", "score", "match", "site", "stats"])
    ap.add_argument("--raw-csv", type=Path,
                    help="use a local registry CSV instead of downloading")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network (implies --raw-csv for collect)")
    ap.add_argument("--budget", type=int, default=None,
                    help="max plans to run paper matching on this run")
    ap.add_argument("--llm-budget", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N plans (for testing)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(not args.quiet)
    if args.llm_budget is not None:
        config.LLM_BUDGET_PER_RUN = args.llm_budget
    use_llm = not args.no_llm and not args.offline

    summary: dict = {}
    scores: dict | None = None

    if args.command in {"daily", "collect"}:
        summary["collect"] = collect.run(args.raw_csv, offline=args.offline)

    plans = list(collect.load_plans().values())
    if args.limit:
        plans = plans[: args.limit]
    if not plans:
        log.error("No plans stored. Run `collect` first.")
        return 1

    # Matching runs first so that the "team track record" component can see
    # which of a PI's earlier plans produced papers.
    if args.command in {"daily", "match"} and not args.offline:
        summary["match"] = match.run(plans, budget=args.budget)

    if args.command in {"daily", "score", "match"}:
        scores = do_score(plans, use_llm=use_llm)
        summary["score"] = {"scored": len(scores)}
        probe = save_corpus_stats()
        if probe:
            summary["similarity_probe"] = probe

    if args.command in {"daily", "site", "match", "score"}:
        if scores is None:
            # `site` alone: recompute deterministically; cached LLM verdicts
            # are re-applied for free, so nothing is lost.
            scores = do_score(plans, use_llm=use_llm)
        summary["site"] = build_site.build(plans, scores, match.load_papers())

    if args.command == "stats":
        papers = match.load_papers()
        summary = {
            "plans": len(plans),
            "papers_rows": len(papers),
            "plans_with_paper": papers["rct_id"].nunique() if len(papers) else 0,
            "scored": config.SCORES_CSV.exists(),
        }

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
