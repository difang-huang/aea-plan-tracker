"""Optional LLM second opinion on novelty and feasibility.

The rule-based scores in `score.py` are the backbone: they run everywhere,
cost nothing, and are reproducible. This module adds a qualitative read on
top when an `ANTHROPIC_API_KEY` is present.

Two design choices keep it honest and cheap:

* The model is shown the plan's **nearest earlier plans** and asked to judge
  novelty *relative to them*. Asking "is this novel?" with no comparison set
  invites the model to reward fashionable topics; asking "is this different
  from these five specific earlier plans?" is a question it can answer.
* Every judgement is cached under the plan's content hash, so a plan is
  judged once and only re-judged when its text actually changes.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from . import config
from .util import cache_read, cache_write, clean_text

log = logging.getLogger("tracker.llm")

PROMPT_VERSION = "v1"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM = """You are an experienced referee for field experiments in economics, \
reviewing pre-analysis plans registered in the AEA RCT Registry.

Judge exactly two things.

NOVELTY (0-100): how much this plan adds beyond what has already been \
pre-registered. You are given the most similar EARLIER plans; judge against \
them, not against your general sense of what is fashionable. A careful \
replication in a new setting is not zero, but a plan that restates an earlier \
plan's design and outcomes in the same setting is below 25. A genuinely new \
question, mechanism, or measurement strategy is above 75.

FEASIBILITY (0-100): whether this trial, as written, will produce a credible \
answer. Weigh statistical power against the stated sample, whether the design \
and outcomes are specified tightly enough to hold the authors to, timeline \
realism, and operational risk. Ignore how interesting the question is.

Be terse and concrete. Cite the plan's own numbers. If a field is missing, say \
so — missing information is itself evidence about feasibility. Do not reward \
verbosity.

Reply with ONLY a JSON object:
{"novelty": <int 0-100>, "feasibility": <int 0-100>,
 "novelty_reason": "<= 40 words", "feasibility_reason": "<= 40 words",
 "risks": ["<= 12 words", ...up to 3],
 "closest_prior": "<rct id of the earlier plan this most resembles, or empty>"}"""


def _plan_brief(plan: dict, neighbours: list[dict]) -> str:
    def f(key: str, label: str, limit: int = 700) -> str:
        v = clean_text(plan.get(key) or "")
        return f"{label}: {v[:limit]}\n" if v else ""

    n_obs = plan.get("n_obs_planned")
    n_cl = plan.get("n_clusters_planned")
    n_arms = plan.get("n_arms")
    size = ", ".join(
        s for s in [
            f"{int(n_obs)} observations" if n_obs else "",
            f"{int(n_cl)} clusters" if n_cl else "",
            f"{int(n_arms)} arms" if n_arms else "",
        ] if s
    ) or "not reported"

    out = [
        f"PLAN {plan.get('rct_id')} (registered {plan.get('first_registered_on') or '?'}, "
        f"status {plan.get('status')})",
        f"Title: {clean_text(plan.get('title'))[:300]}",
        f"Countries: {', '.join(plan.get('countries') or []) or 'not reported'}",
        f"JEL: {', '.join(plan.get('jel') or []) or 'not reported'}",
        f"Planned sample: {size}",
        f"Minimum detectable effect as stated: {clean_text(plan.get('mde_text')) or 'not reported'}",
        f"Analysis plan attached: {'yes' if plan.get('has_analysis_plan') else 'no'}",
        f"Registered before intervention start: {plan.get('prospective')}",
        "",
    ]
    body = "".join([
        f("abstract", "Abstract"),
        f("intervention", "Intervention"),
        f("primary_outcomes", "Primary outcomes", 400),
        f("primary_outcome_explanation", "Primary outcome explanation", 600),
        f("experimental_design_details", "Design details", 900),
        f("randomization_method", "Randomization method", 400),
        f("randomization_unit", "Randomization unit", 100),
    ])
    out.append(body)

    if neighbours:
        out.append("MOST SIMILAR EARLIER PLANS:")
        for nb in neighbours[:5]:
            out.append(f"  - {nb['rct_id']} ({nb.get('registered_on','?')}, "
                       f"cosine {nb['similarity']:.2f}): {nb['title'][:180]}")
    else:
        out.append("MOST SIMILAR EARLIER PLANS: none found in the registry.")
    return "\n".join(out)


def _call(prompt: str) -> dict | None:
    body = {
        "model": config.LLM_MODEL,
        "max_tokens": 600,
        "temperature": 0,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, json=body, headers=headers, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            text = "".join(
                blk.get("text", "") for blk in r.json().get("content", [])
            )
            return _parse(text)
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                log.warning("LLM call failed: %s", exc)
                return None
            time.sleep(4 * (attempt + 1))
    return None


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for key in ("novelty", "feasibility"):
        try:
            out[key] = max(0, min(100, int(round(float(obj.get(key))))))
        except (TypeError, ValueError):
            return None
    out["novelty_reason"] = clean_text(obj.get("novelty_reason", ""))[:300]
    out["feasibility_reason"] = clean_text(obj.get("feasibility_reason", ""))[:300]
    risks = obj.get("risks") or []
    out["risks"] = [clean_text(str(x))[:120] for x in risks][:3]
    out["closest_prior"] = clean_text(str(obj.get("closest_prior", "")))[:24]
    return out


def judge(plan: dict, neighbours: list[dict]) -> dict | None:
    """Cached single-plan judgement."""
    key = (f"llm:{PROMPT_VERSION}:{config.LLM_MODEL}:"
           f"{plan.get('content_hash','')}:{plan.get('rct_id','')}")
    hit = cache_read(key)
    if hit is not None:
        return hit
    if not config.LLM_ENABLED:
        return None
    result = _call(_plan_brief(plan, neighbours))
    if result is not None:
        cache_write(key, result)
    return result


def run(plans: list[dict], scores: dict, budget: int | None = None) -> dict:
    """Add LLM judgements to `scores` in place, newest and least-covered first."""
    budget = budget if budget is not None else config.LLM_BUDGET_PER_RUN

    # Cached results are free — apply them all, then spend budget on the rest.
    pending = []
    applied = cached = 0
    for plan in plans:
        rid = plan.get("rct_id")
        sc = scores.get(rid)
        if not sc:
            continue
        key = (f"llm:{PROMPT_VERSION}:{config.LLM_MODEL}:"
               f"{plan.get('content_hash','')}:{rid}")
        hit = cache_read(key)
        if hit is not None:
            _apply(sc, hit)
            applied += 1
            cached += 1
        else:
            pending.append(plan)

    if not config.LLM_ENABLED:
        if pending:
            log.info("LLM disabled (no ANTHROPIC_API_KEY); %d plans use "
                     "rule-based scores only", len(pending))
        return {"llm_applied": applied, "llm_cached": cached, "llm_new": 0}

    # Newest registrations first: they are what a daily reader looks at.
    pending.sort(key=lambda p: (p.get("first_registered_on") or ""), reverse=True)
    new = 0
    for plan in pending[:budget]:
        res = judge(plan, scores[plan["rct_id"]].get("neighbours", []))
        if res:
            _apply(scores[plan["rct_id"]], res)
            applied += 1
            new += 1
    log.info("LLM: %d judged this run, %d served from cache", new, cached)
    return {"llm_applied": applied, "llm_cached": cached, "llm_new": new}


def _apply(score_rec: dict, res: dict) -> None:
    w = config.LLM_WEIGHT
    score_rec["llm"] = res
    score_rec["novelty_llm"] = res["novelty"]
    score_rec["feasibility_llm"] = res["feasibility"]
    score_rec["novelty_blended"] = round(
        (1 - w) * score_rec["novelty_score"] + w * res["novelty"], 1)
    score_rec["feasibility_blended"] = round(
        (1 - w) * score_rec["feasibility_score"] + w * res["feasibility"], 1)
