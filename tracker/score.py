"""Rate every plan for novelty and feasibility.

Philosophy
----------
Both scores are *corpus-relative and explainable*. Nothing here is a black
box: each score is a weighted sum of named components, every component is
computed from stated registry fields, and every plan carries the sentences
that justify its number. A reader who disagrees with a weight can change one
line in `config.py` and re-run.

Novelty asks: has anyone already pre-registered this? It is driven by
similarity to plans registered *earlier* — a plan can only be unoriginal
relative to what came before it, never relative to what came after.

Feasibility asks: if this trial runs as written, will it produce a credible
answer? It rewards adequate power, a fully specified design, a realistic
timeline, and a team that has finished trials before.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import config
from .normalize import build_text_blob
from .util import parse_date, today

log = logging.getLogger("tracker.score")

# --------------------------------------------------------------------------
# Method / design vocabulary. Rarity is measured against the corpus itself,
# so this list only has to *name* techniques, not rank them.
# --------------------------------------------------------------------------
METHOD_PATTERNS: dict[str, str] = {
    "encouragement_design": r"encouragement design|intention[- ]to[- ]treat encourag",
    "factorial": r"\bfactorial\b|\b2\s*[x×]\s*2\b|cross[- ]?randomi[sz]ed",
    "adaptive": r"adaptive (?:design|randomi|trial)|multi[- ]armed bandit|thompson sampling",
    "stepped_wedge": r"stepped[- ]wedge",
    "crossover": r"cross[- ]?over design",
    "matched_pairs": r"matched[- ]pair|pair[- ]?wise matching|matched quadruple",
    "stratified": r"strat[ai]fied randomi|block randomi",
    "re_randomization": r"re[- ]?randomi[sz]ation|rerandomi",
    "saturation_design": r"saturation design|varying (?:the )?(?:treatment )?saturation|spillover.{0,25}design",
    "list_experiment": r"list experiment|item count technique|randomi[sz]ed response",
    "lab_in_field": r"lab[- ]in[- ]the[- ]field|artefactual field experiment",
    "structural": r"structural (?:model|estimation|parameter)",
    "mechanism_experiment": r"mechanism experiment|test(?:ing)? the mechanism",
    "ml_heterogeneity": r"causal forest|machine learning.{0,30}heterogene|generic machine learning|honest tree",
    "pre_specified_ml": r"lasso|post[- ]double selection|double machine learning",
    "admin_data": r"administrative (?:data|records)|linked admin|tax records|register data",
    "biomarker": r"biomarker|cortisol|blood sample|anthropometric|hemoglobin",
    "sensor_data": r"sensor|accelerometer|gps trac|smart meter|satellite",
    "long_run_followup": r"long[- ]?run follow[- ]?up|ten[- ]year follow|five[- ]year follow|longitudinal follow",
    "audit_study": r"audit study|correspondence study|resume audit|mystery shopper",
    "natural_field": r"natural field experiment",
    "multi_country": r"multi[- ]?country|cross[- ]?country experiment|harmoni[sz]ed across countries",
    "preregistered_replication": r"replicat(?:ion|e) of|pre[- ]?registered replication",
}
METHOD_RE = {k: re.compile(v, re.I) for k, v in METHOD_PATTERNS.items()}

# Design-quality text cues used by feasibility.
POWER_CUES = re.compile(
    r"power calculation|statistical power|minimum detectable effect|\bmde\b|"
    r"power(?:ed)? (?:to detect|at)|80%? power|0\.8 power|intra[- ]?cluster correlation|\bicc\b",
    re.I,
)
MULTIPLICITY_CUES = re.compile(
    r"multiple (?:hypothesis|comparison|testing)|family[- ]?wise|romano[- ]?wolf|"
    r"benjamini|false discovery|bonferroni|index of outcomes|pre[- ]?specified index",
    re.I,
)
ATTRITION_CUES = re.compile(
    r"attrition|lee bounds|manski bounds|tracking (?:protocol|strategy)|"
    r"intensive tracking|non[- ]?response", re.I
)


def _has(text: str, rx: re.Pattern) -> bool:
    return bool(text) and bool(rx.search(text))


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _len_score(text: str, full: int) -> float:
    """0 when empty, 1 once the field is `full` characters of real content."""
    n = len(text or "")
    if n == 0:
        return 0.0
    return _clip01(math.log1p(n) / math.log1p(full))


# ==========================================================================
# Similarity backbone
# ==========================================================================
def build_similarity(plans: list[dict], n_neighbours: int = config.N_NEIGHBOURS):
    """TF-IDF + cosine kNN over the whole corpus.

    Returns (neighbours, max_sim_prior, mean_sim_prior) where `neighbours`
    maps rct_id -> list of {rct_id, title, similarity, registered_on} for the
    closest plans registered no later than this one.
    """
    texts = [p.get("text_blob") or build_text_blob(p) for p in plans]
    ids = [p["rct_id"] for p in plans]
    reg = [p.get("first_registered_on") or "" for p in plans]
    titles = [p.get("title", "") for p in plans]

    # Vectoriser settings have to scale with the corpus: min_df=3 is right for
    # 13k plans and prunes a 10-plan test corpus down to nothing.
    n_docs = len(texts)
    min_df = 3 if n_docs >= 500 else (2 if n_docs >= 60 else 1)
    max_df = 0.55 if n_docs >= 200 else 1.0

    def _fit(min_df_, max_df_):
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df_,
            max_df=max_df_,
            sublinear_tf=True,
            max_features=200_000,
            strip_accents="unicode",
        )
        return vec.fit_transform(texts)

    try:
        X = _fit(min_df, max_df)
    except ValueError:
        log.warning("TF-IDF pruning left nothing at min_df=%s/max_df=%s; "
                    "falling back to unpruned", min_df, max_df)
        X = _fit(1, 1.0)
    log.info("TF-IDF matrix: %d docs x %d features", X.shape[0], X.shape[1])

    # Pull a wide candidate set so that, after dropping later-registered
    # plans, enough earlier neighbours remain.
    k = min(len(plans), max(n_neighbours * 6, 40))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1.0 - dist

    neighbours: dict[str, list[dict]] = {}
    max_prior = np.zeros(len(plans))
    mean_prior = np.zeros(len(plans))

    for i in range(len(plans)):
        rows = []
        for j_pos, j in enumerate(idx[i]):
            if j == i:
                continue
            # "Earlier" = registered on or before this plan; ties broken by
            # trial number so the relation stays antisymmetric.
            if reg[j] and reg[i] and reg[j] > reg[i]:
                continue
            if reg[j] == reg[i] and (plans[j].get("trial_number") or 0) >= (
                plans[i].get("trial_number") or 0
            ):
                continue
            rows.append({
                "rct_id": ids[j],
                "title": titles[j][:220],
                "similarity": round(float(sim[i][j_pos]), 4),
                "registered_on": reg[j],
            })
            if len(rows) >= n_neighbours:
                break
        neighbours[ids[i]] = rows
        if rows:
            sims = [r["similarity"] for r in rows]
            max_prior[i] = sims[0]
            mean_prior[i] = float(np.mean(sims))
        else:  # nothing came before it
            max_prior[i] = 0.0
            mean_prior[i] = 0.0

    return neighbours, dict(zip(ids, max_prior)), dict(zip(ids, mean_prior))


# ==========================================================================
# Corpus-level frequency tables (for rarity components)
# ==========================================================================
def corpus_tables(plans: list[dict]) -> dict:
    jel_c, country_c, unit_c, method_c, combo_c = (
        Counter(), Counter(), Counter(), Counter(), Counter()
    )
    pi_trials: dict[str, list[dict]] = defaultdict(list)

    for p in plans:
        for j in p.get("jel") or []:
            jel_c[j[:1]] += 1
        for c in p.get("countries") or []:
            country_c[c.lower()] += 1
        unit_c[(p.get("randomization_unit") or "").lower()[:40]] += 1
        for m in detect_methods(p):
            method_c[m] += 1
        combo_c[_combo_key(p)] += 1
        for key in p.get("pi_keys") or []:
            pi_trials[key].append(p)

    return {
        "n": len(plans),
        "jel": jel_c,
        "country": country_c,
        "unit": unit_c,
        "method": method_c,
        "combo": combo_c,
        "pi_trials": pi_trials,
    }


def detect_methods(plan: dict) -> list[str]:
    blob = " ".join(
        str(plan.get(f) or "")
        for f in ("experimental_design", "experimental_design_details",
                  "randomization_method", "intervention", "abstract",
                  "primary_outcome_explanation", "treatment_arms")
    )
    return [name for name, rx in METHOD_RE.items() if rx.search(blob)]


def _combo_key(plan: dict) -> str:
    jel = (plan.get("jel") or ["--"])[0][:1]
    country = ((plan.get("countries") or ["--"])[0]).lower()[:20]
    unit = (plan.get("randomization_unit") or "--").lower()[:16]
    return f"{jel}|{country}|{unit}"


def _rarity(count: int, n: int) -> float:
    """Inverse-frequency rarity on a 0-1 scale."""
    if n <= 0 or count <= 0:
        return 1.0
    return _clip01(math.log(n / count) / math.log(n))


# ==========================================================================
# Novelty
# ==========================================================================
def novelty_components(plan: dict, tables: dict, max_sim: float,
                       mean_sim: float, neighbours: list[dict]) -> tuple[dict, list[str]]:
    notes: list[str] = []

    # 1. Semantic rarity: how unlike the single closest earlier plan.
    semantic = _clip01(1.0 - max_sim)
    if not neighbours:
        semantic = 1.0
        notes.append("No earlier plan in the registry is comparable — this is "
                     "the first of its kind by our similarity measure.")
    elif max_sim >= 0.60:
        notes.append(
            f"Very close to an earlier plan ({neighbours[0]['rct_id']}, "
            f"similarity {max_sim:.2f}) — treat as incremental or a replication."
        )
    elif max_sim >= 0.35:
        notes.append(
            f"Related earlier work exists ({neighbours[0]['rct_id']}, "
            f"similarity {max_sim:.2f})."
        )
    else:
        notes.append("No closely similar earlier plan found.")

    # 2. Topic saturation: how crowded the neighbourhood is overall.
    saturation = _clip01(1.0 - mean_sim * 1.35)
    if mean_sim >= 0.35:
        notes.append("Sits in a crowded literature: several earlier plans "
                     "cover adjacent ground.")

    # 3. Method rarity.
    methods = detect_methods(plan)
    if methods:
        rar = [_rarity(tables["method"].get(m, 0), tables["n"]) for m in methods]
        method_rarity = _clip01(max(rar) * 0.7 + (sum(rar) / len(rar)) * 0.3)
        rare_named = [m for m, r in zip(methods, rar) if r > 0.45]
        if rare_named:
            notes.append("Uncommon method(s) for this registry: "
                         + ", ".join(sorted(m.replace("_", " ") for m in rare_named)) + ".")
    else:
        method_rarity = 0.25
        notes.append("No distinctive design or measurement technique detected "
                     "in the text — reads as a standard two-arm trial.")

    # 4. Combination rarity: unusual field x setting x unit mix.
    combo = _rarity(tables["combo"].get(_combo_key(plan), 0), tables["n"])
    countries = [c.lower() for c in (plan.get("countries") or [])]
    if countries:
        c_rar = max(_rarity(tables["country"].get(c, 0), tables["n"]) for c in countries)
        combo = max(combo, 0.5 * combo + 0.5 * c_rar)
        rare_places = [c for c in countries
                       if _rarity(tables["country"].get(c, 0), tables["n"]) > 0.6]
        if rare_places:
            notes.append("Rarely studied setting: "
                         + ", ".join(sorted(p.title() for p in rare_places[:3])) + ".")

    # 5. Priority: credit for getting there first in its own cluster.
    close = [nb for nb in neighbours if nb["similarity"] >= 0.45]
    priority = 1.0 if not close else _clip01(1.0 - len(close) / 6.0)
    if close:
        notes.append(f"{len(close)} earlier plan(s) already occupy this niche.")

    return {
        "semantic_rarity": semantic,
        "topic_saturation": saturation,
        "method_rarity": method_rarity,
        "combination_rarity": combo,
        "priority": priority,
    }, notes


# ==========================================================================
# Feasibility
# ==========================================================================
def implied_mde(n_obs: float | None, n_arms: float | None,
                n_clusters: float | None, icc: float = 0.05) -> float | None:
    """Minimum detectable effect (in SD) at 80% power, alpha = 0.05.

    Two-sided test comparing one treatment arm with control:
        MDE = 2.8 * sqrt(DE) * sqrt(1/n_t + 1/n_c)
    with the design effect DE = 1 + (m - 1) * ICC when randomisation is
    clustered. 2.8 = z_{0.975} + z_{0.80}. The ICC default of 0.05 is a
    conventional middle value for the outcomes this registry deals in; it is
    reported alongside the number so it can be argued with.
    """
    if not n_obs or n_obs <= 1:
        return None
    arms = int(n_arms) if n_arms and n_arms >= 2 else 2
    per_arm = n_obs / arms
    if per_arm < 2:
        return None
    de = 1.0
    if n_clusters and n_clusters >= 2:
        m = n_obs / n_clusters
        if m > 1:
            de = 1.0 + (m - 1.0) * icc
    return 2.8 * math.sqrt(de) * math.sqrt(2.0 / per_arm)


def feasibility_components(plan: dict, tables: dict,
                           paper_by_pi: dict[str, int] | None = None
                           ) -> tuple[dict, list[str], dict]:
    notes: list[str] = []
    flags: dict = {}

    n_obs = plan.get("n_obs_planned")
    n_clusters = plan.get("n_clusters_planned")
    n_arms = plan.get("n_arms")
    mde_reported = plan.get("mde_value")

    # ---- 1. Power adequacy -------------------------------------------------
    power = 0.0
    text = " ".join(str(plan.get(f) or "") for f in
                    ("primary_outcome_explanation", "experimental_design_details",
                     "abstract", "randomization_method"))
    if n_obs:
        power += 0.30
    else:
        notes.append("No planned sample size reported.")
    if n_arms:
        power += 0.05
    if n_clusters:
        power += 0.05
    if plan.get("mde_text"):
        power += 0.20
    else:
        notes.append("No minimum detectable effect stated.")
    if _has(text, POWER_CUES):
        power += 0.15
    if _has(text, MULTIPLICITY_CUES):
        power += 0.10
        flags["handles_multiplicity"] = True
    if _has(text, ATTRITION_CUES):
        power += 0.05
        flags["discusses_attrition"] = True

    imp = implied_mde(n_obs, n_arms, n_clusters)
    if imp is not None:
        flags["implied_mde_sd"] = round(imp, 3)
        if imp <= 0.10:
            power += 0.10
            notes.append(f"Sample supports detecting effects of about "
                         f"{imp:.2f} SD — comfortably powered.")
        elif imp <= 0.25:
            power += 0.05
            notes.append(f"Sample supports about {imp:.2f} SD — adequate for "
                         "a moderate effect.")
        elif imp <= 0.50:
            notes.append(f"Sample only supports about {imp:.2f} SD — powered "
                         "for large effects only.")
        else:
            power -= 0.10
            notes.append(f"Sample supports about {imp:.2f} SD at best — very "
                         "likely underpowered.")
            flags["underpowered"] = True

        if mde_reported and mde_reported > 0:
            ratio = mde_reported / imp
            flags["mde_ratio_reported_to_implied"] = round(ratio, 2)
            if ratio < 0.55:
                power -= 0.15
                notes.append(
                    f"The stated MDE ({mde_reported:.2f} SD) is far smaller "
                    f"than the {imp:.2f} SD this sample supports — the power "
                    "claim and the sample size disagree."
                )
                flags["mde_inconsistent"] = True
            elif ratio <= 2.0:
                power += 0.10
                notes.append("Stated MDE is consistent with the planned sample.")

    if n_obs and n_arms and n_arms >= 2 and n_obs / n_arms < 30:
        power -= 0.15
        notes.append(f"Fewer than 30 observations per arm across {int(n_arms)} "
                     "arms — very thin cells.")
        flags["thin_cells"] = True

    # ---- 2. Design completeness -------------------------------------------
    parts = {
        "randomization_method": _len_score(plan.get("randomization_method"), 160),
        "randomization_unit": 1.0 if plan.get("randomization_unit") else 0.0,
        "treatment_arms": _len_score(plan.get("treatment_arms"), 220),
        "primary_outcomes": _len_score(plan.get("primary_outcomes"), 200),
        "primary_outcome_explanation": _len_score(
            plan.get("primary_outcome_explanation"), 400),
        "experimental_design_details": _len_score(
            plan.get("experimental_design_details"), 700),
        "irb": 1.0 if plan.get("has_irb") else 0.0,
        "analysis_plan": 1.0 if plan.get("has_analysis_plan") else 0.0,
    }
    dweights = {"randomization_method": 0.14, "randomization_unit": 0.08,
                "treatment_arms": 0.12, "primary_outcomes": 0.16,
                "primary_outcome_explanation": 0.16,
                "experimental_design_details": 0.14, "irb": 0.08,
                "analysis_plan": 0.12}
    completeness = sum(parts[k] * w for k, w in dweights.items())
    flags["design_parts"] = {k: round(v, 2) for k, v in parts.items()}
    if plan.get("has_analysis_plan"):
        notes.append("A written analysis plan is attached.")
    else:
        notes.append("No analysis-plan document attached.")
    missing = [k.replace("_", " ") for k, v in parts.items() if v == 0.0]
    if missing:
        notes.append("Left blank: " + ", ".join(sorted(missing)) + ".")

    # ---- 3. Timeline realism ----------------------------------------------
    timeline = 0.5
    prospective = plan.get("prospective")
    if prospective is True:
        timeline += 0.30
        notes.append("Registered before the intervention started.")
    elif prospective is False:
        timeline -= 0.30
        notes.append("Registered after the intervention had already begun — "
                     "this is a retrospective registration.")
        flags["retrospective"] = True

    start = parse_date(plan.get("intervention_start") or plan.get("start_date"))
    end = parse_date(plan.get("intervention_end") or plan.get("end_date"))
    if start and end:
        days = (end - start).days
        flags["duration_days"] = days
        if days <= 0:
            timeline -= 0.15
            notes.append("Intervention end date is not after its start date.")
        elif days < 21:
            timeline -= 0.05
        elif days <= 5 * 365:
            timeline += 0.15
        else:
            timeline -= 0.05
            notes.append("Planned to run for more than five years.")

    stalled = _stalled(plan, end)
    if stalled:
        timeline -= 0.25
        flags["stalled_years"] = stalled
        notes.append(f"End date passed {stalled:.1f} years ago but the trial "
                     "is still listed as under way.")

    # ---- 4. Operational risk (higher score = lower risk) -------------------
    ops = 0.65
    n_countries = plan.get("n_countries") or 0
    if n_countries == 1:
        ops += 0.10
    elif n_countries >= 4:
        ops -= 0.15
        notes.append(f"Runs across {n_countries} countries — heavy coordination load.")
    if plan.get("partners"):
        ops += 0.10
        flags["has_partner"] = True
    else:
        ops -= 0.05
        notes.append("No implementing partner named.")
    if plan.get("sponsors"):
        ops += 0.05
    if n_obs and n_obs > 2_000_000:
        ops -= 0.15
        notes.append("Claims over two million observations — verify the figure.")
        flags["implausible_n"] = True
    if n_clusters and n_obs and n_clusters > n_obs:
        ops -= 0.20
        notes.append("More clusters than observations — the sample-size fields "
                     "are inconsistent.")
        flags["cluster_obs_inconsistent"] = True
    if n_arms and n_arms > 6:
        ops -= 0.10
        notes.append(f"{int(n_arms)} arms is a lot to keep clean in the field.")

    # ---- 5. Team track record ---------------------------------------------
    keys = plan.get("pi_keys") or []
    prior_total = prior_done = 0
    for key in keys:
        for other in tables["pi_trials"].get(key, []):
            if other.get("rct_id") == plan.get("rct_id"):
                continue
            reg_o = other.get("first_registered_on") or ""
            reg_s = plan.get("first_registered_on") or ""
            if reg_o and reg_s and reg_o >= reg_s:
                continue
            prior_total += 1
            if other.get("status") == "completed":
                prior_done += 1
    papers = sum((paper_by_pi or {}).get(k, 0) for k in keys)
    flags["pi_prior_trials"] = prior_total
    flags["pi_prior_completed"] = prior_done
    flags["pi_prior_papers"] = papers

    track = _clip01(
        0.25
        + 0.30 * _clip01(math.log1p(prior_total) / math.log1p(12))
        + 0.30 * _clip01(math.log1p(prior_done) / math.log1p(6))
        + 0.15 * _clip01(math.log1p(papers) / math.log1p(6))
    )
    if prior_done >= 3:
        notes.append(f"The team has completed {prior_done} earlier registered "
                     "trials.")
    elif prior_total == 0:
        notes.append("First registered trial for this team.")

    return (
        {
            "power_adequacy": _clip01(power),
            "design_completeness": _clip01(completeness),
            "timeline_realism": _clip01(timeline),
            "operational_risk": _clip01(ops),
            "team_track_record": track,
        },
        notes,
        flags,
    )


def _stalled(plan: dict, end: date | None) -> float | None:
    if plan.get("status") not in {"on_going", "in_development"}:
        return None
    if not end:
        return None
    years = (today() - end).days / 365.25
    return round(years, 2) if years > 1.5 else None


# ==========================================================================
# Orchestration
# ==========================================================================
def score_all(plans: list[dict], paper_by_pi: dict[str, int] | None = None) -> dict:
    """Score every plan. Returns {rct_id: score record}."""
    plans = [p for p in plans if p.get("rct_id")]
    log.info("Scoring %d plans", len(plans))

    neighbours, max_sim, mean_sim = build_similarity(plans)
    tables = corpus_tables(plans)

    raw: dict[str, dict] = {}
    for p in plans:
        rid = p["rct_id"]
        n_comp, n_notes = novelty_components(
            p, tables, max_sim.get(rid, 0.0), mean_sim.get(rid, 0.0),
            neighbours.get(rid, []),
        )
        f_comp, f_notes, flags = feasibility_components(p, tables, paper_by_pi)
        n_raw = sum(n_comp[k] * w for k, w in config.NOVELTY_WEIGHTS.items())
        f_raw = sum(f_comp[k] * w for k, w in config.FEASIBILITY_WEIGHTS.items())
        raw[rid] = {
            "rct_id": rid,
            "novelty_raw": round(n_raw, 4),
            "feasibility_raw": round(f_raw, 4),
            "novelty_components": {k: round(v, 4) for k, v in n_comp.items()},
            "feasibility_components": {k: round(v, 4) for k, v in f_comp.items()},
            "novelty_notes": n_notes,
            "feasibility_notes": f_notes,
            "flags": flags,
            "methods": detect_methods(p),
            "neighbours": neighbours.get(rid, []),
            "max_prior_similarity": round(float(max_sim.get(rid, 0.0)), 4),
        }

    # Percentile-rank so a score means "better than X% of the registry".
    for field in ("novelty", "feasibility"):
        vals = np.array([raw[r][f"{field}_raw"] for r in raw])
        order = vals.argsort().argsort()
        pct = 100.0 * order / max(len(vals) - 1, 1)
        for rid, p_ in zip(raw.keys(), pct):
            raw[rid][f"{field}_score"] = round(float(p_), 1)
            raw[rid][f"{field}_band"] = band(float(p_))

    log.info("Scored %d plans", len(raw))
    return raw


def band(pct: float) -> str:
    if pct >= 80:
        return "high"
    if pct >= 55:
        return "above average"
    if pct >= 30:
        return "average"
    return "low"
