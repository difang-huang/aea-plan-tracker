"""Turn a raw registry CSV row into a stable, typed plan record.

The export's column names have changed spelling over the years, so every
lookup goes through `pick()` with a list of aliases rather than a hard-coded
header. Unknown columns are preserved in `extra` so nothing is silently lost.
"""

from __future__ import annotations

import re
from typing import Any

from .util import (
    clean_text,
    first_number,
    iso,
    normalize_person,
    parse_date,
    sha1,
    split_multi,
)

# --------------------------------------------------------------------------
# Column aliases: canonical name -> possible export headers
# --------------------------------------------------------------------------
COLUMNS: dict[str, list[str]] = {
    "title": ["Title"],
    "url": ["Url", "URL"],
    "last_update_date": ["Last update date", "Last updated"],
    "published_at": ["Published at"],
    "first_registered_on": ["First registered on", "Registration date"],
    "rct_id": ["RCT_ID", "RCT ID"],
    "doi": ["DOI Number", "DOI"],
    "pi_name": ["Primary Investigator", "Principal Investigator"],
    "status": ["Status", "Project status"],
    "start_date": ["Start date"],
    "end_date": ["End date"],
    "keywords": ["Keywords"],
    "countries": ["Country names", "Countries", "Country"],
    "other_pis": ["Other Primary Investigators", "Other primary investigators"],
    "jel": ["Jel code", "JEL code", "JEL codes"],
    "secondary_ids": ["Secondary IDs", "Secondary ID"],
    "abstract": ["Abstract"],
    "external_links": ["External Links", "External links"],
    "sponsors": ["Sponsors"],
    "partners": ["Partners"],
    "intervention_start": ["Intervention start date"],
    "intervention_end": ["Intervention end date"],
    "intervention": ["Intervention"],
    "primary_outcomes": ["Primary outcome end points", "Primary outcomes"],
    "primary_outcome_explanation": ["Primary outcome explanation"],
    "secondary_outcomes": ["Secondary outcome end points", "Secondary outcomes"],
    "secondary_outcome_explanation": ["Secondary outcome explanation"],
    "experimental_design": ["Experimental design"],
    "experimental_design_details": ["Experimental design details"],
    "randomization_method": ["Randomization method"],
    "randomization_unit": ["Randomization unit"],
    "n_clusters_planned": ["Sample size number clusters", "Sample size: clusters"],
    "n_obs_planned": ["Sample size number observations", "Sample size: observations"],
    "n_arms": ["Sample size number arms", "Sample size: arms"],
    "mde": ["Minimum effect size", "Minimum detectable effect size"],
    "irb": ["IRB"],
    "analysis_plan_docs": ["Analysis Plan Documents", "Analysis plan documents"],
    "intervention_completion_date": ["Intervention completion date"],
    "data_collection_complete": ["Data collection completion"],
    "data_collection_completion_date": ["Data collection completion date"],
    "n_clusters_actual": ["Number of clusters"],
    "attrition_correlated": ["Attrition correlated"],
    "n_obs_actual": ["Total number of observations"],
    "treatment_arms": ["Treatment arms"],
    "public_data": ["Public data"],
    "public_data_url": ["Public data url", "Public data URL"],
    "program_files": ["Program files"],
    "program_files_url": ["Program files url", "Program files URL"],
    "post_trial_documents": ["Post trial documents csv", "Post trial documents"],
    "relevant_papers": ["Relevant papers for csv", "Relevant papers"],
}

_KNOWN_HEADERS = {h for aliases in COLUMNS.values() for h in aliases}

STATUS_MAP = {
    "in_development": "in_development",
    "in development": "in_development",
    "on_going": "on_going",
    "on-going": "on_going",
    "ongoing": "on_going",
    "on going": "on_going",
    "completed": "completed",
    "complete": "completed",
    "withdrawn": "withdrawn",
    "abandoned": "withdrawn",
}

YES = {"yes", "y", "true", "1"}


def pick(row: dict, key: str) -> str:
    for alias in COLUMNS.get(key, [key]):
        if alias in row and row[alias] is not None:
            v = clean_text(row[alias])
            if v:
                return v
    return ""


def _trial_number(url: str, rct_id: str) -> int | None:
    m = re.search(r"/trials/(\d+)", url or "")
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{7})", rct_id or "")
    return int(m.group(1)) if m else None


def _yes(value: str) -> bool | None:
    v = clean_text(value).lower()
    if not v:
        return None
    if v in YES:
        return True
    if v in {"no", "n", "false", "0"}:
        return False
    return None


def normalize_row(row: dict) -> dict:
    """Map one raw CSV row to the canonical plan record."""
    get = lambda k: pick(row, k)  # noqa: E731

    rct_id = get("rct_id")
    url = get("url")
    if not rct_id:
        m = re.search(r"/trials/(\d+)", url)
        if m:
            rct_id = f"AEARCTR-{int(m.group(1)):07d}"

    first_reg = parse_date(get("first_registered_on")) or parse_date(get("published_at"))
    status_raw = get("status")
    status = STATUS_MAP.get(status_raw.strip().lower(), status_raw.strip().lower() or "unknown")

    other_pis = split_multi(get("other_pis"), extra_seps=(",",)) if get("other_pis") else []
    pi_name = get("pi_name")
    all_pis = [p for p in [pi_name, *other_pis] if p]

    rec: dict[str, Any] = {
        "rct_id": rct_id,
        "trial_number": _trial_number(url, rct_id),
        "url": url,
        "doi": get("doi"),
        "title": get("title"),
        "abstract": get("abstract"),
        "status": status,
        "status_raw": status_raw,
        "pi_name": pi_name,
        "pi_key": normalize_person(pi_name),
        "pis": all_pis,
        "pi_keys": sorted({normalize_person(p) for p in all_pis if normalize_person(p)}),
        "n_pis": len(all_pis),

        "first_registered_on": iso(first_reg),
        "registration_year": first_reg.year if first_reg else None,
        "last_update_date": iso(parse_date(get("last_update_date"))),
        "published_at": iso(parse_date(get("published_at"))),
        "start_date": iso(parse_date(get("start_date"))),
        "end_date": iso(parse_date(get("end_date"))),
        "intervention_start": iso(parse_date(get("intervention_start"))),
        "intervention_end": iso(parse_date(get("intervention_end"))),
        "intervention_completion_date": iso(parse_date(get("intervention_completion_date"))),
        "data_collection_completion_date": iso(
            parse_date(get("data_collection_completion_date"))
        ),
        "data_collection_complete": _yes(get("data_collection_complete")),

        "keywords": split_multi(get("keywords"), extra_seps=(",",)),
        "countries": split_multi(get("countries"), extra_seps=(",",)),
        "jel": sorted({j.strip().upper()[:3] for j in
                       split_multi(get("jel"), extra_seps=(",", " ")) if j.strip()}),
        "secondary_ids": get("secondary_ids"),
        "sponsors": split_multi(get("sponsors"), extra_seps=(",",)),
        "partners": split_multi(get("partners"), extra_seps=(",",)),
        "external_links": split_multi(get("external_links"), extra_seps=(" ",)),

        "intervention": get("intervention"),
        "primary_outcomes": get("primary_outcomes"),
        "primary_outcome_explanation": get("primary_outcome_explanation"),
        "secondary_outcomes": get("secondary_outcomes"),
        "secondary_outcome_explanation": get("secondary_outcome_explanation"),
        "experimental_design": get("experimental_design"),
        "experimental_design_details": get("experimental_design_details"),
        "randomization_method": get("randomization_method"),
        "randomization_unit": get("randomization_unit"),
        "treatment_arms": get("treatment_arms"),

        "n_clusters_planned": first_number(get("n_clusters_planned")),
        "n_obs_planned": first_number(get("n_obs_planned")),
        "n_arms": first_number(get("n_arms")),
        "n_clusters_actual": first_number(get("n_clusters_actual")),
        "n_obs_actual": first_number(get("n_obs_actual")),
        "mde_text": get("mde"),

        "irb": get("irb"),
        "analysis_plan_docs": get("analysis_plan_docs"),
        "attrition_correlated": get("attrition_correlated"),
        "public_data": _yes(get("public_data")),
        "public_data_url": get("public_data_url"),
        "program_files": _yes(get("program_files")),
        "program_files_url": get("program_files_url"),
        "post_trial_documents": get("post_trial_documents"),
        "relevant_papers": get("relevant_papers"),
    }

    rec["n_countries"] = len(rec["countries"])
    rec["has_analysis_plan"] = bool(rec["analysis_plan_docs"])
    rec["has_irb"] = bool(rec["irb"])
    rec["mde_value"] = _parse_mde(rec["mde_text"])
    rec["prospective"] = _prospective(rec)
    rec["text_blob"] = build_text_blob(rec)
    rec["content_hash"] = sha1(
        "|".join(
            str(rec.get(k, ""))
            for k in (
                "title", "abstract", "status", "intervention", "primary_outcomes",
                "experimental_design_details", "randomization_method",
                "n_clusters_planned", "n_obs_planned", "n_arms", "mde_text",
                "relevant_papers", "post_trial_documents", "external_links",
                "last_update_date",
            )
        )
    )

    extra = {k: clean_text(v) for k, v in row.items()
             if k not in _KNOWN_HEADERS and clean_text(v)}
    if extra:
        rec["extra"] = extra
    return rec


def _parse_mde(text: str) -> float | None:
    """Best-effort numeric minimum detectable effect, in standard deviations.

    The field is free text: '0.2 SD', '20%', '0.15 standard deviations',
    'a 3 percentage point increase'. We only trust values we can put on a
    common scale; everything else stays None and is treated as 'not reported'.
    """
    s = clean_text(text).lower()
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:sd|s\.d\.|standard deviation)", s)
    if m:
        return abs(float(m.group(1)))
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent|percentage point)", s)
    if m:  # rough conversion: treat x% as x/100 of a SD-like scale
        return abs(float(m.group(1))) / 100.0
    m = re.search(r"^\s*(-?0?\.\d+)\s*$", s)
    if m:
        return abs(float(m.group(1)))
    return None


def _prospective(rec: dict) -> bool | None:
    """Was the plan registered before the intervention started?

    This is the single most informative credibility signal in the registry:
    a plan registered after the intervention began is not really a
    pre-analysis plan.
    """
    reg = rec.get("first_registered_on")
    start = rec.get("intervention_start") or rec.get("start_date")
    if not reg or not start:
        return None
    return reg <= start


TEXT_FIELDS = (
    "title", "abstract", "intervention", "primary_outcomes",
    "primary_outcome_explanation", "secondary_outcomes",
    "experimental_design", "experimental_design_details",
    "randomization_method", "randomization_unit", "treatment_arms",
)


def build_text_blob(rec: dict) -> str:
    """The text used for similarity. Title and abstract are repeated so they
    weigh more than the boilerplate-heavy design fields."""
    parts = [rec.get("title", ""), rec.get("title", ""),
             rec.get("abstract", ""), rec.get("abstract", "")]
    parts += [rec.get(f, "") for f in TEXT_FIELDS[2:]]
    parts += [" ".join(rec.get("keywords", [])), " ".join(rec.get("jel", []))]
    return clean_text(" ".join(p for p in parts if p))
