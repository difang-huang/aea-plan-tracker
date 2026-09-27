"""Unit tests for the parts that must not silently drift.

Network is never touched: the matcher's source calls are monkeypatched.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from tracker import match, normalize, score
from tracker.util import (
    clean_doi, extract_dois, extract_rct_ids, normalize_person, parse_date,
    rct_id_variants, split_multi, surname,
)


# ---------------------------------------------------------------- util ----
@pytest.mark.parametrize("raw,expected", [
    ("2021-03-04", date(2021, 3, 4)),
    ("March 04, 2021", date(2021, 3, 4)),
    ("Mar 4, 2021", date(2021, 3, 4)),
    ("03/04/2021", date(2021, 3, 4)),
    ("2021-03-04T09:00:00", date(2021, 3, 4)),
    ("", None),
    ("nan", None),
    ("not a date", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_split_multi_prefers_strong_separators():
    # A semicolon list must not be shredded on the commas inside names.
    assert split_multi("Duflo, Esther; Banerjee, Abhijit") == [
        "Duflo, Esther", "Banerjee, Abhijit"]
    assert split_multi("Kenya, India", extra_seps=(",",)) == ["Kenya", "India"]
    assert split_multi("") == []


def test_person_normalisation():
    assert normalize_person("Duflo, Esther") == "esther duflo"
    assert normalize_person("Prof. Esther Duflo") == "esther duflo"
    assert surname("Duflo, Esther") == "duflo"


def test_doi_and_id_extraction():
    text = ('See "Cash and welfare", AER 2023, https://doi.org/10.1257/aer.20211234 '
            "and the registry entry AEARCTR-0004321.")
    assert extract_dois(text) == ["10.1257/aer.20211234"]
    assert extract_rct_ids(text) == ["AEARCTR-0004321"]
    assert clean_doi("https://dx.doi.org/10.1257/AER.123.") == "10.1257/aer.123"
    assert "AEARCTR-0004321" in rct_id_variants("AEARCTR-0004321")


# ----------------------------------------------------------- normalize ----
BASE_ROW = {
    "Title": "A trial", "Url": "https://www.socialscienceregistry.org/trials/4242",
    "RCT_ID": "AEARCTR-0004242", "Primary Investigator": "Duflo, Esther",
    "Other Primary Investigators": "Banerjee, Abhijit; Karlan, Dean",
    "Status": "On-going", "First registered on": "March 04, 2021",
    "Intervention start date": "2021-06-01", "Intervention end date": "2022-06-01",
    "Country names": "Kenya, India", "Jel code": "O12 C93",
    "Sample size number observations": "2,400 individuals",
    "Sample size number arms": "3", "Sample size number clusters": "80",
    "Minimum effect size": "0.2 SD", "Abstract": "We study cash transfers.",
}


def test_normalize_row_core_fields():
    rec = normalize.normalize_row(BASE_ROW)
    assert rec["rct_id"] == "AEARCTR-0004242"
    assert rec["trial_number"] == 4242
    assert rec["status"] == "on_going"
    assert rec["registration_year"] == 2021
    assert rec["countries"] == ["Kenya", "India"]
    assert rec["jel"] == ["C93", "O12"]
    assert rec["n_obs_planned"] == 2400.0
    assert rec["n_arms"] == 3.0
    assert rec["mde_value"] == pytest.approx(0.2)
    assert rec["n_pis"] == 3
    assert "esther duflo" in rec["pi_keys"]
    assert rec["prospective"] is True
    assert rec["content_hash"]


def test_retrospective_registration_detected():
    row = dict(BASE_ROW, **{"Intervention start date": "2020-01-01"})
    assert normalize.normalize_row(row)["prospective"] is False


def test_rct_id_recovered_from_url():
    row = dict(BASE_ROW); row["RCT_ID"] = ""
    assert normalize.normalize_row(row)["rct_id"] == "AEARCTR-0004242"


@pytest.mark.parametrize("text,expected", [
    ("0.2 SD", 0.2), ("0.15 standard deviations", 0.15),
    ("5 percentage points", 0.05), ("10%", 0.10), ("0.25", 0.25),
    ("we hope for a big effect", None), ("", None),
])
def test_mde_parsing(text, expected):
    got = normalize._parse_mde(text)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


# --------------------------------------------------------------- score ----
def test_implied_mde_matches_the_textbook_formula():
    # Unclustered, two arms, 800 observations -> 400 per arm.
    got = score.implied_mde(800, 2, None)
    assert got == pytest.approx(2.8 * math.sqrt(2 / 400), rel=1e-9)
    # Clustering inflates the detectable effect.
    assert score.implied_mde(800, 2, 20) > got
    assert score.implied_mde(0, 2, None) is None


def test_implied_mde_falls_with_sample_size():
    small = score.implied_mde(200, 2, None)
    large = score.implied_mde(20000, 2, None)
    assert large < small


def test_method_detection():
    plan = {"experimental_design_details":
            "A 2x2 factorial design with an encouragement design component.",
            "abstract": "We use a causal forest for heterogeneity."}
    found = set(score.detect_methods(plan))
    assert {"factorial", "encouragement_design", "ml_heterogeneity"} <= found


def _plan(**kw):
    base = dict(normalize.normalize_row(BASE_ROW))
    base.update(kw)
    return base


def test_scoring_produces_percentiles_and_explanations():
    plans = []
    for i in range(30):
        row = dict(BASE_ROW)
        row["RCT_ID"] = f"AEARCTR-{4000 + i:07d}"
        row["Url"] = f"https://www.socialscienceregistry.org/trials/{4000 + i}"
        row["Title"] = f"Trial {i} about topic {i % 5}"
        row["Abstract"] = f"We study topic {i % 5} with method {i % 3}."
        row["Sample size number observations"] = str(200 * (i + 1))
        plans.append(normalize.normalize_row(row))

    scores = score.score_all(plans)
    assert len(scores) == len(plans)
    for s in scores.values():
        assert 0 <= s["novelty_score"] <= 100
        assert 0 <= s["feasibility_score"] <= 100
        assert s["novelty_notes"] and s["feasibility_notes"]
        assert set(s["novelty_components"]) == set(
            score.config.NOVELTY_WEIGHTS)
    # Percentiles span the range.
    vals = sorted(s["novelty_score"] for s in scores.values())
    assert vals[0] == 0.0 and vals[-1] == 100.0


def test_novelty_only_looks_backwards():
    """A plan registered first can never be penalised by a later copy."""
    rows = []
    for i, reg in enumerate(["2015-01-01", "2020-01-01"]):
        row = dict(BASE_ROW)
        row["RCT_ID"] = f"AEARCTR-{5000 + i:07d}"
        row["Url"] = f"https://www.socialscienceregistry.org/trials/{5000 + i}"
        row["First registered on"] = reg
        row["Title"] = "Identical cash transfer trial"
        row["Abstract"] = "We study unconditional cash transfers in Kenya."
        rows.append(normalize.normalize_row(row))
    # Pad the corpus so TF-IDF has vocabulary.
    for i in range(8):
        row = dict(BASE_ROW)
        row["RCT_ID"] = f"AEARCTR-{6000 + i:07d}"
        row["Url"] = f"https://www.socialscienceregistry.org/trials/{6000 + i}"
        row["Title"] = f"Unrelated study {i} of schooling and teachers"
        row["Abstract"] = f"Teacher incentives and test scores, variant {i}."
        rows.append(normalize.normalize_row(row))

    scores = score.score_all(rows)
    first, second = scores["AEARCTR-0005000"], scores["AEARCTR-0005001"]
    assert first["neighbours"] == [] or all(
        n["registered_on"] <= "2015-01-01" for n in first["neighbours"])
    assert any(n["rct_id"] == "AEARCTR-0005000" for n in second["neighbours"])
    assert second["max_prior_similarity"] > first["max_prior_similarity"]


def test_latent_probe_reports_without_changing_scores(monkeypatch):
    """The probe measures what a latent space would change, and nothing else."""
    rows = []
    for i in range(80):
        row = dict(BASE_ROW)
        row["RCT_ID"] = f"AEARCTR-{7000 + i:07d}"
        row["Url"] = f"https://www.socialscienceregistry.org/trials/{7000 + i}"
        row["Title"] = f"Study {i} of topic {i % 7}"
        row["Abstract"] = (f"We randomise topic {i % 7} across {i} villages "
                           f"measuring outcome {i % 4} with method {i % 3}.")
        rows.append(normalize.normalize_row(row))

    monkeypatch.setattr(score.config, "SIM_PROBE_SAMPLE", 40)
    first = score.score_all(rows)
    probe = dict(score.SIMILARITY_DIAGNOSTICS)

    assert probe.get("status") in {"measured"} or probe["status"].startswith("skipped")
    if probe["status"] == "measured":
        assert probe["sample"] == 40
        assert 0.0 <= probe["mean_topk_overlap"] <= 1.0
        assert probe["pct_rows_with_latent_only_hit"] >= 0.0
        for ex in probe["examples"]:
            assert ex["latent_similarity"] >= score.config.PROBE_LATENT_HIT
            assert ex["lexical_similarity"] < score.config.PROBE_LEXICAL_FLOOR

    # Running the probe must leave the published scores bit-for-bit identical.
    monkeypatch.setattr(score.config, "SIM_PROBE_SAMPLE", 0)
    second = score.score_all(rows)
    assert score.SIMILARITY_DIAGNOSTICS.get("status") == "disabled"
    for field in ("novelty_score", "feasibility_score", "max_prior_similarity"):
        assert ({r: v[field] for r, v in first.items()}
                == {r: v[field] for r, v in second.items()})


# --------------------------------------------------------------- match ----
def test_declared_references_parses_the_free_text_field():
    plan = {"relevant_papers":
            '"Cash and welfare", AER 2023. https://doi.org/10.1257/aer.20211234 '
            "| Working paper: NBER Working Paper w28123",
            "post_trial_documents": "", "external_links": [],
            "public_data_url": "", "program_files_url": ""}
    got = match.declared_references(plan)
    assert "10.1257/aer.20211234" in got["dois"]
    assert "28123" in got["nber"]
    assert any("Cash and welfare" in t for t in got["titles"])


def test_registry_own_doi_is_not_treated_as_a_paper():
    plan = {"relevant_papers": "https://doi.org/10.1257/rct.4242-1.0",
            "post_trial_documents": "", "external_links": [],
            "public_data_url": "", "program_files_url": ""}
    assert match.declared_references(plan)["dois"] == []


def test_plan_stage_ladder():
    plan = {"status": "completed"}
    assert match.plan_stage(plan, []) == "completed_no_paper"
    assert match.plan_stage(plan, [{"kind": "working_paper", "confidence": 0.95}]) \
        == "working_paper"
    assert match.plan_stage(plan, [{"kind": "published", "confidence": 0.95}]) \
        == "published"
    # A weak inferred link is a candidate, never a confirmed paper.
    assert match.plan_stage(plan, [{"kind": "published", "confidence": 0.6}]) \
        == "candidate_paper"
    assert match.plan_stage({"status": "on_going"}, []) == "in_progress"


def test_results_overdue_only_for_finished_trials_without_papers():
    old = (date.today() - timedelta(days=1200)).isoformat()
    plan = {"status": "completed", "data_collection_completion_date": old}
    assert match.results_overdue(plan, "completed_no_paper") > 3.0
    assert match.results_overdue(plan, "published") is None
    recent = {"status": "completed",
              "data_collection_completion_date": date.today().isoformat()}
    assert match.results_overdue(recent, "completed_no_paper") is None


def test_match_plan_uses_declared_doi_first(monkeypatch):
    work = {
        "id": "https://openalex.org/W123", "doi": "https://doi.org/10.1257/aer.20211234",
        "display_name": "Cash and welfare", "publication_year": 2023,
        "publication_date": "2023-05-01", "type": "article",
        "cited_by_count": 12,
        "authorships": [{"author": {"display_name": "Esther Duflo"}}],
        "primary_location": {"source": {"display_name": "American Economic Review",
                                        "type": "journal"}},
    }
    monkeypatch.setattr(match.sources, "oa_by_doi", lambda doi: work)
    monkeypatch.setattr(match.sources, "oa_fulltext", lambda phrase: [])
    monkeypatch.setattr(match.sources, "arxiv_search",
                        lambda q, max_results=10: [])
    monkeypatch.setattr(match.sources, "crossref_search_title",
                        lambda t, rows=5: [])

    plan = _plan(relevant_papers="https://doi.org/10.1257/aer.20211234")
    got = match.match_plan(plan)
    assert len(got) == 1
    assert got[0]["confidence"] >= match.config.MATCH_CONFIRMED
    assert got[0]["kind"] == "published"
    assert got[0]["method"] == "declared_doi"
    assert got[0]["venue"] == "American Economic Review"


def test_working_paper_is_classified_separately(monkeypatch):
    wp = {
        "id": "https://openalex.org/W999", "doi": "https://doi.org/10.3386/w28123",
        "display_name": "Cash transfers: a working paper", "publication_year": 2022,
        "publication_date": "2022-02-01", "type": "article", "cited_by_count": 3,
        "authorships": [], "primary_location":
            {"source": {"display_name": "NBER Working Paper Series",
                        "type": "repository"}},
    }
    assert match.sources.classify_work(wp) == "working_paper"


def test_inference_is_skipped_for_young_trials():
    fresh = _plan(first_registered_on=date.today().isoformat())
    assert match._worth_inferring(fresh) is False
    old = _plan(first_registered_on="2015-01-01", status="completed")
    assert match._worth_inferring(old) is True
