"""Central configuration for the AEA pre-registration plan tracker."""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLANS_DIR = DATA / "plans"          # sharded by registration year
BUILD = ROOT / "build"              # untracked scratch (raw snapshots)
DOCS = ROOT / "docs"                # GitHub Pages root
DOCS_DATA = DOCS / "data"
CACHE = DATA / "cache"              # api + llm response cache

SCORES_CSV = DATA / "scores.csv"
PAPERS_CSV = DATA / "papers.csv"
MATCH_STATE_CSV = DATA / "match_state.csv"
EVENTS_JSONL = DATA / "events.jsonl"
CORPUS_STATS_JSON = DATA / "corpus_stats.json"

for _p in (DATA, PLANS_DIR, BUILD, DOCS, DOCS_DATA, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Source: AEA RCT Registry
# --------------------------------------------------------------------------
REGISTRY_BASE = "https://www.socialscienceregistry.org"
# The registry's bulk export (the "Download as CSV" button on the advanced
# search page). This is the whole registry in one file; the server takes a
# couple of minutes to generate it, hence the long timeout below.
REGISTRY_CSV_URL = f"{REGISTRY_BASE}/site/csv"
# Fallback: the search endpoint also emits CSV, but only 20 rows per page and
# it ignores per_page, so it has to be walked page by page.
REGISTRY_SEARCH_CSV_URL = f"{REGISTRY_BASE}/trials/search.csv"
REGISTRY_PAGE_SIZE = 20
TRIAL_URL_FMT = f"{REGISTRY_BASE}/trials/{{n}}"

# Identify ourselves. Registries and APIs treat a real UA + contact as polite.
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "").strip()
USER_AGENT = (
    "aea-plan-tracker/1.0 (+https://github.com/{repo}; research bot)".format(
        repo=os.environ.get("GITHUB_REPOSITORY", "aea-plan-tracker")
    )
    + (f" mailto:{CONTACT_EMAIL}" if CONTACT_EMAIL else "")
)

# The registry refreshes its export roughly every ten minutes; one pull a day
# is far inside any reasonable politeness budget.
HTTP_TIMEOUT = 180
# The bulk export is built on demand and can take several minutes.
BULK_TIMEOUT = int(os.environ.get("BULK_TIMEOUT", "900"))
# Pause between pages when falling back to paginated search.
PAGE_SLEEP = float(os.environ.get("PAGE_SLEEP", "0.4"))
HTTP_RETRIES = 4
HTTP_BACKOFF = 5.0

# A full registry pull should land in this ballpark; anything far below it
# means we got a search page instead of the bulk export.
MIN_EXPECTED_PLANS = int(os.environ.get("MIN_EXPECTED_PLANS", "5000"))

# --------------------------------------------------------------------------
# Bibliographic sources
# --------------------------------------------------------------------------
OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org"
ARXIV_BASE = "http://export.arxiv.org/api/query"

# Per-run budget so a daily job stays well inside every service's limits and
# finishes in a few minutes. Backfill therefore spreads over several days.
MATCH_BUDGET_PER_RUN = int(os.environ.get("MATCH_BUDGET_PER_RUN", "500"))
# Hard wall-clock stop for the matching step. The CI job has a 90-minute
# budget; matching must never be the reason the site fails to rebuild, so it
# stops early and leaves the rest for tomorrow.
MATCH_TIME_BUDGET_SEC = int(os.environ.get("MATCH_TIME_BUDGET_SEC", "2400"))
# Re-check an already-matched plan this often (papers get published later).
REMATCH_AFTER_DAYS = int(os.environ.get("REMATCH_AFTER_DAYS", "45"))
# Re-check an unmatched plan less eagerly the older and quieter it is.
RETRY_UNMATCHED_AFTER_DAYS = int(os.environ.get("RETRY_UNMATCHED_AFTER_DAYS", "30"))

OPENALEX_SLEEP = float(os.environ.get("OPENALEX_SLEEP", "0.12"))
CROSSREF_SLEEP = float(os.environ.get("CROSSREF_SLEEP", "0.25"))
ARXIV_SLEEP = float(os.environ.get("ARXIV_SLEEP", "3.0"))  # arXiv asks for 3s

# --------------------------------------------------------------------------
# Matching thresholds
# --------------------------------------------------------------------------
# Confidence bands. Anything below CANDIDATE_MIN is discarded outright.
MATCH_CONFIRMED = 0.80   # shown as "matched"
MATCH_CANDIDATE = 0.55   # shown as "likely / needs review"
CANDIDATE_MIN = 0.40

# --------------------------------------------------------------------------
# LLM layer (optional)
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
LLM_ENABLED = bool(ANTHROPIC_API_KEY) and os.environ.get("LLM_ENABLED", "1") != "0"
LLM_BUDGET_PER_RUN = int(os.environ.get("LLM_BUDGET_PER_RUN", "120"))
# Weight on the LLM judgement in the blended headline score.
LLM_WEIGHT = float(os.environ.get("LLM_WEIGHT", "0.35"))

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
NOVELTY_WEIGHTS = {
    "semantic_rarity": 0.40,   # how unlike every earlier plan it is
    "topic_saturation": 0.20,  # how crowded its neighbourhood is
    "method_rarity": 0.15,     # unusual designs / measurement
    "combination_rarity": 0.15,  # unusual JEL x country x outcome mix
    "priority": 0.10,          # first-mover credit within its cluster
}

FEASIBILITY_WEIGHTS = {
    "power_adequacy": 0.28,
    "design_completeness": 0.24,
    "timeline_realism": 0.18,
    "operational_risk": 0.15,
    "team_track_record": 0.15,
}

# Number of nearest earlier plans kept per plan (shown on the site, and fed
# to the LLM as the comparison set).
N_NEIGHBOURS = 6

# Similarity space. Novelty is measured with TF-IDF cosine over the plan text.
#
# The standing objection to that is paraphrase: two plans describing the same
# design in different words ("cash grant" / "unconditional transfer") look
# unrelated to a bag of words. The usual fix is a latent or embedding space.
# It was tried: LSA over this same matrix, blended with the lexical space. On a
# 600-plan benchmark with hand-written paraphrase pairs -- same study,
# deliberately disjoint vocabulary -- the blend ranked the paraphrase partner
# *identically* to pure TF-IDF, and strictly worse when weighted further toward
# the latent space. So it is not shipped: an unvalidated change does not get to
# move anybody's published score.
#
# Instead the latent space is measured on every run, at registry scale, against
# the lexical neighbours actually published (see probe_latent_space in
# score.py). If the probe shows real paraphrase pairs that TF-IDF misses, the
# blend is worth building; if it keeps showing none, the objection was wrong.
LSA_COMPONENTS = int(os.environ.get("LSA_COMPONENTS", "300"))
# Plans sampled per run for the probe. 0 skips it.
SIM_PROBE_SAMPLE = int(os.environ.get("SIM_PROBE_SAMPLE", "800"))
# The case the lexical score is accused of missing: two plans the latent space
# calls close (>= PROBE_LATENT_HIT) while TF-IDF calls them barely related
# (< PROBE_LEXICAL_FLOOR). Every such pair is recorded so it can be read.
PROBE_LEXICAL_FLOOR = float(os.environ.get("PROBE_LEXICAL_FLOOR", "0.20"))
PROBE_LATENT_HIT = float(os.environ.get("PROBE_LATENT_HIT", "0.60"))
PROBE_EXAMPLES = int(os.environ.get("PROBE_EXAMPLES", "15"))

SITE_TITLE = "AEA Pre-Registration Plan Tracker"
SITE_TAGLINE = (
    "Every plan in the AEA RCT Registry, scored for novelty and feasibility, "
    "and linked to the paper it became."
)
