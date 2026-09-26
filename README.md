# AEA Pre-Registration Plan Tracker

Every pre-analysis plan in the [AEA RCT Registry](https://www.socialscienceregistry.org/),
collected daily, scored for **novelty** and **feasibility**, and linked to the
working paper or published article it became — with the paper's status tracked
over time.

The whole thing runs on a GitHub Actions cron job and publishes to GitHub
Pages. No server, no database, no API keys required.

- **Site:** `https://difang-huang.github.io/aea-plan-tracker/`
- **Data:** [`data/`](data/) — plain CSV, one commit per day, diffable
- **Change log:** [`data/events.jsonl`](data/) — every new plan, status change and paper discovery

---

## What it does

### 1. Collect (daily)

`tracker/collect.py` pulls the registry's full CSV export, normalises it into a
typed record per trial, and shards the result by registration year
(`data/plans/plans_2024.csv`). Only the current year's shard changes on a normal
day, so daily commits stay small even though the corpus is ~13,000 plans.

Every difference against the previous snapshot is appended to
`data/events.jsonl`. That file is the project's memory: it makes "what happened
this week" and "how long from registration to paper" answerable, which a
snapshot alone never can.

Two guards keep a bad upstream day from destroying good data: a truncated
download is rejected, and so is any export with more than 10% fewer plans than
we already have.

### 2. Score

Both scores are **corpus-relative percentiles** — 70 means the plan ranks above
70% of the registry on that dimension, not that it is "70% good". Nothing is a
black box: each score is a weighted sum of named components, every component
comes from stated registry fields, and every plan carries the sentences that
justify its number.

**Novelty** asks *has anyone already pre-registered this?* It is measured only
against plans registered **earlier** — a plan cannot be made unoriginal by
something that came after it.

| Component | Weight | What it measures |
|---|---|---|
| Semantic rarity | 0.40 | 1 − cosine similarity to the closest earlier plan (TF-IDF over title, abstract, intervention, outcomes, design) |
| Topic saturation | 0.20 | How crowded the neighbourhood is overall |
| Method rarity | 0.15 | Inverse corpus frequency of ~23 detected designs and measurement strategies (encouragement, factorial, adaptive, saturation, list experiments, admin-data linkage, causal forests, …) |
| Combination rarity | 0.15 | How unusual the JEL × country × randomisation-unit mix is |
| Priority | 0.10 | First-mover credit within its own cluster |

**Feasibility** asks *if this runs as written, will it produce a credible
answer?* It ignores how interesting the question is.

| Component | Weight | What it measures |
|---|---|---|
| Power adequacy | 0.28 | Implied MDE from the stated sample vs. the MDE the authors claim; power/ICC/multiplicity/attrition language; cell sizes |
| Design completeness | 0.24 | Presence and substance of randomisation method and unit, arms, primary outcomes and their explanation, design details, IRB, attached analysis plan |
| Timeline realism | 0.18 | Prospective vs. retrospective registration, plausible duration, stalled trials |
| Operational risk | 0.15 | Country count, named partner, internally consistent sample-size fields, arm count |
| Team track record | 0.15 | The PI's earlier registered trials, how many they finished, how many produced papers |

The power check is a real computation, not a keyword count:

```
MDE = 2.8 · √(design effect) · √(2 / n_per_arm)     design effect = 1 + (m − 1)·ICC
```

with `2.8 = z₀.₉₇₅ + z₀.₈₀` and a default ICC of 0.05 when randomisation is
clustered. When a plan states an MDE far below what its sample can support, the
plan is flagged: its power claim and its sample size disagree.

**Optional LLM layer.** With an `ANTHROPIC_API_KEY` secret set, new or changed
plans also get a qualitative read. The model is shown the plan's five nearest
**earlier** plans and asked to judge novelty against *those specific plans* —
asking "is this novel?" with no comparison set just rewards fashionable topics.
Verdicts are cached under the plan's content hash, so each plan is judged once.
Rule-based and model scores are both shown; the headline blends them
(65/35 by default). Without a key, everything still works — you lose the prose,
not the scores.

### 3. Match plans to papers

An evidence ladder, strongest rung first. Every stored link records which rung
it came from and why, so you can filter to author-declared links alone if you
want a zero-inference dataset.

| Rung | Method | Confidence | Source |
|---|---|---|---|
| 1 | DOI declared by the authors in the registry entry | 0.97 | Registry → OpenAlex/Crossref |
| 1b | arXiv id declared by the authors | 0.95 | Registry → arXiv |
| 2 | Paper's full text cites the trial id (`AEARCTR-0001234`) | 0.93 | OpenAlex full-text search |
| 3 | Author-listed citation without a DOI, resolved by title | 0.70–0.88 | Crossref |
| 4 | Shares an author with the plan **and** the text overlaps | 0.40–0.79 | OpenAlex + TF-IDF |

Rung 4 is the only inferential one; it is capped below "confirmed", requires an
author-surname overlap (content similarity alone is too weak), and is skipped
for trials younger than 18 months. Its similarity score always travels with it
as evidence.

Each link is classified **published** vs. **working paper** from the venue type
and name, so an NBER or CEPR discussion paper never gets counted as a
publication.

### 4. Track status

```
registered → under way → completed, no paper → candidate paper → working paper → published
```

Transitions are written to the event log with dates, which yields the
registration→paper lag distribution and the site's activity feed.

The tracker also computes a **results-overdue** flag: trials whose data
collection finished 2+ years ago with no paper found anywhere. That is a
descriptive publication-bias signal and is deliberately **never** used to
penalise any score.

### 5. Publish

`tracker/build_site.py` writes a static site to `docs/`: a searchable, sortable
explorer over every plan, a page per plan showing the score decomposition, the
sentences behind it, the nearest earlier plans and every matched paper, plus a
dashboard. No JavaScript framework and no CDN — hand-rolled SVG charts, light
and dark themes, and it works from any static host.

---

## Setup

1. Create a repository and push this code to it.
2. **Settings → Pages → Source: GitHub Actions.**
3. **Actions → Daily update → Run workflow.** The first run backfills the whole
   registry and takes roughly 10–20 minutes.

That is the whole setup. The cron then runs at 06:10 UTC daily.

### Optional configuration

| Where | Name | Effect |
|---|---|---|
| Settings → Secrets → Actions | `ANTHROPIC_API_KEY` | Turns on the LLM layer |
| Settings → Variables → Actions | `CONTACT_EMAIL` | Puts you in OpenAlex's and Crossref's polite pools (faster, friendlier) |

Tunables live in `tracker/config.py`: component weights, confidence thresholds,
per-run API budgets, and how often matched and unmatched plans are re-checked.

### First run and backfill

Matching is budgeted (500 plans per run by default) so a daily job stays inside
every service's rate limits and finishes quickly. The full corpus therefore
backfills over roughly a month of daily runs, prioritised sensibly: never-checked
plans first, then plans whose text changed, then stale matched plans, then stale
unmatched ones — and within each tier, completed and older trials first, since
those are the ones likely to have papers. To go faster, run the workflow
manually with a larger `match_budget`.

Scoring is not budgeted: all ~13,000 plans are rescored every run, which takes
well under a minute.

---

## Local use

```bash
pip install -r requirements.txt

python -m tracker.cli daily                 # the whole pipeline
python -m tracker.cli collect               # download + diff only
python -m tracker.cli score --no-llm        # rescore from stored plans
python -m tracker.cli match --budget 50     # match 50 plans
python -m tracker.cli site                  # rebuild docs/ from stored data

# no network at all — uses the synthetic fixture
python scripts/make_fixture.py fixtures/registry_sample.csv 500
python -m tracker.cli collect --offline --raw-csv fixtures/registry_sample.csv
python -m tracker.cli site --offline --no-llm
python -m http.server 8000 --directory docs

pytest -q
```

---

## Data dictionary

| File | One row per | Notable columns |
|---|---|---|
| `data/plans/plans_YYYY.csv` | registered plan | `rct_id`, `title`, `abstract`, `status`, `first_registered_on`, `n_obs_planned`, `n_clusters_planned`, `n_arms`, `mde_text`, `has_analysis_plan`, `prospective`, `content_hash` |
| `data/scores.csv` | plan | `novelty_score`, `feasibility_score` (percentiles), `*_raw`, `*_llm`, `*_blended`, `max_prior_similarity`, `closest_prior_plan`, `methods`, `implied_mde_sd`, `underpowered`, `retrospective` |
| `data/papers.csv` | plan × paper link | `kind`, `title`, `venue`, `doi`, `url`, `cited_by`, `method`, `confidence`, `evidence`, `first_seen` |
| `data/match_state.csv` | plan | `last_checked`, `checked_hash`, `n_papers`, `best_kind` — the matcher's scheduling memory |
| `data/events.jsonl` | event | `date`, `type`, `rct_id`, plus type-specific fields |

`prospective = False` means the plan was registered *after* its intervention
started. It is the single most informative credibility signal in the registry
and is worth filtering on before using these data for anything.

---

## Caveats worth reading before you cite this

- **Scores are a triage tool, not a verdict.** They read what authors wrote in
  structured fields. A superbly designed trial described tersely will score
  below a mediocre one described at length. Use them to sort 13,000 plans down
  to 50 worth reading, then read those 50.
- **Novelty is textual.** Cosine similarity over TF-IDF finds plans that *talk*
  alike. Two studies with identical designs and different vocabulary will look
  unrelated to it; the near-neighbour list on each plan page is there so you can
  see exactly what the score reacted to.
- **The implied-MDE calculation assumes ICC = 0.05** for clustered designs.
  That is a conventional middle value, not this trial's actual ICC. It is shown
  on the plan page so you can substitute your own.
- **Rung-4 paper matches are inferences.** Filter `method == "author_similarity"`
  out if you need precision over recall.
- **Absence of a paper is not evidence of absence.** Coverage depends on
  OpenAlex and Crossref indexing. The results-overdue flag is a prompt to go and
  look, not a finding.

## Sources and etiquette

Data from the [AEA RCT Registry](https://www.socialscienceregistry.org/)
(one export pull a day; the registry refreshes it roughly every ten minutes),
[OpenAlex](https://openalex.org/), [Crossref](https://www.crossref.org/) and
[arXiv](https://arxiv.org/). All requests identify the bot and carry a contact
address when `CONTACT_EMAIL` is set. Responses are cached so reruns cost the
upstreams nothing. If you build on the registry data, cite it as the registry
asks; monthly snapshots with DOIs are on the
[Harvard Dataverse](https://dataverse.harvard.edu/dataverse/aearegistry).

## Licence

MIT for the code. The registry data and bibliographic metadata carry their own
upstream terms.
