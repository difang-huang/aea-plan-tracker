"""Generate a synthetic registry export for offline development and CI.

The header is the real one from the AEA export; the rows are invented but
reproduce the quirks the parser has to survive: free-text sample sizes,
mixed date spellings, multi-value cells with different separators, embedded
commas and quotes, retrospective registrations, blank fields, and a handful
of plans that declare a paper.

    python scripts/make_fixture.py fixtures/registry_sample.csv 400
"""

from __future__ import annotations

import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

HEADER = [
    "Title", "Url", "Last update date", "Published at", "First registered on",
    "RCT_ID", "DOI Number", "Primary Investigator", "Status", "Start date",
    "End date", "Keywords", "Country names", "Other Primary Investigators",
    "Jel code", "Secondary IDs", "Abstract", "External Links", "Sponsors",
    "Partners", "Intervention start date", "Intervention end date",
    "Intervention", "Primary outcome end points", "Primary outcome explanation",
    "Secondary outcome end points", "Secondary outcome explanation",
    "Experimental design", "Experimental design details", "Randomization method",
    "Randomization unit", "Sample size number clusters",
    "Sample size number observations", "Sample size number arms",
    "Minimum effect size", "IRB", "Analysis Plan Documents",
    "Intervention completion date", "Data collection completion",
    "Data collection completion date", "Number of clusters",
    "Attrition correlated", "Total number of observations", "Treatment arms",
    "Public data", "Public data url", "Program files", "Program files url",
    "Post trial documents csv", "Relevant papers for csv",
]

TOPICS = [
    ("cash transfers", "unconditional cash transfers to poor households",
     "household consumption, food security", "C93 O12"),
    ("microfinance", "group liability microcredit for small enterprises",
     "business profits, loan take-up", "G21 O16"),
    ("teacher incentives", "performance pay for primary school teachers",
     "test scores in maths and literacy", "I21 M52"),
    ("health insurance", "subsidised community health insurance enrolment",
     "out-of-pocket health spending, utilisation", "I13 O15"),
    ("job search assistance", "structured job search coaching for the unemployed",
     "employment rate at six months, earnings", "J64 J68"),
    ("information nudges", "SMS reminders about vaccination appointments",
     "vaccination completion rate", "I12 D91"),
    ("agricultural extension", "farmer field schools on fertiliser use",
     "yields per hectare, fertiliser adoption", "Q12 O13"),
    ("savings commitment", "commitment savings accounts with default deposits",
     "savings balance after 12 months", "D14 G51"),
    ("energy conservation", "social comparison reports on electricity bills",
     "kWh consumed per month", "Q41 D12"),
    ("tax compliance", "deterrence letters to self-employed filers",
     "declared income, filing rate", "H26 K42"),
    ("early childhood", "home visiting programme for caregivers of infants",
     "child cognitive development score", "I21 J13"),
    ("governance audits", "community monitoring of local public works",
     "leakage in project expenditure", "D73 H41"),
]
COUNTRIES = ["Kenya", "India", "Bangladesh", "Mexico", "Uganda", "United States",
             "Brazil", "Indonesia", "Ghana", "Peru", "Malawi", "Pakistan",
             "Rwanda", "Nepal", "Colombia", "Philippines", "Tanzania", "Chile"]
NAMES = ["Duflo, Esther", "Banerjee, Abhijit", "Karlan, Dean", "Miguel, Edward",
         "Kremer, Michael", "Ashraf, Nava", "Bandiera, Oriana", "Fafchamps, Marcel",
         "Olken, Benjamin", "Field, Erica", "Dupas, Pascaline", "Jayachandran, Seema",
         "Casey, Katherine", "Blattman, Christopher", "Ozier, Owen", "Haushofer, Johannes"]
DESIGNS = [
    ("Individual level randomization into treatment and control.", "Individual"),
    ("Cluster randomized at the village level, stratified by district.", "Village"),
    ("2 x 2 factorial design crossing the price subsidy with the information arm.", "Household"),
    ("Encouragement design: randomly selected households receive an invitation.", "Household"),
    ("Matched-pair cluster randomization across schools within each block.", "School"),
    ("Stepped-wedge rollout across 40 clinics over 18 months.", "Clinic"),
]
RANDOM_METHODS = [
    "Randomization done in office by a computer using Stata's runiform with a fixed seed.",
    "Public lottery held in each village with community members observing.",
    "Re-randomization until covariate balance on baseline outcomes was achieved.",
    "Stratified randomization by gender and baseline test score tercile.",
    "",
]
MDES = ["0.2 SD", "0.15 standard deviations", "5 percentage points", "10%",
        "0.1", "", "a 3 percentage point increase in take-up", "0.25 SD"]
DATE_FORMATS = ["%Y-%m-%d", "%B %d, %Y", "%m/%d/%Y"]


def d(x: date, style: int = 0) -> str:
    return x.strftime(DATE_FORMATS[style % len(DATE_FORMATS)])


def make_row(i: int, rng: random.Random) -> dict:
    topic, intervention, outcomes, jel = rng.choice(TOPICS)
    design, unit = rng.choice(DESIGNS)
    n = 4000 + i
    reg = date(2013, 1, 1) + timedelta(days=int(rng.random() * 4400))
    retro = rng.random() < 0.18
    istart = reg + timedelta(days=int(rng.random() * 400) * (-1 if retro else 1))
    iend = istart + timedelta(days=60 + int(rng.random() * 900))
    completed = iend < date(2026, 1, 1) and rng.random() < 0.6
    status = rng.choice(["completed"] * 3 + ["on_going"] * 4 +
                        ["in_development"] * 2 + ["withdrawn"])
    if completed:
        status = "completed"

    n_obs = rng.choice([120, 400, 900, 1500, 2400, 6000, 14000, 60, 250000])
    n_clusters = rng.choice(["", "40", "120", "8", "300"])
    n_arms = rng.choice(["2", "2", "3", "4", "6"])

    pis = rng.sample(NAMES, k=rng.randint(1, 4))
    countries = rng.sample(COUNTRIES, k=rng.randint(1, 3))

    papers = ""
    post = ""
    if status == "completed" and rng.random() < 0.35:
        if rng.random() < 0.6:
            papers = (f'"{topic.title()} and household welfare: evidence from '
                      f'{countries[0]}", American Economic Review, 2023. '
                      f"https://doi.org/10.1257/aer.2021{i:04d}")
        else:
            papers = (f"Working paper: The effect of {topic} in {countries[0]}. "
                      f"NBER Working Paper w{28000 + i}")
        if rng.random() < 0.3:
            post = f"Pre-analysis plan addendum | Final report {reg.year + 2}"

    abstract = (
        f"We study the effect of {intervention} in {', '.join(countries)}. "
        f"The study is a randomized controlled trial with {n_obs} participants. "
        f"{design} We measure {outcomes}. "
        + ("We power the study to detect a 0.2 SD change using an intra-cluster "
           "correlation of 0.05. " if rng.random() < 0.5 else "")
        + ("Multiple hypothesis testing is handled with Romano-Wolf stepdown "
           "corrections across a pre-specified family of outcomes. "
           if rng.random() < 0.35 else "")
        + ("We link survey responses to administrative records held by the "
           "revenue authority. " if rng.random() < 0.2 else "")
    )

    return {
        "Title": f"{topic.title()} in {countries[0]}: a randomized evaluation ({reg.year})",
        "Url": f"https://www.socialscienceregistry.org/trials/{n}",
        "Last update date": d(reg + timedelta(days=30), i),
        "Published at": d(reg, i + 1),
        "First registered on": d(reg, i),
        "RCT_ID": f"AEARCTR-{n:07d}",
        "DOI Number": f"10.1257/rct.{n}-1.0",
        "Primary Investigator": pis[0],
        "Status": status,
        "Start date": d(istart, i),
        "End date": d(iend, i),
        "Keywords": rng.choice(["", f"{topic}; field experiment; development",
                                f"{topic}, RCT, policy"]),
        "Country names": ", ".join(countries),
        "Other Primary Investigators": "; ".join(pis[1:]),
        "Jel code": jel,
        "Secondary IDs": "",
        "Abstract": abstract,
        "External Links": rng.choice(["", f"https://example.org/study/{n}"]),
        "Sponsors": rng.choice(["", "J-PAL", "World Bank; DFID", "NSF"]),
        "Partners": rng.choice(["", "Ministry of Education", "Local NGO, District Office"]),
        "Intervention start date": d(istart, i),
        "Intervention end date": d(iend, i),
        "Intervention": intervention.capitalize() + ".",
        "Primary outcome end points": outcomes,
        "Primary outcome explanation": rng.choice([
            "", f"The primary outcome is an index of {outcomes}, standardised "
                "against the control group and estimated by OLS with strata "
                "fixed effects and baseline controls."]),
        "Secondary outcome end points": rng.choice(["", "intra-household bargaining, migration"]),
        "Secondary outcome explanation": "",
        "Experimental design": design,
        "Experimental design details": rng.choice(["", design + " " + rng.choice(RANDOM_METHODS)]),
        "Randomization method": rng.choice(RANDOM_METHODS),
        "Randomization unit": unit,
        "Sample size number clusters": n_clusters,
        "Sample size number observations": str(n_obs),
        "Sample size number arms": n_arms,
        "Minimum effect size": rng.choice(MDES),
        "IRB": rng.choice(["", "IRB approval obtained, protocol 2019-0421"]),
        "Analysis Plan Documents": rng.choice(["", "", "analysis_plan.pdf"]),
        "Intervention completion date": d(iend, i) if completed else "",
        "Data collection completion": "Yes" if completed else "No",
        "Data collection completion date": d(iend + timedelta(days=90), i) if completed else "",
        "Number of clusters": n_clusters if completed else "",
        "Attrition correlated": "",
        "Total number of observations": str(n_obs) if completed else "",
        "Treatment arms": rng.choice(["", "Control; Information only; Information plus subsidy"]),
        "Public data": "Yes" if completed and rng.random() < 0.3 else "No",
        "Public data url": "",
        "Program files": "No",
        "Program files url": "",
        "Post trial documents csv": post,
        "Relevant papers for csv": papers,
    }


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures/registry_sample.csv")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rng = random.Random(20260904)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        for i in range(count):
            w.writerow(make_row(i, rng))
    print(f"wrote {out} with {count} rows")


if __name__ == "__main__":
    main()
