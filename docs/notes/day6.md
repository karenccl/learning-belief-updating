# Day 6: integrate and communicate

## Status

- **Clean-clone check.** Cloned the repository into a fresh directory and ran
  `pytest` (76 passed), `01_smoke_test.py`, `01b_latent_examples.py` and
  `05_make_figures.py`. The smoke-test and latent tables are byte-identical to
  the originals (apart from line endings), and all 18 figures regenerate from the
  committed tables. I also re-ran 6 parameter-recovery agents and 2
  model-recovery agents (12 CV files) in the clone: all 18 fit files match the
  originals exactly (estimates, held-out scores, true parameters).
- README finalised (installation, configuration, run order, timing, layout).
- Report assembled from the daily notes: `report/report.md` (abstract, CV line,
  Tables 1–5, Figures 1–6).
- No new analyses were added today (stop rule).

## Rereading the ten comprehension questions

All ten are answered in the daily notes (Q2 day 1; Q1, Q9 day 2; Q4, Q5, Q10
day 3; Q1 revisited, Q3 day 4; Q6, Q7, Q8 day 5) and each maps onto a report
section: Q1, Q3 → Results (recovery); Q4, Q5 → Methods (CV); Q6, Q8 → Results
(predictive checks); Q7 → Discussion; Q9 → Methods (choice rule); Q10 →
Methods (Bayesian learner); Q2 → Introduction. None exposed a gap.

## Deliverables checklist

- [x] Reproducible repository with environment specification (`environment.yml`, `pyproject.toml`).
- [x] Task and formal model specification (report Table 1; `results/tables/model_specification.csv`).
- [x] Unit tests and minimal end-to-end smoke test.
- [x] Simulation pipeline with saved seeds and configuration.
- [x] Multi-start fitting pipeline with diagnostic outputs.
- [x] Parameter-recovery analyses and figures.
- [x] Model-recovery analyses and confusion matrix (with tie column).
- [x] Sequential held-out predictive comparison.
- [x] Parametric predictive checks of post-reversal adaptation.
- [x] Optional real-data analysis: omitted (feasibility criteria not met on Day 1).
- [x] README with installation, configuration, and reproduction commands.
- [x] `docs/design_decisions.md`, `docs/compute_budget.md`, daily notes.
- [x] Report, abstract, and CV line.
- [ ] Push to GitHub: no remote is configured; the repository is local only.
