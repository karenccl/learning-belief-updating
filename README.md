# Learning from feedback and belief updating under volatility

Simulation-based validation of computational learning models for a two-armed
reversal-learning task: parameter recovery, model recovery, held-out predictive
comparison, and parametric predictive checks.

**Report:** [`report/report.md`](report/report.md). **Design decisions:**
[`docs/design_decisions.md`](docs/design_decisions.md). **Compute budget:**
[`docs/compute_budget.md`](docs/compute_budget.md). **Daily notes:** `docs/notes/`.

## Aims

1. A reproducible pipeline for simulating, fitting, comparing, and checking candidate learning models.
2. Determine whether model parameters can be recovered under realistic task conditions.
3. Determine whether candidate models are behaviourally distinguishable.
4. Compare models using out-of-sample sequential predictive log likelihood.
5. Test whether fitted models reproduce behavioural summaries, especially adaptation after reversals.

## Models

| Model | Free parameters |
|---|---|
| Random choice (benchmark) | — |
| Perseveration (benchmark) | bias b, ρ |
| Fixed-learning-rate RL | α, β |
| Valence-asymmetric RL | α⁺, α⁻, β |
| Pearce–Hall hybrid (α_t = κA_t) | η, κ, β |
| Bayesian change-point learner | h, β |

Task defaults: two options (0/1), binary reward with p = 0.80 / 0.20, option 0
initially advantageous, 160 trials. Stable: one reversal after trial 120;
moderate: a reversal every 40 trials; volatile: a reversal every 20 trials.

## Headline results

- Fixed RL is well identified (r > 0.9 for agents with β ≥ 2). Hybrid η is
  essentially unrecoverable, and α⁺/α⁻ are recovered only moderately.
- At 160 trials, 72% of simulated agents are practical ties between models
  (margin < 0.01 LL/trial). Among non-tied agents, selection accuracy is 0.53; ignoring
  the tie rule, the top-scoring model is the generating model for 0.38 of all agents (chance 0.25).
- The Bayesian change-point learner is the most distinguishable model;
  asymmetric RL is almost never recovered, even on its own data.
- Unregularised ML produces rare runaway fits that dominate held-out *means*;
  predictive checks catch a lose-switch misfit that the held-out score missed.
- Post-hoc sensitivity analysis (bounded β ≤ 20, h < 0.5): the catastrophic
  scores disappear and fragile model selections are corrected, but recovery and
  identifiability are essentially unchanged ([notes](docs/notes/posthoc_sensitivity.md)).

## Installation

Requires conda (Anaconda or Miniconda).

```bash
conda env create -f environment.yml
```

```bash
conda activate rl-reversal
```

Note for Windows: the environment pins OpenBLAS (`libblas=*=*openblas`). With
the MKL 2026 build, SciPy's L-BFGS-B crashed the interpreter (`0xc06d007f`).
Install SciPy from conda-forge and do not let pip replace it.

## Configuration

Everything is set in [`config/default_config.yaml`](config/default_config.yaml):
the master seed (20261001), reward probabilities, condition schedules, CV block
length, number of starts, agents per cell, parameter ranges, and `compute.n_jobs`
(joblib workers; −2 = all cores but one). Every agent's seed is derived from
the master seed and its (stage, model, condition, trial count, agent) index, so
any subset of the grid can be re-run and reproduces the same fits.

## Reproducing the results

Run from the repository root, in this order. Fits are written one JSON per
(agent, fitted model) under `results/fits/` (not committed); existing files are
skipped, so an interrupted run resumes where it stopped.

```bash
python -m pytest
```

```bash
python scripts/00_time_fits.py
```

```bash
python scripts/01_smoke_test.py
```

```bash
python scripts/01b_latent_examples.py
```

```bash
python scripts/02_parameter_recovery.py --agents 100
```

```bash
python scripts/03_model_recovery.py --agents 100
```

```bash
python scripts/06_predictive_checks.py
```

```bash
python scripts/08_sensitivity_bounded.py
```

```bash
python scripts/07_collect_tables.py
```

```bash
python scripts/05_make_figures.py
```

| Script | Does | Time (21 workers) |
|---|---|---|
| `00_time_fits.py` | times fits per model → `results/tables/timing.csv` | ~1 min |
| `01_smoke_test.py` | 10 agents × 50 trials, fixed RL, end to end | ~20 s |
| `01b_latent_examples.py` | task-schedule table + latent trajectories | ~5 s |
| `02_parameter_recovery.py` | 4 models × 3 trial counts × 3 conditions × N agents | ~2.5 min |
| `03_model_recovery.py` | 4 generating × 3 conditions × N agents, 6 models, masked CV | ~10 min |
| `06_predictive_checks.py` | demo datasets, CV, predictive checks, forward chaining | ~8.5 min |
| `08_sensitivity_bounded.py` | **post-hoc** refit with β ≤ 20 and h < 0.5; comparison tables | ~13 min |
| `07_collect_tables.py` | fits → `results/tables/*.csv` | ~1 min |
| `05_make_figures.py` | tables → `results/figures/*.png` | ~30 s |
| `04_optional_real_data.py` | records that the real-data extension was omitted | — |

Figures are generated only from the committed CSV tables, so
`python scripts/05_make_figures.py` works in a fresh clone without re-running
any fits.

## Repository layout

```text
config/default_config.yaml   all settings, master seed
docs/                        design decisions, compute budget, daily notes (day1-6)
src/task.py                  environment and reversal schedules
src/models.py                transforms, NumPy reference agents, JAX lax.scan likelihoods
src/simulate.py              trial-level simulation
src/fit.py                   multi-start L-BFGS-B fitting with diagnostics and flags
src/cross_validation.py      masked leave-block-out CV, forward chaining
src/recovery.py              recovery jobs, collection, metrics, confusion matrices
src/predictive_checks.py     behavioural summaries and parametric predictive checks
src/seeds.py                 deterministic per-agent SeedSequence children
src/plots.py                 all figures (from tables only)
tests/                       84 unit tests (task, models, likelihood, fitting, CV, checks)
scripts/                     numbered pipeline scripts (above)
results/tables, figures      committed outputs; results/fits is git-ignored
report/report.md             2-4 page report, abstract, CV line
notebooks/                   learning exercises, not part of the report
                             (map_exercise_fixed_rl.ipynb: post-hoc MAP vs ML vs bounded, fixed RL)
```
