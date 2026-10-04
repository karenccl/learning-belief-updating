# Design decisions (locked, v3)

These decisions were fixed before Day 1. Changing any of them mid-project
requires a dated note in the *Change log* at the bottom of this file.

| Decision | Locked choice |
|---|---|
| Option coding | Options are coded 0 and 1 everywhere, in equations and in code. |
| Volatility conditions | Stable: exactly one reversal, after 75% of trials. Moderate: reversal every 40 trials. Volatile: reversal every 20 trials. |
| Perseveration coding | Symmetric: ρ·s_{t-1} with s_{t-1} ∈ {−1,+1} and s_0 = 0. Primary models fix ρ = 0; only the perseveration benchmark estimates it. |
| Adaptive model | Pearce–Hall-style hybrid: α_t = κ·A_t, free parameters η, κ, β. |
| Cross-validation | Leave-block-out with a **masked likelihood**: latent states run through the full sequence; only held-out trials are scored. Blocks are fixed 40-trial segments. |
| Model-comparison score | Held-out sequential log likelihood per trial, used identically in model recovery and in any real-data analysis. |
| Model-recovery grid | 160 trials only × 3 volatility conditions. Trial count is varied in parameter recovery only. |
| Multi-start | 20 starts for single fits and smoke tests; 10 starts inside recovery loops. |
| Near-random fits | Flagged (`near_random = True`), never excluded. |
| Compute | Likelihoods are JIT-compiled with `jax.lax.scan` and differentiated with `jax.grad` by the end of Day 2. |
| Unit of recovery | One **simulated agent** = one independently fitted participant. Recovery cells contain 50 agents (first pass) or 100 (final figures). The 100-agent "demonstration dataset" is used only for predictive checks. |
| Parameter sampling | Uniform over the ranges in the parameter-range table, except β, which is log-uniform on [0.5, 8]. |
| Starting values | Unconstrained starts drawn from N(0, 1.5²) per parameter. |
| Convergence | A start has *converged* if its loss and parameters are finite and its gradient norm is below 1e-3; the scipy status is recorded but not used as the criterion. |
| Boundary flag | Parameters bounded in (0,1): estimate below 0.01 or above 0.99. β: estimate above 20 (runaway). ρ: \|ρ\| > 4.9. |
| Ties in model recovery | If the winner beats the runner-up by less than 0.01 log-likelihood units per trial, the agent is counted in a separate "tie" column. |
| Seeds | One master seed in the config; per-agent seeds from `np.random.SeedSequence(master_seed).spawn(n)`, indexed deterministically by (model, condition, trial count, agent). |

## Parameter ranges for simulation

| Parameter | Range | Sampling |
|---|---|---|
| α (fixed RL) | 0.05–0.80 | uniform |
| α⁺, α⁻ (asymmetric RL) | 0.05–0.80 | uniform |
| β (all primary models) | 0.5–8.0 | log-uniform |
| η (hybrid) | 0.05–0.80 | uniform |
| κ (hybrid) | 0.10–0.90 | uniform |
| h (Bayesian) | 0.01–0.30 | uniform |

## Implementation notes (interpretations of the spec, not changes)

- **Seed indexing.** `SeedSequence(master_seed).spawn(N)` is called once with a
  fixed N large enough for the whole grid; the child used for a given agent is
  selected by a deterministic integer index computed from
  (stage, model, condition, trial count, agent). Stage is included so that the
  parameter-recovery, model-recovery and demonstration datasets do not share
  random streams. Each child seed drives parameter sampling, simulation, and
  the optimiser's starting values for that agent.
- **Convergence fallback.** If no start converges (gradient norm ≥ 1e-3 for all
  starts), the finite start with the lowest loss is selected and the fit is
  recorded with `converged = False`. The fit is only `failed` if *no* start is
  finite.
- **near_random.** Every learning model contains near-random behaviour as
  β → 0, so its *training* NLL can never be worse than the random benchmark's
  (n · log 2); a literal "does not improve" test would never fire. The flag is
  therefore defined with an AIC penalty: `near_random = True` when
  n_scored · log 2 − NLL < k (the improvement over random is smaller than the
  number of free parameters, in nats).
- **Perseveration benchmark** has two free parameters, bias b (unbounded,
  identity transform) and ρ = 5·tanh(θ_ρ). On trial 1 the model uses
  P(a=1) = 0.5 exactly (bias is not applied), as specified.
- **Tie rule** compares the top two *primary* models' held-out scores per
  trial. Benchmarks are scored and reported but are not candidates in the
  confusion matrix.

## Change log

_(Dated entries recording any deviation from the locked choices.)_

- **2026-09-29 — post-hoc sensitivity analysis (added after seeing the locked
  results; the locked analysis remains primary and is unchanged).** The locked
  results showed that unregularised ML occasionally runs β → ∞ along degenerate
  ridges (RL: α → 0/1; asymmetric RL: α⁺ → 0; Bayesian: h → 0.5). A single such
  fold can reach a held-out LL of −10⁶ per trial and decide the mean-score model
  comparison. Locked Bayesian fits also include an "expect alternation" regime
  (ĥ > 0.5), which is outside the model's intended use. We therefore refit
  everything with a *bounded* transform variant:
  β = 10⁻⁴ + (20 − 10⁻⁴)·logistic(θ_β) for all models, and h = 0.5·logistic(θ_h)
  for the Bayesian learner. All other rules are unchanged. The boundary flag
  becomes β > 19.8 (within 1% of the cap) and h < 0.01 or h > 0.495. The runs
  use the same simulated agents, folds and starting values
  (`scripts/08_sensitivity_bounded.py`; fits in `results/fits/*_bounded/`;
  tables `*_bounded.csv` and `sensitivity_*.csv`). The question is only whether
  the conclusions change. Results are reported in a separately labelled report
  section and never replace the locked numbers.
