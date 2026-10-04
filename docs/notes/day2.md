# Day 2: remaining non-Bayesian models and robust fitting

## Status

- Asymmetric RL, the random benchmark and the perseveration benchmark are
  implemented in JAX (`lax.scan`) with NumPy references. (All six models were
  written together on Day 1 because the reference agents share one interface;
  the Day 2 tests are the ones that validate the non-Bayesian extensions.)
- Tests: learning-rate branching (including δ = 0 → α⁺), symmetric
  perseveration under 0↔1 relabelling, JAX vs reference ≤ 1e-8 for every model
  on masked data, `jax.grad` vs central finite differences.
- Multi-start fitting with the locked rules (N(0, 1.5²) starts, gradient-norm
  convergence, boundary flags, `near_random` flag, proportion of starts within
  0.1 of the best). Every fit's JSON stores all starts.
- Smoke-test step 5 (fixed vs asymmetric RL on each 50-trial dataset, scored by
  held-out LL) runs; see the caveat below.
- Small parameter recovery (fixed + asymmetric RL, 160 trials, 3 conditions,
  50 agents): 300 fits in 18 s wall-clock on 21 workers.
- Timing and projection written to `docs/compute_budget.md`: ≈ 3–10 ms per
  optimisation; the full design projects to ≈ 10 single-core minutes, so the
  stop rule is not triggered and final figures will use 100 agents per cell.

## Findings worth keeping

**Runaway β is a real identifiability limit.** 12/300 agents in the small run
had β̂ > 20 (up to 3 × 10⁵). They fall in two patterns: (i) α̂ → 0 with β̂ → ∞,
almost all near-chance agents (true β < 1, accuracy ≈ 0.5), where Q is a scaled
running sum of prediction errors and only αβ is identified (the profile
likelihood changes by 1.5 nats between β = 2 and β = 10⁵); (ii) α̂ → 1 with
β̂ → ∞, i.e. near-deterministic win-stay/lose-shift. These agents are flagged,
not excluded. Raw-scale Pearson r and mean bias for β are therefore dominated
by a handful of agents; `parameter_recovery_metrics` also reports log-scale β
metrics, median-based errors, and metrics for agents with true β ≥ 2.

**Fixed RL matches the expected benchmark.** 160 trials, moderate, true β ≥ 2:
r(α) = 0.92, r(log β) = 0.91 (n = 29). Across all β the correlations are lower
(0.66, 0.87) because low-β agents carry little information, as expected.

**Held-out scores can be catastrophic when training sets are tiny.** In the
50-trial smoke test one fold trains on only 10 trials; β runs away and the
held-out log likelihood reaches −10⁶ per trial. With 160 trials each fold trains
on 120 trials, but the same mechanism can occasionally fire, so model-recovery
summaries will report the β-runaway rate per fold alongside the scores.

**Asymmetric RL.** In this small run α⁻ was recovered at least as well as α⁺
(high-β r = 0.85 vs 0.71 in the moderate condition), which is the opposite of the
"Expected results" note. Revisit with the full 100-agent run before interpreting.

## Writing: Methods — Fitting procedure (draft)

Each simulated agent was fitted independently by maximum likelihood. Parameters
were optimised on an unconstrained scale and transformed: bounded parameters
(α, α⁺, α⁻, η, κ, h) through the logistic function, β = softplus(θ_β) + 10⁻⁴,
ρ = 5 tanh(θ_ρ); the perseveration bias b is unbounded. The negative log
likelihood was computed with `jax.lax.scan` in float64, using stable
log-sigmoid choice probabilities, and JIT-compiled; its exact gradient
(`jax.value_and_grad`) was passed to SciPy's L-BFGS-B (ftol 10⁻¹², gtol 10⁻⁶,
≤ 2000 iterations). Each fit used multiple starts drawn independently from
N(0, 1.5²) on the unconstrained scale (20 starts for single fits, 10 inside
recovery loops). A start was deemed *converged* if its loss and parameters were
finite and the Euclidean norm of the gradient was below 10⁻³; the converged
start with the lowest loss was selected (the lowest finite start if none
converged, recorded as unconverged), and a fit *failed* only if no start was
finite. We recorded, for every start, the loss, parameters, gradient norm,
optimiser status, iterations and warnings, and the proportion of starts within
0.1 nats of the best as an index of likelihood ruggedness. Estimates were flagged
as at a boundary if a (0, 1) parameter was below 0.01 or above 0.99, if β > 20
(runaway), or if |ρ| > 4.9. Fits whose improvement over the random-choice
benchmark (n log 2 − NLL) was smaller than the number of free parameters were
flagged `near_random`. Flagged fits were never excluded.

## Comprehension checks

**Q1. Why can α and β trade off, and which plots reveal it?** Choice
probabilities depend on β(Q₁ − Q₀), and the size of the value difference is
itself set by α: small α keeps Q near 0.5 so differences are small, which a
larger β can compensate, while large α produces large, volatile differences
that a smaller β damps. Different (α, β) pairs therefore predict similar choice
sequences, especially with few trials or low information. It shows up as a
negative correlation between recovered α and β across agents (pairwise
scatterplot / correlation matrix of estimates), as curved or banded
true-vs-recovered scatter, and in the extreme as the α → 0, β → ∞ runaway seen
above.

**Q9. Why is I(a_{t−1} = 1) not a valid perseveration term, and what does
s_{t−1} fix?** The indicator adds ρ to the logit only when the previous choice
was option 1; after option 0 it adds nothing. So it pushes towards option 1 after
choosing option 1 but never towards repeating option 0: it confounds
perseveration with a history-dependent bias towards one option label and is not
invariant to relabelling the options. The symmetric code s_{t−1} ∈ {−1, +1}
adds +ρ after option 1 and −ρ after option 0, i.e. it always pushes towards
repeating whatever was chosen; relabelling options mirrors the choice
probabilities exactly (unit-tested).
