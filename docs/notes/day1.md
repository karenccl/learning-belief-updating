# Day 1: environment, simulation, and a first fit

## Status

- `make_environment` / `reversal_schedule` implemented for all three conditions and
  trial counts (80/160/320, plus 50 for the smoke test); tests check the schedule
  table in the spec and that probabilities swap exactly at scheduled reversals.
- Fixed-RL simulation (plain NumPy agent) and fixed-RL likelihood in JAX
  (`lax.scan`, `jit`, `value_and_grad`) with a plain-Python reference.
- Hand-calculated three-trial likelihood is the first unit test (below).
- Smoke test steps 1–4 and 6 pass: 10 agents × 50 trials, α = 0.30, β = 4.0;
  all fits finite and converged; median recovered α = 0.296, β = 4.48.
  Figure: `results/figures/smoke_true_vs_recovered.png`.
- Smoke-test starts: the smoke-test section of the spec says "10 multi-start
  initialisations" but the locked table says 20 for smoke tests; the locked
  table wins (20 starts).

### Environment problem found and fixed

The first test run crashed the interpreter inside `scipy.optimize.minimize`
(Windows fatal exception `0xc06d007f`, a missing DLL entry point). Cause: the
conda-forge scipy build linked against the MKL 2026 BLAS; even a two-line
L-BFGS-B call on a quadratic crashed. Switching the environment to OpenBLAS
(`libblas=*=*openblas`) fixed it; `environment.yml` pins this.

## Hand-calculated three-trial likelihood

α = 0.3, β = 4, choices 0, 0, 1, rewards 1, 0, 1, ρ = 0, Q₀ = Q₁ = 0.5.

| t | Q₀ | Q₁ | logit β(Q₁−Q₀) | P(observed choice) | log P | update |
|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.5 | 0 | P(a=0) = 0.5 | −0.693147 | δ = 0.5 → Q₀ = 0.65 |
| 2 | 0.650 | 0.5 | −0.60 | P(a=0) = σ(0.6) = 0.645656 | −0.437488 | δ = −0.65 → Q₀ = 0.455 |
| 3 | 0.455 | 0.5 | 0.18 | P(a=1) = σ(0.18) = 0.544879 | −0.607192 | — |

NLL = 1.737827 (my first pencil pass gave 1.737831; the last digits were lost
to rounding σ(0.18). The unit test uses the exact value and the JAX and
reference implementations both agree with it to 1e-10).

## Real-data feasibility checklist (decision: omit)

No reversal-learning dataset was confirmed by the end of Day 1 as satisfying all
eight criteria (licence, trial-level choices and outcomes, interpretable
contingencies, participant IDs, reconstructable reversal schedule, enough trials,
task match). Per the scope rules, the real-data extension is omitted and the
project is **simulation-based by design**. `scripts/04_optional_real_data.py`
records this decision and exits.

## Writing: Methods — Task (draft)

Simulated participants complete a two-armed probabilistic bandit with
unannounced contingency reversals. On each trial t the agent chooses
a_t ∈ {0, 1} and receives a binary reward r_t ~ Bernoulli(p_{a_t,t}). The
currently advantageous option pays with probability 0.80 and the other with
0.20; option 0 is advantageous at the start. At each scheduled reversal the two
probabilities swap. Three volatility conditions are used: *stable* (a single
reversal after 75% of trials), *moderate* (a reversal every 40 trials) and
*volatile* (a reversal every 20 trials). The default task has 160 trials
(1, 3 and 7 reversals respectively); parameter recovery additionally uses 80 and
320 trials, with the stable reversal after trials 60 and 240. Because the stable
condition always has exactly one reversal, its effective hazard rate differs
across trial counts (1/80 vs 1/320). Only the chosen option's outcome is shown,
and outcomes are delivered immediately.

## Writing: Methods — Models, fixed RL (draft)

All models share the choice rule
P(a_t = 1) = logistic(β[Q_{1,t} − Q_{0,t}] + ρ s_{t−1}),
with s_{t−1} = +1 if the previous choice was option 1, −1 if it was option 0,
and s₀ = 0; ρ = 0 in all four primary models. In fixed-learning-rate
Rescorla–Wagner RL, after choosing a_t the prediction error
δ_t = r_t − Q_{a_t,t} updates only the chosen value:
Q_{a_t,t+1} = Q_{a_t,t} + α δ_t, with Q_{0,1} = Q_{1,1} = 0.5. Free parameters are
α ∈ (0, 1) and β > 0, estimated as α = logistic(θ_α), β = softplus(θ_β) + 10⁻⁴.

## Comprehension check Q2

*Why must data be simulated before parameter and model recovery can be assessed?*
Recovery asks whether the fitting procedure returns the parameters (or model)
that produced the data, which is only answerable when the ground truth is known.
With real data the true parameters and mechanism are unknown, so a good fit
cannot distinguish "correct inference" from "a flexible model absorbing noise".
Simulating from known models under the exact task design also tells us, before
looking at real behaviour, which parameters and model distinctions that design
can support at all (Wilson & Collins, 2019, rules 5–6 in spirit: simulate
before you fit).
