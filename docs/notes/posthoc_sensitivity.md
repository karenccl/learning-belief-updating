# Post-hoc additions (2026-09-29, after the six-day plan)

Two review points, both addressed *after* seeing the locked results. Neither
changes the locked numbers; both are labelled post-hoc in the report.

## 1. The Bayesian ĥ ≈ 0.5 fits: which case are they?

Expanding the transition gives b_t = 0.5 + (1 − 2h)(b′_{t−1} − 0.5), so:

- **h = 0.5 exactly:** b_t ≡ 0.5, Q₀ = Q₁, pure random choice (not WSLS).
- **h → 0.5 with β → ∞:** the belief carries only the last trial's evidence
  scaled by ε = 1 − 2h. After (a, r) from b ≈ 0.5, b′ ≈ 0.8 or 0.2, so
  Q₁ − Q₀ ≈ ∓0.36ε and the logit is a graded win-stay/lose-shift of strength
  ≈ 0.36·β(1 − 2h). Only the product β(1 − 2h) is identified.

Check on the 900 locked Bayesian parameter-recovery fits (`bayes_h_ridge.csv`):

| Regime | n | median ĥ | median β̂ | median β(1 − 2ĥ) | median gain vs random (nats) | median true β |
|---|---|---|---|---|---|---|
| exactly h = 0.5 (random) | 0 | — | — | — | — | — |
| WSLS ridge, \|ĥ − 0.5\| < 0.02 | 108 | 0.4998 | 1,847 | 0.88 | 1.9 | 1.04 |
| alternation, ĥ > 0.52 | 37 | 0.991 | 0.62 | −0.58 | 0.77 | 0.75 |
| interior | 755 | 0.102 | 2.73 | 1.74 | 11.4 | 2.52 |

All ridge fits are on the graded-WSLS side: none sits at 0.5, all beat random,
and all 10 starts agree. On one agent (stable, 320 trials), holding
β(1 − 2h) = 2.99 fixed, the NLL falls monotonically as h → 0.5 (ε = 0.2: 182.60;
ε = 10⁻⁴: 181.44) and jumps to the random value (221.81) at h = 0.5. The
supremum is not attained, which is why the optimiser drifts along the ridge
until the gradient falls below tolerance. (My first description, "h → 0.5, β → ∞
reduces to one-trial-back WSLS", was right about the limit but should have said
that the product is what is identified and that h = 0.5 itself is random.
The earlier "60 of 450" count came from the 50-agent first pass.)

## 2. Sensitivity analysis: bounded β and h

Refit of all parameter recovery (3,600), model recovery (1,200 × 6) and demo CV
(400 × 6) with β = 10⁻⁴ + 20·logistic(θ) and h = 0.5·logistic(θ); same agents,
folds and starts (`scripts/08_sensitivity_bounded.py`, 798 s).

- Catastrophic held-out scores (< −2 LL/trial) from learning models: 114 → 0 in
  model recovery (asymmetric RL 106 → 0, fixed 6 → 0, hybrid 2 → 0); worst score
  −8.1 × 10⁵ → −1.67. The perseveration benchmark keeps its 11 (its bias is
  unbounded; not changed).
- Demo-dataset selection (mean held-out LL): fixed-RL data, hybrid → **fixed
  RL** (now matches the median); asymmetric-RL data, hybrid → fixed RL; hybrid
  data, fixed RL (unchanged); Bayesian data, Bayesian CP (unchanged).
- Model recovery: tie rate 0.716 → 0.696; accuracy excluding ties 0.53 → 0.51;
  raw winner correct 0.378 → 0.373. Bayesian 0.56 → 0.59; asymmetric 0.18 → 0.19;
  hybrid 0.35 → 0.32; fixed 0.42 → 0.39. **Conclusions unchanged.**
- Parameter recovery: essentially unchanged, except Bayesian ĥ in the moderate
  (0.37 → 0.27) and volatile (0.66 → 0.54) conditions. That is a ranking
  artefact: degenerate-regime agents have higher true h (0.20 vs 0.145), and
  their spurious locked ĥ ≈ 0.5–1 ranked them high by accident. Within the
  interior regime, recovery is 0.49 → 0.52. Bounded ridge fits sit at β = 20
  with ĥ = 0.479, i.e. β(1 − 2ĥ) ≈ 0.84 ≈ the locked 0.88: the product is
  preserved.

**Take-home:** unregularised ML on short sequences systematically produces a
few near-deterministic fits. Bounds remove the resulting scoring pathology but
do not create identifiability that the design lacks.
