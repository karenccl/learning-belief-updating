# Day 4: recovery analyses

## Status

- Parameter recovery: 4 models × {80, 160, 320} trials × 3 conditions ×
  **100 agents** (3,600 fits, 10 starts each). First pass at 50 agents, then
  extended to 100 (agents 0–49 identical by seed construction).
- Model recovery: 4 generating models × 3 conditions × **100 agents** at 160
  trials; all 6 models fitted with masked leave-block-out CV (4 × 40-trial folds,
  10 starts per fold): 1,200 agents, 28,800 fold fits.
- No failed fits and no unconverged selected fits anywhere (every selected start
  had gradient norm < 1e-3).
- Tables: `parameter_recovery_metrics.csv`, `parameter_recovery_agents.csv`,
  `recovered_parameter_correlations.csv`, `near_random_by_beta.csv`,
  `model_recovery_{scores,winners,confusion,summary,pairwise_confusability,
  heldout_scores,score_differences}.csv`, `optimisation_diagnostics.csv`.
- Figures: `fig2_true_vs_recovered_{80,160,320}.png`,
  `fig3_recovery_vs_trial_count.png`, `bias_mae_by_condition.png`,
  `parameter_tradeoffs.png`, `near_random_by_beta.png`,
  `fig4_confusion_matrix.png`, `margin_distribution.png`,
  `fig5_heldout_comparison.png`, `fig5b_paired_differences.png`.

## Stop-rule check (best-case cell: 320 trials, true β ≥ 2)

Pearson r between true and recovered (log scale for β), by condition
(moderate / stable / volatile):

| Model | Parameter | r |
|---|---|---|
| Fixed RL | α | 0.94 / 0.95 / 0.97 |
| Fixed RL | β | 0.89 / 0.83 / 0.90 |
| Asymmetric RL | α⁺ | 0.69 / 0.77 / 0.82 |
| Asymmetric RL | α⁻ | 0.89 / 0.94 / 0.84 |
| PH hybrid | κ | 0.84 / 0.87 / 0.86 |
| PH hybrid | η | 0.23 / 0.47 / 0.37 |
| Bayesian CP | h | 0.68 / 0.73 / 0.80 |
| Bayesian CP | β | −0.05 / 0.06 / 0.75 |

Two cells are weak: hybrid η (the spec anticipates this) and Bayesian β outside
the volatile condition. For the Bayesian learner I ran a dedicated check: with
2,000 volatile trials (40 agents, h ∈ [0.05, 0.25], β ∈ [2, 8]) recovery is
r(h) = 0.94, r(log β) = 0.98, so simulation and likelihood agree. The weak cells
are identifiability limits, not bugs. Both parameters set the plateau of choice
determinism; with only 1–3 reversals, h and β trade off (Spearman correlation of
estimates ĥ vs log β̂ = +0.53 at 160 trials).

## Results notes: parameter recovery (Spearman, all agents)

- **Fixed RL** is the best-identified model. At 160 trials, ρ(α) = 0.62–0.83 and
  ρ(β) = 0.79–0.94; with true β ≥ 2, r(α) > 0.9, matching the expected ≥ 0.8.
  Recovery improves with trial count in every condition. Surprisingly, α is
  recovered best in the *stable* condition at 160 trials (0.83 vs 0.62–0.66), and
  β is recovered worst in the volatile condition.
- **Asymmetric RL**: ρ ≈ 0.5–0.6 at 160 trials, rising to 0.6–0.75 at 320. In
  the high-β subset, α⁻ is recovered *better* than α⁺ (0.84–0.94 vs 0.69–0.82
  at 320 trials), the reverse of the spec's expectation. A plausible reason:
  most agents choose the better option, so positive prediction errors are
  frequent but small (r = 1 when Q is already high), whereas negative prediction
  errors, though rarer, are large and followed by conspicuous switches. α⁺ is
  also the parameter most often at a boundary (14–24% at 160 trials).
- **PH hybrid**: κ is moderately recovered (ρ 0.51–0.55 at 160; 0.66–0.73 at
  320), η barely at all (ρ ≤ 0.36; 30–39% of η̂ at a boundary). η is best
  recovered in the volatile condition, where surprise varies most. Against the
  expectation, it is worst in the *moderate*, not the stable, condition. β is
  well recovered (0.76–0.95) because it is largely determined by choice
  consistency.
- **Bayesian CP**: h is best recovered in the volatile condition (ρ = 0.66 at
  160), as expected. Many near-chance agents sit on a degenerate ridge at
  ĥ ≈ 0.5 with β̂ → ∞. As h → 0.5 the belief keeps only the last trial's
  evidence scaled by (1 − 2h), so the model becomes a graded win-stay/lose-shift
  rule, the Bayesian counterpart of the RL α → 0, β → ∞ degeneracy.
- **Trade-offs** (Spearman between estimates, 160 trials): asymmetric RL
  α⁺–β = −0.43, hybrid κ–β = −0.41, Bayesian h–log β = +0.53; fixed RL α–β is
  small overall (−0.10) but has the runaway corner (α̂ → 0 or 1 with β̂ → ∞).
- **near_random** flags fall steeply with true β and with trial count
  (`near_random_by_beta.png`).

## Results notes: model recovery (160 trials, 1,200 agents)

- **Most agents are practical ties.** The locked tie rule (winner − runner-up
  < 0.01 LL/trial, i.e. < 1.6 nats over 160 held-out trials) classifies 72% of
  agents as ties; the median margin is 0.005 LL/trial. Accuracy counting ties as
  errors is 0.15; excluding ties it is 0.53 (chance 0.25). Without the tie rule,
  the raw winner is correct for 38% of agents.
- **The Bayesian learner is the most distinguishable** (sensitivity 0.32; 0.86
  of its non-tied agents are correctly identified), in every condition. It is
  also the most common *wrong* choice for RL-generated agents (6–10%).
- **Asymmetric RL is almost never recovered**: 0.09 overall and 0.06 in the
  volatile condition. On asymmetric-RL data, fixed RL and the hybrid beat it at
  the median (+0.005 LL/trial; fixed RL beats it for 69–80% of agents). The
  extra parameter costs more out of sample than it gains, compounded by a
  pathology of the fitted asymmetric model: in 2.9% of folds it converges to
  α⁺ → 0, α⁻ ≈ 0.7–0.97, β → 10⁶–10⁷, a deterministic lose-shift rule that
  assigns near-zero probability to any violating held-out choice (held-out LL
  down to −3 × 10⁶ per trial). These folds make mean score differences
  meaningless, so summaries use medians and per-agent winners.
- **Hybrid vs fixed RL**: on hybrid data, fixed RL's median difference is
  +0.001 (moderate) and −0.000 (volatile); it beats the hybrid for about half
  the agents. As expected, the two are behaviourally interchangeable for most
  sampled η.
- **Volatility**: sensitivity for the hybrid rises from 0.06 (moderate) to 0.15
  (volatile), and tie rates are similar across conditions (0.70–0.74). Reversal
  frequency does not rescue model identifiability at 160 trials in this design.
- **Benchmarks**: every primary model beats perseveration and random at the
  median on every dataset (differences ≈ 0.03–0.13 LL/trial), so the learning
  models capture more than choice autocorrelation.

## Comparison with "Expected results"

Matches: fixed RL recovery at 160 moderate with β ≳ 2; weak η vs κ; η confusion
with fixed RL; h best in volatile; Bayesian learner most distinguishable;
fixed/asymmetric confusion; learning models above random on their own data
(except near-chance agents and runaway folds, analysed above).
Deviations: α⁻ better than α⁺; η worst in moderate rather than stable; the
Bayesian learner is *not* more distinguishable in the volatile condition than in
the others (sensitivity 0.27 vs 0.33–0.36); and the tie rate is much higher than
the spec implies.

## Comprehension checks

**Q1 (revisited with my plots).** `parameter_tradeoffs.png` shows the trade-off
in two forms. Across agents, recovered κ–β (hybrid) and α⁺–β (asymmetric)
correlate at about −0.4, although true values were sampled independently. At
the level of single agents, the runaway corner (α̂ near 0 or 1 with β̂ > 20)
appears as triangles at the top edge of the β panels in `fig2_*`. The
high-β-subset curves in `fig3` show that the problem is concentrated in agents
whose choices carry little information.

**Q3. What does an off-diagonal entry in a confusion matrix mean
scientifically?** Entry (i, j) is the probability that data truly generated by
mechanism i are better predicted, out of sample, by model j. A large
off-diagonal entry means that under this task design, trial count and
parameter distribution, behaviour from mechanism i is not distinguishable from
mechanism j. Selecting j in real data would therefore not be evidence against
i. It is a property of the design and model set, not of any participant. It can
be asymmetric: here the Bayesian learner absorbs some RL-generated agents, but
RL models rarely absorb Bayesian-generated ones.
