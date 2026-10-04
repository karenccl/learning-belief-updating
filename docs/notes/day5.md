# Day 5: predictive validation

## Status

- Demonstration datasets: 100 agents per primary generating model, moderate
  condition, 160 trials (`data/simulated/demo_*.csv`, regenerable from the seed).
  True parameters were not used at any point in this step.
- All six models were cross-validated on each dataset (masked leave-block-out,
  4 folds × 10 starts). Tables: `demo_cv_scores.csv`, `demo_cv_summary.csv`
  (mean, median, paired differences vs the selected model with 95% CI, and the
  proportion of agents beating it). Forward-chaining sensitivity check on the
  same datasets: `demo_forward_chaining_scores.csv`.
- Parametric predictive checks: the selected model and the true generating
  model were fitted to each agent's full data (20 starts), and 100 replicate
  datasets were simulated per agent from the fitted parameters in the same task
  environment. Tables: `ppc_reversal_curves.csv`, `ppc_block_curves.csv`,
  `ppc_summaries.csv`, `ppc_fitted_learning_rates.csv`. Figures:
  `fig6_ppc_reversal_curves.png`, `ppc_summaries.png`,
  `demo_paired_differences.png`.
- Real data: none (Day 1 decision).
- Total wall-clock time for Day 5: 509 s.

## Results notes: held-out comparison on the demonstration datasets

The locked selection rule (highest mean held-out LL per trial, i.e. largest
summed held-out LL) selected:

| Demo data from | Selected (mean) | Best by median | True model: mean / median |
|---|---|---|---|
| Fixed RL | PH hybrid (−0.509) | Fixed RL (−0.506) | −12.51 / −0.506 |
| Asymmetric RL | PH hybrid (−0.545) | Fixed RL ≈ hybrid (−0.615) | asym. RL mean dominated by runaway folds |
| PH hybrid | Fixed RL (−0.547) | PH hybrid (−0.594) ≈ fixed (−0.595) | −0.550 / −0.594 |
| Bayesian CP | Bayesian CP (−0.604) | Bayesian CP (−0.655) | — |

- **Only the Bayesian learner is recovered at the dataset level.** It beats fixed
  RL by 0.010 LL/trial (95% CI 0.006–0.013) and every other model.
- **On fixed-RL data the selection is driven by one agent.** A single
  runaway-β fold drags fixed RL's mean to −12.5, although fixed RL has the best
  median and beats the hybrid for 67% of agents. The summed log likelihood is a
  legitimate score (a catastrophically overconfident prediction *is* a bad
  prediction), but here it turns a practical tie into a "win". I report both and
  do not change the locked rule after seeing the data.
- **Fixed RL and the hybrid are interchangeable** on hybrid- and fixed-generated
  data (paired median differences ≤ 0.003 LL/trial, well inside the tie zone).
- **Asymmetric RL is never selected**, even on its own data: the lose-shift
  runaway pathology (α⁺ → 0, β → 10⁷) produces extreme held-out losses.
- **Benchmarks**: every primary model beats perseveration and random by
  0.08–0.18 LL/trial (mean) on every dataset.
- **Forward chaining** gives a different, noisier picture: early folds train on
  only 40–80 trials, runaway fits become common (fixed RL on its own data: mean
  −16,859), and the median ranks reshuffle. The masked scheme's larger training
  sets matter more here than its information-flow caveat; forward chaining is
  too data-hungry to be the primary scheme at 160 trials.

## Results notes: parametric predictive checks

- **Reversal curves.** All fitted models reproduce the qualitative shape: a drop
  to about 0.2–0.35 on the first post-reversal trial, recovery over about 10
  trials, and the pre-reversal plateau. 83–97% of the 30 observed points lie
  within the 90% interval of simulated group curves.
- **A mismatch the score missed.** On fixed-RL data, the *selected* hybrid
  underpredicts lose-switch (0.478 [0.468, 0.489] vs observed 0.498; predictive
  p = 0.00) and early post-reversal accuracy (trials 0–4: 0.347 vs 0.389). The
  true fixed-RL model reproduces both. The same underprediction of lose-switch
  appears for the hybrid on asymmetric-RL data (p = 0.04).
- **The true model can misfit too.** On hybrid data, the refitted *true* hybrid
  overpredicts win-stay (0.718 vs 0.707, p = 0.97) and repeat probability
  (0.644 vs 0.634, p = 1.00). The likely cause is bias in ML point estimates of
  the weakly identified η/κ (30–39% of η̂ at a boundary), propagated into the
  simulations. The intervals include simulation noise only, not parameter
  uncertainty, so they are too narrow.
- **Agent-level coverage** of the 90% interval is 0.86–0.99 across statistics;
  single-agent intervals are wide, so agent-level checks have little power.
- **Fitted learning rates.** The hybrid's implied α_t rises after reversals
  (hybrid data: 0.196 before → 0.265 in trials 0–4 → 0.228 after), as the
  Pearce–Hall mechanism predicts. It does the same on fixed- and asymmetric-RL
  data (0.317 → 0.363), so this signature alone cannot be read as evidence for
  adaptive learning rates.

## Writing: limitations (first draft)

1. **Non-identifiability under the planned design.** At 160 trials most
   pairwise model distinctions fall inside a 0.01 LL/trial tie zone, and η, α⁺
   and the Bayesian β are poorly recovered in some conditions. Conclusions about
   those parameters or distinctions should not be drawn from comparable real
   data.
2. **Unregularised ML.** Runaway estimates (β → ∞ with α or h at a degenerate
   value) produce catastrophic held-out predictions for a few percent of folds
   and dominate means. Priors or hierarchical estimation would regularise them;
   this was out of scope (deferred).
3. **Restricted model space.** One adaptive and one Bayesian learner, no
   perseveration in the primary models, fixed reward probabilities
   (0.80/0.20), and one simulation parameter distribution. Recovery results are
   conditional on all of these.
4. **Simulation assumptions.** The Bayesian learner knows the true p_H and p_L,
   an advantage the RL models lack; this may explain part of its
   distinguishability.
5. **Masked CV caveat.** Held-out outcomes feed state updates that affect later
   training trials. Forward chaining removes this but is too data-hungry at 160
   trials.
6. **Predictive-check intervals** ignore parameter uncertainty.

## Comprehension checks

**Q6. Why "parametric predictive check" rather than "posterior predictive
check"?** A posterior predictive check simulates from parameters drawn from a
posterior distribution, so its intervals include parameter uncertainty. With
maximum likelihood there is no posterior: we simulate from one point estimate
per agent, so the intervals reflect only the task's and the choice rule's
sampling noise. Calling it "posterior" would imply a Bayesian treatment and an
uncertainty propagation that did not happen. The true-hybrid misfit above shows
why the difference matters.

**Q7. If an adaptive learner wins in volatile but not stable simulations, what
does that imply and not prove about human learning?** It would show that, under
these task conditions, volatile environments generate behaviour where
prediction-error-scaled learning rates improve out-of-sample prediction among
the candidate models, and that the task can detect that signature when
reversals are frequent. It would not show that humans adapt their learning rate
to volatility, that the adaptive model is the mechanism, or that people in
stable environments lack adaptation (low power there looks the same as absence).
Here, even the simulated adaptive learner was rarely distinguishable from fixed
RL, and the post-reversal rise in fitted α_t appeared for fixed-RL data too.

**Q8. If a model predicts choices well but fails to reproduce post-reversal
adaptation, what should you conclude?** That good aggregate likelihood is
coming from the parts of the task that dominate the trial count (stable stretches
of choosing the better option), while a process that matters scientifically is
missing or misspecified. The model should not be used to make claims about
reversal learning. The next step is to diagnose the mismatch (e.g. too slow a
switch suggests missing surprise-driven learning or perseveration) and extend
or replace the model. Here the selected hybrid on fixed-RL data is exactly this
case: its score is competitive, but it underpredicts lose-switch and early
post-reversal accuracy.
