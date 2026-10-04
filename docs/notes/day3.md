# Day 3: adaptive and Bayesian models, and cross-validation

## Status

- Pearce–Hall hybrid and Bayesian change-point learner implemented in JAX with
  NumPy references (JAX = reference to 1e-8; gradients match finite differences).
- Tests: hybrid order of operations (α_t uses A_t before the outcome; A updated
  after the value update); η = 0 reduces exactly to fixed RL with α = 0.5κ
  (reference and JAX); α_t ∈ (0, 1) on every trial; Bayesian beliefs stay in
  (0, 1) and imply Q₀ + Q₁ = p_H + p_L = 1; hand-calculated two-trial update
  (b′₁ = 0.8, b₂ = 0.74, Q = (0.644, 0.356), b′₂ = 0.148/0.356); monotone belief
  change under deterministic evidence converging to the hazard-limited fixed point.
- Masked leave-block-out CV and `sequential_predictive_log_likelihood`, plus a
  forward-chaining variant. Tests: masks disjoint and covering; changing a
  held-out choice leaves the training loss on all earlier trials bit-identical;
  held-out score equals the sum of per-trial log probabilities on held-out trials;
  random benchmark scores exactly log 0.5 per trial.
- Latent trajectories for one volatile-condition agent per model:
  `results/figures/latent_trajectories.png`; task schedule figure:
  `results/figures/fig1_task_schedule.png`.
- Small model-recovery check (5 agents × 4 generating models × 3 conditions,
  all 6 fitted models, 4 folds × 10 starts): no failed or unconverged folds;
  40/1440 folds (2.8%) have β̂ > 20; every learning model beats random on
  median held-out LL. The Bayesian learner passed its hand-calculated test on
  the first attempt, so the stop rule was not needed.

## Writing: Methods — Pearce–Hall hybrid (draft)

The adaptive learner follows the hybrid Rescorla–Wagner/Pearce–Hall form (Li et
al., 2011). A single associability A_t ∈ (0, 1), shared by both options and
initialised at A₁ = 0.5, scales the learning rate: α_t = κ A_t,
Q_{a_t,t+1} = Q_{a_t,t} + α_t δ_t, and after the value update
A_{t+1} = A_t + η(|δ_t| − A_t). Free parameters are η, κ ∈ (0, 1) and β > 0.
Surprising outcomes (large |δ|) raise associability and so the next learning
rate; this is a prediction-error-driven associability model, not a normative
volatility estimator. Two simplifications relative to Li et al. are noted: the
associability is shared rather than cue-specific, and when η → 0 the model
reduces to fixed RL with α = 0.5κ, so some confusion with fixed RL at low η is
expected.

## Writing: Methods — Bayesian change-point learner (draft)

The Bayesian learner tracks z_t ∈ {0, 1}, the identity of the currently
advantageous option, assuming it switches with hazard h on each trial and that
the advantageous option pays with p_H = 0.8 and the other with p_L = 0.2 (the
true generative values; this gives the learner correct knowledge of the reward
structure that the RL models lack). With b′_{t−1} the posterior that z = 0 after
trial t − 1 (b′₀ = 0.5), the predictive belief is
b_t = (1 − h) b′_{t−1} + h(1 − b′_{t−1}); expected rewards
Q_{0,t} = b_t p_H + (1 − b_t) p_L and Q_{1,t} = (1 − b_t) p_H + b_t p_L enter the
shared choice rule; after observing (a_t, r_t) the belief is updated by Bayes'
rule, b′_t = ℓ₀ b_t / (ℓ₀ b_t + ℓ₁ (1 − b_t)), with ℓ_z = P(r_t | a_t, z). Free
parameters are h ∈ (0, 1) and β > 0. Because Q₁ − Q₀ = (p_H − p_L)(1 − 2b_t) is
bounded by ±0.6, β is on a different effective scale from the RL models.

## Writing: Methods — Cross-validation (draft)

Models were compared by held-out sequential log likelihood from a masked
leave-block-out procedure. Trials were divided into fixed 40-trial blocks by
trial number, independently of the reversal schedule (four folds at 160
trials). For each fold, parameters were estimated by minimising the negative log
likelihood summed over training trials only, while the model's latent states
were propagated through the *entire* sequence of observed choices and outcomes.
With the fitted parameters fixed, the model was run through the full sequence
again and the log probabilities of the observed choices on held-out trials were
summed; within a held-out block the model updates after each observed outcome
and never uses future trials. The score is the sum over folds divided by the
number of trials. We did not re-initialise latent states at the start of held-out
blocks, because a mid-task reset imposes forgetting the participant never
experienced and penalises slow-recovering models more than fast ones.
*Caveat:* in this scheme held-out choices and outcomes are inputs to the state
updates that affect later training trials, although they are never scored
during training; a forward-chaining check (fit blocks 1..k−1, predict block k)
removes this information flow.

## Comprehension checks

**Q4. Why is random trial-wise CV inappropriate for sequential learning data?**
Trials are not exchangeable: each choice depends on the whole history of
previous choices and outcomes through the latent state. Randomly scattering
held-out trials either breaks the sequence the model needs (if held-out trials
are removed from the state updates) or lets neighbouring training trials carry
almost the same information as the held-out trial (strong temporal
autocorrelation), inflating apparent predictive accuracy. Contiguous blocks keep
the temporal structure intact and make the held-out prediction a genuine
out-of-sample test of the dynamics.

**Q5. Why can outcomes update the model within a held-out block, but future
trials must not be used? Why does re-initialising bias comparison?** Sequential
prediction mirrors the task: before choice t the participant has seen every
outcome up to t − 1, so the model may use them too; scoring P(a_t | history up
to t − 1) is exactly the one-step-ahead predictive probability. Using choice t
itself or anything later would leak the answer. Re-initialising latent states at
the start of a held-out block pretends the participant forgot everything; models
that recover quickly from a wrong state (high α, high h) lose less from this
artificial reset, so the comparison would partly measure speed of recovery from
a reset rather than the learning mechanism.

**Q10. Why is β not comparable between RL models and the Bayesian learner?**
β multiplies the value difference, and the two model families produce value
differences on different scales: for the Bayesian learner
Q₁ − Q₀ = (p_H − p_L)(1 − 2b_t) ∈ [−0.6, 0.6] and saturates at a hazard-limited
value, whereas RL value differences depend on α and the reward history and can
approach ±1. The same β therefore implies different choice determinism, and β
also absorbs model misspecification differently in each model.
