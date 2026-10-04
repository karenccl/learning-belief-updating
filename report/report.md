# Identifiability and Predictive Comparison of Learning Models in a Simulated Reversal-Learning Task

## Abstract

Computational models of learning are routinely fitted to reversal-learning
data, but whether their parameters and mechanisms can be recovered under a
given design is rarely checked. We simulated a two-armed probabilistic reversal
task (p = 0.80/0.20; stable, moderate and volatile reversal schedules; 80–320
trials) and assessed four learning models: fixed-learning-rate RL,
valence-asymmetric RL, a Pearce–Hall hybrid with prediction-error-driven
associability, and a Bayesian change-point learner. We also included random and
perseveration benchmarks. Using maximum likelihood with JIT-compiled JAX
likelihoods and multi-start optimisation, we ran parameter recovery (3,600
simulated agents), model recovery scored by masked leave-block-out held-out
log likelihood (1,200 agents × 6 fitted models), and parametric predictive checks
of post-reversal adaptation. Fixed RL was well identified (true-vs-recovered
r > 0.9 for agents with β ≥ 2). The hybrid's associability rate η was essentially
unrecoverable (Spearman ≤ 0.36), and asymmetric-RL learning rates were recovered
only moderately. At 160 trials, 72% of agents were practical ties: the winning
model beat the runner-up by less than 0.01 log-likelihood units per trial.
Excluding ties, selection accuracy was 0.53 against a chance level of 0.25. The
Bayesian learner was the most distinguishable model; asymmetric RL was almost
never recovered, even on its own data. Unregularised maximum likelihood produced
occasional runaway estimates, which made held-out means unreliable. Predictive
checks detected a lose-switch misfit that the held-out score had missed. A
post-hoc sensitivity analysis with bounded β (≤ 20) and hazard (h < 0.5) removed
every catastrophic held-out score and corrected the fragile dataset-level model
selections, but left parameter and model recovery essentially unchanged. Under
this design, only coarse model distinctions and a subset of parameters support
inference.

**CV line.** Built a reproducible JAX/SciPy pipeline (84 unit tests) that
simulates, fits and validates five reinforcement-learning and Bayesian models of
reversal learning; parameter and model recovery over 4,800 simulated agents
showed that most mechanism distinctions are not identifiable at typical trial
counts.

## 1. Introduction

Reversal learning tasks, where the better of two options switches without
warning, are a standard probe of how people learn from feedback and update
beliefs when the environment changes. Models of this behaviour differ in how
the learning rate is set. It may be fixed, differ for good and bad outcomes, be
driven by recent surprise (Pearce–Hall associability), or come from Bayesian
inference about latent contingency changes. Fitted parameters are often read as
individual traits, and winning models as mechanisms. Both readings require that
the task can recover the parameters and tell the models apart (Wilson & Collins,
2019). We therefore asked, entirely in simulation with known ground truth:
(RQ1) whether each model's parameters are recoverable; (RQ2) whether the
models can be distinguished by held-out predictive performance; (RQ4) whether
fitted models reproduce post-reversal behaviour; and (RQ5) how trial count and
reversal frequency affect identifiability. A real-data application (RQ3) was
omitted because no dataset passed the pre-specified feasibility check.

## 2. Methods

**Task.** Two options (coded 0/1) paid a binary reward with probability 0.80
(advantageous) or 0.20. Option 0 was initially advantageous, and the
probabilities swapped at unannounced reversals: *stable*, one reversal after 75%
of trials; *moderate*, every 40 trials; *volatile*, every 20 trials (Fig. 1). At
160 trials this gives 1, 3 and 7 reversals. Parameter recovery also used 80 and
320 trials. Because the stable condition always has one reversal, its hazard
rate differs across trial counts.

**Models** (Table 1). All models share the choice rule
P(a_t = 1) = logistic(β[Q₁ − Q₀] + ρ s_{t−1}), with s ∈ {−1, +1} coding the
previous choice symmetrically and ρ = 0 in the four primary models. *Fixed RL*
updates the chosen value by αδ_t. *Asymmetric RL* uses α⁺ for δ ≥ 0 and α⁻
otherwise. The *Pearce–Hall hybrid* sets α_t = κA_t, with a single shared
associability A_{t+1} = A_t + η(|δ_t| − A_t) (A₁ = 0.5; Li et al., 2011). The
*Bayesian change-point learner* tracks the probability that option 0 is
currently advantageous, applies a hazard-h transition before each choice,
updates by Bayes' rule with the true p_H = 0.8 and p_L = 0.2, and chooses on the
implied expected rewards. Because its value difference is bounded by ±0.6, its
β is not comparable with the RL models'. Benchmarks were random choice and
perseveration (bias + ρ s_{t−1}).

*Table 1. Model specification (simulation ranges uniform; β log-uniform).*

| Model | Update | Free parameters (simulation range) |
|---|---|---|
| Random | P(a=1) = 0.5 | — |
| Perseveration | logit = b + ρ s_{t−1} | b, ρ ∈ (−5, 5) |
| Fixed RL | Q_a ← Q_a + αδ | α (0.05–0.80), β (0.5–8) |
| Asymmetric RL | Q_a ← Q_a + α^{sign δ} δ | α⁺, α⁻ (0.05–0.80), β |
| PH hybrid | α_t = κA_t; A ← A + η(\|δ\| − A) | η (0.05–0.80), κ (0.10–0.90), β |
| Bayesian CP | hazard transition + Bayes update | h (0.01–0.30), β |

**Fitting.** Parameters were estimated separately for each agent by maximum
likelihood on an unconstrained scale (logistic, softplus + 10⁻⁴ and 5·tanh
transforms). The likelihood was written with `jax.lax.scan` in float64,
JIT-compiled and differentiated exactly. JAX and plain-Python reference
implementations agree to 10⁻⁸, and gradients match finite differences.
Optimisation used L-BFGS-B from N(0, 1.5²) starts (10 in recovery loops, 20
otherwise). A start counted as converged if it was finite with gradient norm
< 10⁻³. Boundary estimates (bounded parameter < 0.01 or > 0.99; β > 20; |ρ| > 4.9)
and near-random fits (improvement over random smaller than the number of
parameters, in nats) were flagged, never excluded. Seeds were derived per agent
from one master seed. The full pipeline reproduces bit-for-bit from a clean
clone.

**Recovery designs.** *Parameter recovery*: 4 models × 3 trial counts × 3
conditions × 100 agents, each refitted with its generating model. *Model
recovery*: 4 generating models × 3 conditions × 100 agents at 160 trials, all six
models fitted. Models were scored by **masked leave-block-out** cross-validation.
Four fixed 40-trial blocks were held out in turn; latent states ran through the
whole sequence; only training trials entered the fitting objective; and only
held-out trials were scored with the fitted parameters. The agent's selected
model had the highest held-out log likelihood per trial. It was recorded as a
*tie* if its margin over the runner-up was below 0.01. We report three accuracy
measures that differ in their denominator. *Accuracy* counts ties as errors
(correct / all agents). *Accuracy excluding ties* is correct / non-tied agents.
*Raw-winner accuracy* ignores the tie rule: the generating model had the highest
held-out score, over all agents. Chance is 0.25 for all three. Caveat: held-out outcomes
still feed state updates that affect later training trials. A forward-chaining
variant, which removes this, was run as a sensitivity check.

**Predictive checks.** For each generating model, a separate 100-agent
demonstration dataset (moderate, 160 trials) was treated as observed data. We
selected the model with the best mean held-out score on that dataset. That model
and the true model were then fitted to each agent, and 100 replicate datasets
per agent were simulated from the fitted parameters. These are parametric (not
posterior) predictive checks.

## 3. Results

**Parameter recovery (RQ1, RQ5; Fig. 2–3, Table 2).** No fit failed, and every
selected start converged. Fixed RL recovered well. Asymmetric-RL and
hybrid-κ recovery was moderate, and hybrid η was essentially unrecoverable.
Recovery improved with trial count for most parameters, but not for η or the
Bayesian β. Low-β agents carried little information: near-random flags fell from
24–58% of fits for β < 1 to 0% for β > 4. Estimates showed systematic
trade-offs, although true parameters were sampled independently. Spearman
correlations between estimates at 160 trials were −0.43 for asymmetric α⁺–β,
−0.41 for hybrid κ–β, and +0.53 for Bayesian h–log β. Two degenerate ridges
explain most runaway estimates. For RL, α → 0 (or 1) with β → ∞ identifies only
αβ (or becomes win-stay/lose-shift). For the Bayesian learner, writing the
transition as b_t = 0.5 + (1 − 2h)(b′_{t−1} − 0.5) shows why. At h = 0.5 exactly,
beliefs stay at 0.5 and the model is pure random choice. As h → 0.5 with
β → ∞, however, the belief carries only the last trial's evidence, scaled by
(1 − 2h). The choice logit then becomes a graded win-stay/lose-shift rule with
strength ≈ 0.36·β(1 − 2h), and only the product β(1 − 2h) is identified.
Of the 900 locked Bayesian fits, 108 lie on this ridge (none at h = 0.5 exactly;
median 1 − 2ĥ = 5 × 10⁻⁴, β̂ ≈ 1,850, β(1 − 2ĥ) = 0.88; all beat random).
Along the ridge the NLL decreases monotonically as h → 0.5 at fixed product,
then jumps to the random-choice value at h = 0.5. The supremum is therefore
never attained. A further 37 fits have ĥ > 0.52 (median 0.99): an "expect
alternation" regime outside the model's intended use, found only in low-β agents
(`bayes_h_ridge.csv`). Contrary to our
expectation, α⁻ was recovered better than α⁺. We also examined whether the weak
Bayesian β recovery reflected a coding error. With 2,000 volatile trials, both
Bayesian parameters recovered well (r = 0.94, 0.98), so it is a design
limitation, not a bug.

*Table 2. Spearman correlation between true and recovered parameters (100 agents
per cell), and Pearson r for agents with true β ≥ 2 at 320 trials (log scale for
β). Condition order: stable / moderate / volatile.*

| Model: parameter | 80 trials | 160 trials | 320 trials | 320, β ≥ 2 |
|---|---|---|---|---|
| Fixed RL: α | .58 / .64 / .68 | .83 / .62 / .66 | .84 / .77 / .78 | .95 / .94 / .97 |
| Fixed RL: β | .78 / .84 / .59 | .94 / .85 / .79 | .97 / .98 / .89 | .83 / .89 / .90 |
| Asym. RL: α⁺ | .41 / .32 / .30 | .55 / .53 / .51 | .61 / .49 / .62 | .77 / .69 / .82 |
| Asym. RL: α⁻ | .53 / .32 / .40 | .55 / .60 / .58 | .75 / .58 / .69 | .94 / .89 / .84 |
| Asym. RL: β | .60 / .38 / .25 | .62 / .61 / .65 | .84 / .62 / .64 | .74 / .61 / .59 |
| PH hybrid: η | .04 / −.03 / .25 | .23 / .12 / .36 | .33 / .03 / .26 | .47 / .23 / .37 |
| PH hybrid: κ | .56 / .27 / .43 | .53 / .55 / .51 | .73 / .69 / .66 | .87 / .84 / .86 |
| PH hybrid: β | .73 / .65 / .64 | .95 / .83 / .76 | .97 / .84 / .93 | .94 / .89 / .87 |
| Bayesian CP: h | .48 / .15 / .56 | .42 / .37 / .66 | .63 / .60 / .51 | .73 / .68 / .80 |
| Bayesian CP: β | .56 / .17 / .33 | .56 / .47 / .45 | .56 / .53 / .52 | .06 / −.05 / .75 |

**Model recovery (RQ2, RQ5; Fig. 4–5, Table 3).** At 160 trials, 72% of agents
were practical ties (median winning margin 0.005 per trial, i.e. under one nat
over the whole task). Excluding ties, 53% were correctly classified, against a
chance level of 25%. The Bayesian learner was the most identifiable model in
every condition. Asymmetric RL was almost never recovered: on its own data,
fixed RL and the hybrid beat it at the median (+0.005). The fitted asymmetric
model also converged in 2.9% of folds to a deterministic lose-shift solution
(α⁺ → 0, β → 10⁷). Its held-out log likelihood then fell as low as −3 × 10⁶ per
trial, which makes means meaningless, so we report medians and per-agent
winners. Fixed RL and the hybrid were interchangeable. Reversal frequency barely
changed identifiability: tie rates were 0.70, 0.71 and 0.74 in the stable,
moderate and volatile conditions. The Bayesian learner was *not* easier to
identify under volatility, contrary to expectation. Every primary model beat
both benchmarks at the median (by 0.03–0.13 LL/trial), so the learning models
captured more than choice autocorrelation.

*Table 3. Model recovery at 160 trials (100 agents × 3 conditions per row).
"Selected" columns show P(selected | generating), pooled over conditions.*

| Generating | Fixed RL | Asym. RL | PH hybrid | Bayes CP | Tie | Accuracy excl. ties | Raw winner correct |
|---|---|---|---|---|---|---|---|
| Fixed RL | **.08** | .04 | .04 | .06 | .77 | .37 | .42 |
| Asymmetric RL | .03 | **.09** | .05 | .09 | .74 | .35 | .18 |
| PH hybrid | .04 | .03 | **.11** | .10 | .72 | .41 | .35 |
| Bayesian CP | .01 | .01 | .04 | **.32** | .63 | .86 | .56 |

**Predictive comparison and checks (RQ4; Fig. 6, Table 4).** On the
demonstration datasets, the locked rule (highest mean held-out LL) recovered the
Bayesian learner. It beat fixed RL by 0.010 LL/trial (95% CI 0.006–0.013). On
fixed-RL data, one runaway fold dragged fixed RL's mean below the hybrid's, so
the hybrid was "selected", even though fixed RL had the better median and beat
the hybrid for 67% of agents. The predictive checks exposed the difference: the
selected hybrid underpredicted lose-switch (0.478 [0.468, 0.489] vs observed
0.498) and early post-reversal accuracy (0.347 vs 0.389), while fixed RL
reproduced both. All fitted models reproduced the shape of the reversal curve
(83–97% of points inside the 90% band). The refitted *true* hybrid overpredicted
win-stay and repetition on its own data. The likely cause is estimation bias in
its weakly identified parameters, since the intervals include simulation noise
but not parameter uncertainty. The hybrid's fitted learning rate rose after
reversals on all RL-generated datasets, including fixed-RL data, so this
signature is not diagnostic of adaptive learning. Forward chaining, with only
40–120 training trials, was dominated by runaway fits and reshuffled the rankings.

*Table 4. Held-out LL per trial on the demonstration datasets (masked
leave-block-out; median over 100 agents, mean in parentheses where distorted by
runaway folds).*

| Data from | Fixed RL | Asym. RL | PH hybrid | Bayes CP | Persev. | Random |
|---|---|---|---|---|---|---|
| Fixed RL | **−0.506** (−12.5) | −0.529 (−1811) | −0.514 | −0.548 | −0.667 | −0.693 |
| Asym. RL | −0.615 | −0.632 (−932) | −0.615 | −0.624 | −0.702 | −0.693 |
| PH hybrid | −0.595 | −0.630 (−974) | −0.594 | −0.598 | −0.693 | −0.693 |
| Bayes CP | −0.661 | −0.677 (−5916) | −0.663 | **−0.655** | −0.710 | −0.693 |

*Table 5. Optimisation diagnostics (parameter recovery / CV folds in model
recovery).* Failure and non-convergence rates were 0 everywhere. Boundary-flag
rates: fixed RL 0.07 / 0.12; asymmetric RL 0.30 / 0.40; hybrid 0.47 / 0.72
(mostly η); Bayesian 0.25 / 0.47. β-runaway rates: 0.03 / 0.03; 0.10 / 0.14;
0.01 / 0.03; 0.13 / 0.05. near_random rates: 0.08, 0.12, 0.15, 0.20 in parameter
recovery. The mean proportion of starts within 0.1 nats of the best was
0.85–0.98, so likelihood surfaces were mostly unimodal and the difficulty lay in
flat ridges, not local optima.

**Post-hoc sensitivity analysis: bounded β and h (not pre-specified).** Because a
single runaway fold could decide the mean-score comparison, we refitted every
agent after seeing the results. The same agents, folds and starting values were
used, but with β = 10⁻⁴ + 20·logistic(θ) for all models and h = 0.5·logistic(θ)
for the Bayesian learner. True values (β ≤ 8, h ≤ 0.3) lie inside the bounds.
The locked analysis above remains primary. The bounds removed **every**
catastrophic held-out score from the learning models: in model recovery,
agents with held-out LL < −2 per trial fell from 106 to 0 (asymmetric RL), 6 to 0
(fixed RL) and 2 to 0 (hybrid), and the worst score rose from −8 × 10⁵ to −1.7.
On the demonstration datasets, mean-based and median-based selections now
agree for fixed-RL data (fixed RL, which the locked mean had lost to the hybrid)
and for asymmetric-RL data (fixed RL). Hybrid data still favour fixed RL.
**Model recovery barely changed**: tie rate 0.72 → 0.70; accuracy excluding
ties 0.53 → 0.51; raw-winner accuracy 0.38 → 0.37; the Bayesian learner remained the only well-recovered model
(0.56 → 0.59), and asymmetric RL was still rarely recovered (0.18 → 0.19). Its
failure is therefore genuine non-identifiability, not the runaway pathology.
**Parameter recovery was essentially unchanged**; at 160 trials, Spearman
coefficients moved by < 0.1 for most parameters.

The exception is the Bayesian ĥ, whose Spearman *fell* in the moderate
(0.37 → 0.27) and volatile (0.66 → 0.54) conditions. This is a ranking artefact
of the ridge, not a real loss. Agents that land in the degenerate regimes have
higher true h (mean 0.20 vs 0.145). Their spurious locked estimates
(ĥ ≈ 0.5–1) ranked them high by accident, whereas within the interior regime
recovery is unchanged (0.49 → 0.52). The bounded fits confirm that the product
is what is identified. Ridge fits move to the β cap (20) with median ĥ = 0.479,
giving β(1 − 2ĥ) ≈ 0.84, essentially the locked value of 0.88. In short, bounding
fixes the *scoring* pathology but not the *identifiability* limits
(`sensitivity_bounded.png`, `sensitivity_*.csv`).

## 4. Discussion

Under this design and model space, the Bayesian change-point learner provided
the most identifiable account of its own simulated data, and fixed RL was the
best-identified parameterised model. Recovery analyses indicate which parameters
and model distinctions are identifiable under these conditions. They do not
establish any model as the unique psychological mechanism. Three conclusions
follow. First, **most mechanism distinctions are below the resolution of a
160-trial task** with 0.80/0.20 rewards. With 72% ties, selecting "the best
model" per participant would mostly reflect noise. Second, **several
psychologically attractive parameters should not be interpreted** at this
design: η (associability), α⁺ versus α⁻ at 160 trials or fewer, and the
Bayesian β outside volatile schedules. The popular conclusion that people learn
asymmetrically from positive and negative outcomes is especially fragile: the
asymmetric model was rarely recovered even when true, and its fits can
degenerate into lose-shift rules. Third, **held-out scores and predictive checks
complement each other.** A handful of runaway fits can decide an aggregate
comparison, and only the behavioural checks revealed that the "winner" misfit
lose-switch behaviour. The post-hoc bounded refit confirms the diagnosis: once β
cannot run away, the fixed-RL dataset is correctly attributed to fixed RL. The
general lesson is that unregularised ML on short sequences systematically
produces a few near-deterministic fits, and any score that sums log
probabilities inherits their failures. Bounds or priors, or robust summaries,
should be part of the analysis plan, not an afterthought.

**Limitations.** (i) Estimation was unregularised maximum likelihood. The
post-hoc bounded analysis shows that simple bounds remove the catastrophic
held-out scores without improving identifiability. Priors or hierarchical models
might do better on the latter; this was out of scope. (ii) The model space is
small, with one adaptive and one Bayesian learner and no perseveration in the
primary models. Results also depend on the sampled parameter ranges and the
fixed reward probabilities. (iii) The Bayesian learner knows the true reward
probabilities, an advantage the RL models lack, which may inflate its
distinguishability. (iv) In masked cross-validation, held-out outcomes feed
later state updates. The forward-chaining alternative was too data-hungry to
replace it at 160 trials. (v) Predictive-check intervals omit parameter
uncertainty. (vi) There is no real-data application.

**Future work.** Hierarchical or MAP estimation; longer tasks or more extreme
reward contrasts designed to separate α⁺/α⁻ and fixed/adaptive learning rates;
a Bayesian learner that estimates p_H and p_L; perseveration in the primary
models (Katahira, 2018); and a real reversal-learning dataset passing the
feasibility checklist.

## 5. Reproducibility

The code, configuration (`config/default_config.yaml`, master seed 20261001) and
environment (`environment.yml`: Python 3.11, JAX 0.10.2, SciPy 1.17 with
OpenBLAS) are in the repository. Reproduction commands are in `README.md`. All
figures are generated from `results/tables/*.csv`. The complete pipeline takes
about 20 minutes on 21 CPU workers. A clean clone reproduces the tests, smoke
test, figures, and individual recovery fits bit-for-bit.

**Figures** (`results/figures/`): 1 `fig1_task_schedule.png`;
2 `fig2_true_vs_recovered_{80,160,320}.png`; 3 `fig3_recovery_vs_trial_count.png`;
4 `fig4_confusion_matrix.png`; 5 `fig5_heldout_comparison.png`,
`fig5b_paired_differences.png`; 6 `fig6_ppc_reversal_curves.png`. Supplementary:
`bias_mae_by_condition.png`, `parameter_tradeoffs.png`, `near_random_by_beta.png`,
`margin_distribution.png`, `demo_paired_differences.png`, `ppc_summaries.png`,
`latent_trajectories.png`.

## References

Daw, N. D. (2011). Trial-by-trial data analysis using computational models. In
M. R. Delgado, E. A. Phelps, & T. W. Robbins (Eds.), *Decision Making, Affect,
and Learning: Attention and Performance XXIII* (pp. 3–38). Oxford University Press.

Katahira, K. (2018). The statistical structures of reinforcement learning with
asymmetric value updates. *Journal of Mathematical Psychology, 87*, 31–45.

Li, J., Schiller, D., Schoenbaum, G., Phelps, E. A., & Daw, N. D. (2011).
Differential roles of human striatum and amygdala in associative learning.
*Nature Neuroscience, 14*(10), 1250–1252. https://doi.org/10.1038/nn.2904

Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational
modeling of behavioral data. *eLife, 8*, e49547. https://doi.org/10.7554/eLife.49547
