# Compute budget

Measured on Day 2 with `scripts/00_time_fits.py` (Windows 11, 22 logical cores,
JAX 0.10.2 CPU, float64, one XLA thread per process). Times are medians of 5
repeats, after JIT compilation; the full table is `results/tables/timing.csv`.

## Single multi-start fit, 160 trials (moderate schedule)

| Model | 10 starts (s) | 20 starts (s) | mean f-evals / start | 4-fold CV × 10 starts (s) | first call incl. compile (s) |
|---|---|---|---|---|---|
| Fixed RL | 0.029 | 0.065 | 12 | 0.14 | 0.14 |
| Asymmetric RL | 0.046 | 0.097 | 21 | 0.19 | 0.14 |
| PH hybrid | 0.096 | 0.187 | 41 | 0.36 | 0.15 |
| Bayesian CP | 0.040 | 0.078 | 16 | 0.16 | 0.14 |
| Perseveration | 0.042 | 0.076 | 18 | 0.15 | 0.10 |
| Random | 0 | 0 | — | 0.001 | — |

One optimisation therefore costs roughly 3–10 ms. Compilation happens once per
(model, trial count) per process; masks change between folds without
recompiling because array shapes are fixed.

## Projection at the locked design

| Stage | Optimisations | Projected single-core time |
|---|---|---|
| Parameter recovery (4 models × 3 trial counts × 3 conditions × 50 agents × 10 starts) | 18,000 | ≈ 2 min |
| Model recovery (4 generating × 4 primary fitted × 3 conditions × 50 agents × 4 folds × 10 starts) | 96,000 | ≈ 8 min |
| + benchmarks fitted in model recovery (perseveration; random is free) | +24,000 | ≈ +2 min |

Projections scale linearly with agents, so 100 agents per cell roughly doubles
them. Jobs run in parallel across agents with joblib (21 worker processes), so
wall-clock time is dominated by process start-up and compilation.

## Decision

The projection is two orders of magnitude below the 8-hour threshold, so the
Day 2 stop rule (reduce to 30 agents per cell) does not apply. Plan:

1. First pass at 50 agents per cell (Day 4), as specified.
2. Final figures at **100 agents per cell** for both parameter and model recovery.
   Because seeds are indexed by agent, agents 0–49 of the final run are identical
   to the first pass; only agents 50–99 are new, and existing fit files are reused.

## Actual run times

| Run | Fits | Wall-clock (21 workers) |
|---|---|---|
| Parameter recovery, first pass (50 agents/cell, 1,800 agents × 10 starts) | 1,800 | 66 s |
| Parameter recovery, final (100 agents/cell; agents 50–99 new) | +1,800 | 77 s |
| Model recovery, first pass (600 agents × 6 fitted models × 4 folds × 10 starts) | 3,600 CV runs | ≈ 5 min |
| Model recovery, final (1,200 agents; agents 50–99 new) | +3,600 CV runs | 313 s |

Wall-clock times are within a small factor of the single-core projection
divided by the number of workers; the remainder is worker start-up and JIT
compilation (once per model × trial count per worker).
