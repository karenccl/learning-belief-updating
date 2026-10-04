"""POST-HOC sensitivity analysis (decided after seeing the locked results; not pre-registered).

Question: do the conclusions change when the runaway ridges are removed by bounding
    beta = 1e-4 + (20 - 1e-4) * logistic(theta)       (beta <= 20; all models)
    h    = 0.5 * logistic(theta)                        (Bayesian hazard in (0, 0.5))?
True simulation values (beta <= 8, h <= 0.3) lie well inside the bounds.

The same simulated agents, CV folds and optimiser starting values as the locked analysis are
used (seed streams are those of the locked stages); only the parameter transforms differ.
Fits go to results/fits/*_bounded/. The locked analysis remains the primary result.

Outputs (results/tables/): *_bounded.csv versions of the recovery tables (via
07_collect_tables.py), demo_cv_summary_bounded.csv, bayes_h_ridge.csv, and
sensitivity_{parameter_recovery,model_recovery,catastrophic_folds,demo_selection}.csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import TABLES, load_config  # noqa: E402
from src.models import ALL_MODELS, PRIMARY_MODELS  # noqa: E402
from src.recovery import (  # noqa: E402
    collect_model_recovery, run_model_recovery_agent, run_parallel, run_parameter_recovery_agent,
)
from src.task import CONDITIONS  # noqa: E402

HERE = Path(__file__).resolve().parent
CATASTROPHIC = -2.0  # held-out LL per trial below this = catastrophically overconfident prediction


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bayes_ridge_table() -> pd.DataFrame:
    """Classify locked Bayesian fits: interior, graded-WSLS ridge (h -> 0.5, beta -> inf), alternation (h > 0.5)."""
    a = pd.read_csv(TABLES / "parameter_recovery_agents.csv")
    b = a[a.model == "bayes_cp"].copy()
    b["one_minus_2h"] = 1 - 2 * b.est_h
    b["beta_x_1m2h"] = b.est_beta * b.one_minus_2h  # the identified product on the ridge
    b["wsls_logit_strength"] = 0.36 * b.beta_x_1m2h  # first-order logit of stay after win vs switch after loss
    b["gain_vs_random_nats"] = b.random_nll - b.loss
    b["regime"] = np.select(
        [(b.est_h - 0.5).abs() < 1e-6, (b.est_h - 0.5).abs() < 0.02, b.est_h > 0.52],
        ["exactly_random_h_0.5", "wsls_ridge_h_near_0.5", "alternation_h_above_0.5"], "interior")
    cols = ["n_trials", "condition", "agent", "true_h", "true_beta", "est_h", "est_beta", "one_minus_2h",
            "beta_x_1m2h", "wsls_logit_strength", "gain_vs_random_nats", "near_random", "regime"]
    b[cols].to_csv(TABLES / "bayes_h_ridge.csv", index=False)
    return b


def compare_parameter_recovery() -> pd.DataFrame:
    key = ["model", "parameter", "n_trials", "condition"]
    cols = ["spearman_rho", "pearson_r_high_beta", "median_ae", "prop_boundary"]
    lo = pd.read_csv(TABLES / "parameter_recovery_metrics.csv")[key + cols]
    bo = pd.read_csv(TABLES / "parameter_recovery_metrics_bounded.csv")[key + cols]
    m = lo.merge(bo, on=key, suffixes=("_locked", "_bounded"))
    m.to_csv(TABLES / "sensitivity_parameter_recovery.csv", index=False)
    return m


def compare_model_recovery() -> pd.DataFrame:
    rows = []
    for variant, suffix in [("locked", ""), ("bounded", "_bounded")]:
        s = pd.read_csv(TABLES / f"model_recovery_summary{suffix}.csv")
        s["variant"] = variant
        rows.append(s)
    out = pd.concat(rows)
    out.to_csv(TABLES / "sensitivity_model_recovery.csv", index=False)
    return out


def compare_catastrophic(locked: pd.DataFrame, bounded: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for variant, d in [("locked", locked), ("bounded", bounded)]:
        for m, g in d.groupby("fitted_model"):
            rows.append({"dataset": label, "variant": variant, "fitted_model": m, "n_scores": len(g),
                         "n_catastrophic": int((g.heldout_ll_per_trial < CATASTROPHIC).sum()),
                         "min_score": g.heldout_ll_per_trial.min(),
                         "mean_score": g.heldout_ll_per_trial.mean(),
                         "median_score": g.heldout_ll_per_trial.median()})
    return pd.DataFrame(rows)


def demo_selection(summ_locked: pd.DataFrame, summ_bounded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, s in [("locked", summ_locked), ("bounded", summ_bounded)]:
        s = s[s.scheme == "leave_block_out_masked"]
        for gen, g in s.groupby("generating_model"):
            rows.append({"variant": variant, "generating_model": gen,
                         "selected_by_mean": g.loc[g["mean"].idxmax(), "fitted_model"],
                         "best_by_median": g.loc[g["median"].idxmax(), "fitted_model"],
                         "true_model_mean": float(g.loc[g.fitted_model == gen, "mean"].iloc[0]),
                         "true_model_median": float(g.loc[g.fitted_model == gen, "median"].iloc[0])})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "sensitivity_demo_selection.csv", index=False)
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--demo-agents", type=int, default=cfg["predictive_checks"]["demo_agents"])
    ap.add_argument("--n-jobs", type=int, default=cfg["compute"]["n_jobs"])
    args = ap.parse_args()
    t0 = time.perf_counter()

    pr = [(cfg, m, c, n, a, "bounded") for m in PRIMARY_MODELS for n in cfg["recovery"]["parameter"]["trial_counts"]
          for c in CONDITIONS for a in range(args.agents)]
    mr = [(cfg, m, c, a, tuple(ALL_MODELS), "model_recovery", None, "bounded") for m in PRIMARY_MODELS
          for c in CONDITIONS for a in range(args.agents)]
    demo = [(cfg, m, "moderate", a, tuple(ALL_MODELS), "demo", 160, "bounded") for m in PRIMARY_MODELS
            for a in range(args.demo_agents)]
    for name, func, jobs in [("parameter recovery", run_parameter_recovery_agent, pr),
                             ("model recovery", run_model_recovery_agent, mr),
                             ("demo CV", run_model_recovery_agent, demo)]:
        t = time.perf_counter()
        run_parallel(func, jobs, n_jobs=args.n_jobs, verbose=0)
        print(f"{name} (bounded): {len(jobs)} agents, {time.perf_counter() - t:.0f}s", flush=True)

    collect = _load_script("07_collect_tables")
    collect.parameter_recovery_tables("parameter_recovery_bounded", "_bounded")
    collect.model_recovery_tables(cfg, "model_recovery_bounded", "_bounded")

    demo_b = collect_model_recovery("demo_bounded")
    demo_b = demo_b[demo_b.agent < args.demo_agents]
    demo_b.to_csv(TABLES / "demo_cv_scores_bounded.csv", index=False)
    summ_b = _load_script("06_predictive_checks").summarise_scores(demo_b, "leave_block_out_masked")
    summ_b.to_csv(TABLES / "demo_cv_summary_bounded.csv", index=False)

    ridge = bayes_ridge_table()
    print("\nLocked Bayesian fits by regime:\n", ridge.regime.value_counts().to_string())
    print(ridge.groupby("regime")[["est_h", "est_beta", "beta_x_1m2h", "gain_vs_random_nats", "true_beta"]]
          .median().round(3).to_string())

    cat = pd.concat([
        compare_catastrophic(pd.read_csv(TABLES / "model_recovery_scores.csv"),
                             pd.read_csv(TABLES / "model_recovery_scores_bounded.csv"), "model_recovery"),
        compare_catastrophic(pd.read_csv(TABLES / "demo_cv_scores.csv"), demo_b, "demo")])
    cat.to_csv(TABLES / "sensitivity_catastrophic_folds.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nCatastrophic held-out scores (< -2 LL/trial):\n",
          cat.pivot_table(index=["dataset", "fitted_model"], columns="variant",
                          values=["n_catastrophic", "min_score"]).round(3).to_string())
    mr_cmp = compare_model_recovery()
    print("\nModel recovery (pooled):\n", mr_cmp[mr_cmp.condition == "all"].pivot_table(
        index="generating_model", columns="variant",
        values=["accuracy", "accuracy_excluding_ties", "tie_rate", "best_raw_correct"]).round(3).to_string())
    print("\nDemo selection:\n", demo_selection(pd.read_csv(TABLES / "demo_cv_summary.csv"), summ_b).to_string())
    prc = compare_parameter_recovery()
    print("\nParameter recovery at 160 trials (Spearman):\n",
          prc[prc.n_trials == 160].pivot_table(index=["model", "parameter"], columns="condition",
                                               values=["spearman_rho_locked", "spearman_rho_bounded"])
          .round(2).to_string())
    print(f"\nwall {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
