"""Collect fit JSONs into summary tables under results/tables/.

Figures (scripts/05_make_figures.py) are generated only from these tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.config import FITS, TABLES, load_config  # noqa: E402
from src.models import MODEL_PARAMS, PRIMARY_MODELS  # noqa: E402
from src.recovery import (  # noqa: E402
    collect_model_recovery, collect_parameter_recovery, confusion_matrix, model_recovery_winners,
    near_random_by_beta, parameter_recovery_metrics,
)


def _mean_ci(x: pd.Series) -> pd.Series:
    x = x.dropna()
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    h = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
    return pd.Series({"n": n, "mean": x.mean(), "se": se, "ci_low": x.mean() - h, "ci_high": x.mean() + h,
                      "median": x.median(), "q25": x.quantile(0.25), "q75": x.quantile(0.75)})


def parameter_recovery_tables(stage: str = "parameter_recovery", suffix: str = "") -> None:
    df = collect_parameter_recovery(stage)
    df.to_csv(TABLES / f"parameter_recovery_agents{suffix}.csv", index=False)
    parameter_recovery_metrics(df).to_csv(TABLES / f"parameter_recovery_metrics{suffix}.csv", index=False)
    near_random_by_beta(df).to_csv(TABLES / f"near_random_by_beta{suffix}.csv", index=False)
    rows = []  # correlations among recovered parameters (trade-offs), per model x trial count x condition
    for (m, n, c), g in df.groupby(["model", "n_trials", "condition"]):
        names = MODEL_PARAMS[m]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ea = np.log(g[f"est_{a}"]) if a == "beta" else g[f"est_{a}"]
                eb = np.log(g[f"est_{b}"]) if b == "beta" else g[f"est_{b}"]
                rows.append({"model": m, "n_trials": n, "condition": c, "param_a": a, "param_b": b,
                             "spearman_est": stats.spearmanr(ea, eb)[0],
                             "spearman_true": stats.spearmanr(g[f"true_{a}"], g[f"true_{b}"])[0]})
    pd.DataFrame(rows).to_csv(TABLES / f"recovered_parameter_correlations{suffix}.csv", index=False)
    print(f"parameter recovery{suffix}: {len(df)} agents")


def model_recovery_tables(cfg: dict, stage: str = "model_recovery", suffix: str = "") -> None:
    long = collect_model_recovery(stage)
    if long.empty:
        return
    long.to_csv(TABLES / f"model_recovery_scores{suffix}.csv", index=False)
    tie = cfg["recovery"]["model"]["tie_threshold_per_trial"]
    win = model_recovery_winners(long, tie_threshold=tie)
    win.to_csv(TABLES / f"model_recovery_winners{suffix}.csv", index=False)
    conf = pd.concat([confusion_matrix(win, by_condition=True), confusion_matrix(win, by_condition=False)])
    conf.to_csv(TABLES / f"model_recovery_confusion{suffix}.csv", index=False)

    summ = []
    for cond, g in [("all", win)] + list(win.groupby("condition")):
        for gen, h in [("all", g)] + list(g.groupby("generating_model")):
            summ.append({"condition": cond, "generating_model": gen, "n_agents": len(h),
                         "accuracy": h.correct.mean(),
                         "accuracy_excluding_ties": h.loc[~h.tie, "correct"].mean(),
                         "tie_rate": h.tie.mean(),
                         "best_raw_correct": (h.best_model == h.generating_model).mean(),
                         "median_margin": h.margin_per_trial.median()})
    pd.DataFrame(summ).to_csv(TABLES / f"model_recovery_summary{suffix}.csv", index=False)

    # Pairwise confusability: P(select j | i) + P(select i | j), from pooled confusion
    pair = []
    for c_ in ["all"] + sorted(win.condition.unique()):
        p = conf[conf.condition == c_].pivot(index="generating_model", columns="selected", values="proportion")
        for i, a in enumerate(PRIMARY_MODELS):
            for b in PRIMARY_MODELS[i + 1:]:
                pair.append({"condition": c_, "model_a": a, "model_b": b, "p_b_given_a": p.loc[a, b],
                             "p_a_given_b": p.loc[b, a], "confusability": p.loc[a, b] + p.loc[b, a]})
    pd.DataFrame(pair).to_csv(TABLES / f"model_recovery_pairwise_confusability{suffix}.csv", index=False)

    # Held-out score summaries (all fitted models incl. benchmarks)
    s = (long.groupby(["generating_model", "condition", "fitted_model"])["heldout_ll_per_trial"]
         .apply(_mean_ci).unstack().reset_index())
    diag = (long.groupby(["generating_model", "condition", "fitted_model"])
            .agg(failed_folds=("n_failed_folds", "sum"), unconverged_folds=("n_unconverged_folds", "sum"),
                 boundary_folds=("n_boundary_folds", "sum"), beta_runaway_folds=("n_beta_runaway_folds", "sum"),
                 near_random_folds=("n_near_random_folds", "sum"), n_agents=("agent", "size")).reset_index())
    s.merge(diag, on=["generating_model", "condition", "fitted_model"]).to_csv(
        TABLES / f"model_recovery_heldout_scores{suffix}.csv", index=False)

    # Agent-level differences: fitted model minus the generating model
    true_score = long[long.fitted_model == long.generating_model][
        ["generating_model", "condition", "agent", "heldout_ll_per_trial"]].rename(
        columns={"heldout_ll_per_trial": "true_model_score"})
    d = long.merge(true_score, on=["generating_model", "condition", "agent"])
    d["diff_vs_true"] = d["heldout_ll_per_trial"] - d["true_model_score"]
    dd = (d[d.fitted_model != d.generating_model]
          .groupby(["generating_model", "condition", "fitted_model"])["diff_vs_true"]
          .apply(_mean_ci).unstack().reset_index())
    dd["prop_beats_true"] = (d[d.fitted_model != d.generating_model]
                             .groupby(["generating_model", "condition", "fitted_model"])["diff_vs_true"]
                             .apply(lambda x: (x > 0).mean()).to_numpy())
    dd.to_csv(TABLES / f"model_recovery_score_differences{suffix}.csv", index=False)
    print(f"model recovery{suffix}: {win.shape[0]} agents, accuracy {win.correct.mean():.3f}, ties {win.tie.mean():.3f}")


def optimisation_diagnostics() -> None:
    rows = []
    pr = TABLES / "parameter_recovery_agents.csv"
    if pr.exists():
        df = pd.read_csv(pr)
        for (m, n, c), g in df.groupby(["model", "n_trials", "condition"]):
            rows.append({"stage": "parameter_recovery", "fitted_model": m, "generating_model": m,
                         "n_trials": n, "condition": c, "n_fits": len(g), "failure_rate": g.failed.mean(),
                         "unconverged_rate": (~g.converged).mean(), "boundary_rate": g.any_boundary.mean(),
                         "beta_runaway_rate": g.get("boundary_beta", pd.Series(dtype=bool)).mean(),
                         "near_random_rate": g.near_random.mean(),
                         "mean_prop_starts_within_0.1": g["prop_starts_within_0.1"].mean()})
    mr = TABLES / "model_recovery_scores.csv"
    if mr.exists():
        long = pd.read_csv(mr)
        for (m, c), g in long.groupby(["fitted_model", "condition"]):
            nf = 4 * len(g)
            rows.append({"stage": "model_recovery_cv_folds", "fitted_model": m, "generating_model": "all",
                         "n_trials": 160, "condition": c, "n_fits": nf,
                         "failure_rate": g.n_failed_folds.sum() / nf,
                         "unconverged_rate": g.n_unconverged_folds.sum() / nf,
                         "boundary_rate": g.n_boundary_folds.sum() / nf,
                         "beta_runaway_rate": g.n_beta_runaway_folds.sum() / nf,
                         "near_random_rate": g.n_near_random_folds.sum() / nf})
    pd.DataFrame(rows).to_csv(TABLES / "optimisation_diagnostics.csv", index=False)


def model_specification_table() -> None:
    spec = [
        ("random", "P(a=1) = 0.5", "-", "-"),
        ("perseveration", "P(a=1) = logistic(b + rho s_{t-1}); trial 1: 0.5", "b (unbounded), rho (-5, 5)", "-"),
        ("fixed_rl", "Q_a <- Q_a + alpha delta", "alpha (0,1), beta > 0", "alpha 0.05-0.80; beta 0.5-8 (log)"),
        ("asymmetric_rl", "Q_a <- Q_a + alpha^{+/-} delta (sign of delta)", "alpha+, alpha- (0,1), beta > 0",
         "alpha+/- 0.05-0.80; beta 0.5-8 (log)"),
        ("hybrid_ph", "alpha_t = kappa A_t; Q_a <- Q_a + alpha_t delta; A <- A + eta(|delta| - A)",
         "eta, kappa (0,1), beta > 0", "eta 0.05-0.80; kappa 0.10-0.90; beta 0.5-8 (log)"),
        ("bayes_cp", "b_t = (1-h) b'_{t-1} + h (1 - b'_{t-1}); Q from b_t with pH=.8, pL=.2; Bayes update",
         "h (0,1), beta > 0", "h 0.01-0.30; beta 0.5-8 (log)"),
    ]
    pd.DataFrame(spec, columns=["model", "update_rule", "free_parameters", "simulation_ranges"]).assign(
        choice_rule="P(a=1) = logistic(beta (Q1 - Q0)); rho = 0 in primary models").to_csv(
        TABLES / "model_specification.csv", index=False)


def main() -> None:
    cfg = load_config()
    TABLES.mkdir(parents=True, exist_ok=True)
    model_specification_table()
    if (FITS / "parameter_recovery").exists():
        parameter_recovery_tables()
    if (FITS / "model_recovery").exists():
        model_recovery_tables(cfg)
    # Post-hoc sensitivity analysis (bounded beta and h); see scripts/08_sensitivity_bounded.py
    if (FITS / "parameter_recovery_bounded").exists():
        parameter_recovery_tables("parameter_recovery_bounded", "_bounded")
    if (FITS / "model_recovery_bounded").exists():
        model_recovery_tables(cfg, "model_recovery_bounded", "_bounded")
    optimisation_diagnostics()


if __name__ == "__main__":
    main()
