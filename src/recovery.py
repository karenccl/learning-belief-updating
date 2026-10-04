"""Parameter recovery and model recovery: per-agent jobs, collection, and metrics.

One simulated agent = one independently fitted participant. Every job writes one
JSON per (agent, fitted model) and skips files that already exist, so interrupted
runs resume automatically.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import FITS, ROOT
from .cross_validation import cross_validate
from .fit import fit_arrays, save_json
from .models import ALL_MODELS, MODEL_PARAMS, PRIMARY_MODELS
from .seeds import agent_rng, agent_stream
from .simulate import simulate_model
from .task import environment_for_condition


# ---------------------------------------------------------------------------
# Simulation of one agent
# ---------------------------------------------------------------------------
def sample_parameters(model: str, rng: np.random.Generator, ranges: dict) -> dict:
    """Uniform over the configured ranges, except beta which is log-uniform."""
    out = {}
    for name in MODEL_PARAMS[model]:
        lo, hi = ranges[name]
        if name == "beta":
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            out[name] = float(rng.uniform(lo, hi))
    return out


def simulate_agent(cfg: dict, stage: str, model: str, condition: str, n_trials: int, agent: int,
                   params: dict | None = None):
    seed = cfg["master_seed"]
    key = (stage, model, condition, n_trials, agent)
    if params is None:
        params = sample_parameters(model, agent_rng(seed, *key, "params"), cfg["parameter_ranges"])
    env = environment_for_condition(condition, n_trials, cfg["task"])
    df = simulate_model(model, params, env, seed=agent_stream(seed, *key, "simulation"),
                        participant_id=f"{model}_{condition}_{n_trials}_{agent:03d}",
                        cv_block_length=cfg["task"]["cv_block_length"])
    return params, df


def behaviour_summary(df: pd.DataFrame) -> dict:
    return {"accuracy": float((df.choice == df.advantageous_option).mean()),
            "p_repeat": float((df.choice.diff() == 0).iloc[1:].mean())}


def fit_path(stage: str, generating: str, condition: str, n_trials: int, agent: int, fitted: str) -> Path:
    return FITS / stage / generating / f"{condition}_{n_trials}_{agent:03d}_{fitted}.json"


# ---------------------------------------------------------------------------
# Per-agent jobs (run in worker processes)
# ---------------------------------------------------------------------------
def output_stage(stage: str, variant: str) -> str:
    """Fits of a sensitivity variant go to their own directory; seeds stay those of ``stage``,
    so the variant refits exactly the same simulated agents from the same starting values."""
    return stage if variant == "locked" else f"{stage}_{variant}"


def run_parameter_recovery_agent(cfg: dict, model: str, condition: str, n_trials: int, agent: int,
                                 variant: str = "locked") -> str:
    path = fit_path(output_stage("parameter_recovery", variant), model, condition, n_trials, agent, model)
    if path.exists():
        return "skipped"
    t0 = time.perf_counter()
    params, df = simulate_agent(cfg, "parameter_recovery", model, condition, n_trials, agent)
    fit = fit_arrays(
        model, df.choice.to_numpy(), df.reward.to_numpy(float),
        n_starts=cfg["fitting"]["n_starts_recovery"],
        rng=agent_rng(cfg["master_seed"], "parameter_recovery", model, condition, n_trials, agent,
                      "fit", ALL_MODELS.index(model)),
        start_sd=cfg["fitting"]["start_sd"], grad_norm_tol=cfg["fitting"]["grad_norm_tol"],
        variant=variant,
    )
    fit.update({"stage": output_stage("parameter_recovery", variant), "generating_model": model, "fitted_model": model,
                "condition": condition, "n_trials": n_trials, "agent": agent, "true_params": params,
                **behaviour_summary(df), "elapsed_s": time.perf_counter() - t0})
    save_json(fit, path)
    return "done"


def run_model_recovery_agent(cfg: dict, model: str, condition: str, agent: int,
                             fitted_models=ALL_MODELS, stage: str = "model_recovery",
                             n_trials: int | None = None, variant: str = "locked") -> str:
    n_trials = n_trials or cfg["recovery"]["model"]["trial_count"]
    out = output_stage(stage, variant)
    todo = [m for m in fitted_models if not fit_path(out, model, condition, n_trials, agent, m).exists()]
    if not todo:
        return "skipped"
    params, df = simulate_agent(cfg, stage, model, condition, n_trials, agent)
    c, r = df.choice.to_numpy(), df.reward.to_numpy(float)
    for fitted in todo:
        t0 = time.perf_counter()
        cv = cross_validate(
            fitted, c, r, n_starts=cfg["fitting"]["n_starts_recovery"],
            block_length=cfg["task"]["cv_block_length"],
            seed_fn=lambda k, f=fitted: agent_rng(cfg["master_seed"], stage, model, condition, n_trials,
                                                  agent, "fit", ALL_MODELS.index(f), k),
            start_sd=cfg["fitting"]["start_sd"], grad_norm_tol=cfg["fitting"]["grad_norm_tol"],
            variant=variant,
        )
        cv.update({"stage": out, "variant": variant, "generating_model": model, "fitted_model": fitted,
                   "condition": condition, "n_trials": n_trials, "agent": agent, "true_params": params,
                   **behaviour_summary(df), "elapsed_s": time.perf_counter() - t0})
        save_json(cv, fit_path(out, model, condition, n_trials, agent, fitted))
    return "done"


def run_parallel(func, jobs: list[tuple], n_jobs: int = -2, verbose: int = 5) -> list:
    """Run ``func(*job)`` over jobs with joblib/loky, one process per core."""
    from joblib import Parallel, delayed

    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=1)(delayed(func)(*j) for j in jobs)


# ---------------------------------------------------------------------------
# Collection (fits -> tidy tables)
# ---------------------------------------------------------------------------
def _load_all(stage: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((FITS / stage).rglob("*.json"))]


def collect_parameter_recovery(stage: str = "parameter_recovery") -> pd.DataFrame:
    rows = []
    for f in _load_all(stage):
        row = {k: f[k] for k in ("generating_model", "condition", "n_trials", "agent", "loss", "random_nll",
                                 "converged", "failed", "near_random", "any_boundary", "n_converged_starts",
                                 "prop_starts_within_0.1", "accuracy", "p_repeat", "elapsed_s")}
        row["model"] = row.pop("generating_model")
        for name in MODEL_PARAMS[row["model"]]:
            row[f"true_{name}"] = f["true_params"][name]
            row[f"est_{name}"] = f["params"][name]
            row[f"boundary_{name}"] = f["boundary"][name]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "n_trials", "condition", "agent"]).reset_index(drop=True)


def parameter_recovery_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, n, cond), g in df.groupby(["model", "n_trials", "condition"]):
        for name in MODEL_PARAMS[model]:
            ok = g[~g.failed]
            t, e = ok[f"true_{name}"].to_numpy(), ok[f"est_{name}"].to_numpy()
            err = e - t
            row = {
                "model": model, "parameter": name, "n_trials": n, "condition": cond, "n_agents": len(g),
                "pearson_r": stats.pearsonr(t, e)[0] if len(t) > 2 else np.nan,
                "spearman_rho": stats.spearmanr(t, e)[0] if len(t) > 2 else np.nan,
                "bias": err.mean(), "median_bias": np.median(err), "mae": np.abs(err).mean(),
                "median_ae": np.median(np.abs(err)), "rmse": np.sqrt((err ** 2).mean()),
                "prop_converged": g.converged.mean(), "prop_failed": g.failed.mean(),
                "prop_boundary": g[f"boundary_{name}"].mean(), "prop_near_random": g.near_random.mean(),
            }
            if name == "beta":
                # beta is sampled log-uniformly and ML estimates can run away (alpha-beta degeneracy),
                # so raw-scale means are dominated by a few agents; also report the log scale.
                lerr = np.log(e) - np.log(t)
                row.update({"pearson_r_log": stats.pearsonr(np.log(t), np.log(e))[0],
                            "bias_log": lerr.mean(), "mae_log": np.abs(lerr).mean(),
                            "median_bias_log": np.median(lerr)})
            # "Expected results" benchmark: agents with true beta >= 2 carry more information
            hb = ok[ok["true_beta"] >= 2] if "true_beta" in ok else ok.iloc[:0]
            row["n_high_beta"] = len(hb)
            row["spearman_rho_high_beta"] = (stats.spearmanr(hb[f"true_{name}"], hb[f"est_{name}"])[0]
                                             if len(hb) > 2 else np.nan)
            row["pearson_r_high_beta"] = (
                stats.pearsonr(np.log(hb.true_beta), np.log(hb.est_beta))[0] if name == "beta" and len(hb) > 2
                else stats.pearsonr(hb[f"true_{name}"], hb[f"est_{name}"])[0] if len(hb) > 2 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def near_random_by_beta(df: pd.DataFrame, bins=(0.5, 1, 2, 4, 8)) -> pd.DataFrame:
    d = df.copy()
    d["true_beta_bin"] = pd.cut(d["true_beta"], bins=list(bins), include_lowest=True).astype(str)
    return (d.groupby(["model", "n_trials", "true_beta_bin"], observed=True)
             .agg(n=("near_random", "size"), prop_near_random=("near_random", "mean"))
             .reset_index())


def collect_model_recovery(stage: str = "model_recovery") -> pd.DataFrame:
    """Long table: one row per (agent, fitted model) with its held-out score and diagnostics."""
    rows = []
    for f in _load_all(stage):
        folds = f["folds"]
        rows.append({
            "generating_model": f["generating_model"], "fitted_model": f["fitted_model"],
            "condition": f["condition"], "n_trials": f["n_trials"], "agent": f["agent"],
            "heldout_ll": f["heldout_ll"], "heldout_ll_per_trial": f["heldout_ll_per_trial"],
            "n_failed_folds": f["n_failed_folds"],
            "n_unconverged_folds": sum(not x["converged"] for x in folds),
            "n_boundary_folds": sum(x["any_boundary"] for x in folds),
            "n_beta_runaway_folds": sum(bool(x["boundary"].get("beta", False)) for x in folds),
            "n_near_random_folds": sum(x["near_random"] for x in folds),
            "accuracy": f["accuracy"],
            **{f"true_{k}": v for k, v in f["true_params"].items()},
        })
    return pd.DataFrame(rows)


def model_recovery_winners(long: pd.DataFrame, candidates=PRIMARY_MODELS, tie_threshold: float = 0.01) -> pd.DataFrame:
    """One row per agent: scores per candidate, winner, runner-up margin, tie flag."""
    idx = ["generating_model", "condition", "n_trials", "agent"]
    wide = long.pivot_table(index=idx, columns="fitted_model", values="heldout_ll_per_trial").reset_index()
    wide.columns.name = None
    scores = wide[list(candidates)].to_numpy()
    order = np.argsort(-scores, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(len(wide))
    wide["best_model"] = np.array(candidates)[best]
    wide["runner_up"] = np.array(candidates)[second]
    wide["margin_per_trial"] = scores[rows, best] - scores[rows, second]
    wide["tie"] = wide["margin_per_trial"] < tie_threshold
    wide["selected"] = np.where(wide["tie"], "tie", wide["best_model"])
    wide["correct"] = wide["selected"] == wide["generating_model"]
    return wide


def confusion_matrix(winners: pd.DataFrame, candidates=PRIMARY_MODELS, by_condition: bool = True) -> pd.DataFrame:
    cols = list(candidates) + ["tie"]
    groups = ["condition"] if by_condition else []
    out = []
    for key, g in (winners.groupby(groups) if groups else [("all", winners)]):
        counts = pd.crosstab(g.generating_model, g.selected).reindex(index=list(candidates), columns=cols,
                                                                      fill_value=0)
        long = counts.stack().rename("count").reset_index()
        long["row_total"] = long.groupby("generating_model")["count"].transform("sum")
        long["proportion"] = long["count"] / long["row_total"]
        long["condition"] = key[0] if isinstance(key, tuple) else key
        out.append(long)
    return pd.concat(out, ignore_index=True)
