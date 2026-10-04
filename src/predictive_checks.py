"""Parametric predictive checks (model simulations from ML-fitted parameters).

In the simulation-only core, the "observed" data are the 100-agent demonstration
dataset simulated from each generating model (moderate condition, 160 trials).
The true parameters are not used anywhere in this step.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .config import FITS
from .fit import fit_arrays, save_json
from .models import ALL_MODELS, make_agent
from .recovery import simulate_agent
from .seeds import agent_rng, agent_stream
from .simulate import simulate_arrays
from .task import environment_for_condition

PRE, POST = 10, 20  # window: 10 trials before to 20 after a reversal (offset 0 = first new trial)
STATS = ("accuracy", "win_stay", "lose_switch", "p_repeat")


def reversal_window(condition_interval: int | None) -> tuple[int, int]:
    """Truncate the window so it never crosses a neighbouring reversal (volatile: 10 before, 19 after)."""
    if condition_interval is not None and condition_interval <= PRE + POST:
        return PRE, min(POST, condition_interval - 1)
    return PRE, POST


def behavioural_summaries(choices, rewards, advantageous, reversal, pre: int = PRE, post: int = POST) -> dict:
    """Per-agent summaries. Curves are NaN where a window would cross another reversal or the task edge."""
    c = np.asarray(choices)
    r = np.asarray(rewards)
    correct = (c == np.asarray(advantageous)).astype(float)
    n = len(c)
    rev = np.flatnonzero(np.asarray(reversal))  # array index of first trial with new contingencies
    bounds = np.concatenate([[0], rev, [n]])
    offsets = np.arange(-pre, post)
    curve = np.full((len(rev), len(offsets)), np.nan)
    for i, k in enumerate(rev):
        lo, hi = bounds[i], bounds[i + 2]  # previous block start, next reversal
        for j, o in enumerate(offsets):
            if lo <= k + o < hi:
                curve[i, j] = correct[k + o]
    block_start = bounds[np.searchsorted(bounds, np.arange(n), side="right") - 1]
    position = np.arange(n) - block_start
    stay = c[1:] == c[:-1]
    prev_r = r[:-1]
    with warnings.catch_warnings():  # offsets outside every window are NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        rev_curve = np.nanmean(curve, axis=0) if len(rev) else np.full(len(offsets), np.nan)
    return {
        "reversal_curve": rev_curve,
        "block_curve": np.array([correct[position == p].mean() for p in range(position.max() + 1)]),
        "accuracy": correct.mean(),
        "win_stay": stay[prev_r == 1].mean() if (prev_r == 1).any() else np.nan,
        "lose_switch": (~stay[prev_r == 0]).mean() if (prev_r == 0).any() else np.nan,
        "p_repeat": stay.mean(),
    }


def fitted_learning_rates(model: str, params: dict, choices, rewards) -> np.ndarray | None:
    """Trialwise learning rate implied by fitted parameters on the observed sequence."""
    if model not in ("asymmetric_rl", "hybrid_ph", "fixed_rl"):
        return None
    agent = make_agent(model, params)
    out = np.empty(len(choices))
    prev_s = 0.0
    for t, (a, r) in enumerate(zip(choices, rewards)):
        agent.logit(t, prev_s)
        out[t] = agent.update(int(a), int(r))["alpha_t"]
        prev_s = 2.0 * a - 1.0
    return out


def ppc_path(generating: str, agent: int, fitted: str):
    return FITS / "demo_ppc" / generating / f"moderate_160_{agent:03d}_{fitted}.json"


def run_ppc_agent(cfg: dict, generating: str, agent: int, fitted_models: tuple[str, ...]) -> str:
    """Fit each model to one demo agent (full data, 20 starts), simulate replicates, save summaries."""
    todo = [m for m in fitted_models if not ppc_path(generating, agent, m).exists()]
    if not todo:
        return "skipped"
    seed = cfg["master_seed"]
    n_trials, cond = 160, "moderate"
    key = ("demo", generating, cond, n_trials, agent)
    _, df = simulate_agent(cfg, *key[:4], agent)
    env = environment_for_condition(cond, n_trials, cfg["task"])
    c, r = df.choice.to_numpy(), df.reward.to_numpy()
    interval = cfg["task"]["conditions"][cond].get("interval")
    pre, post = reversal_window(interval)
    obs = behavioural_summaries(c, r, env["advantageous_option"], env["reversal"], pre, post)
    n_rep = cfg["predictive_checks"]["replicates_per_agent"]
    for fitted in todo:
        fit = fit_arrays(fitted, c, r.astype(float), n_starts=cfg["fitting"]["n_starts_single"],
                         rng=agent_rng(seed, *key, "fit", 200 + ALL_MODELS.index(fitted)),
                         start_sd=cfg["fitting"]["start_sd"], grad_norm_tol=cfg["fitting"]["grad_norm_tol"])
        reps = {"reversal_curve": [], "block_curve": [], **{s: [] for s in STATS}}
        for rep in range(n_rep):
            sc, sr = simulate_arrays(fitted, fit["params"], env,
                                     seed=agent_stream(seed, *key, "simulation", 1 + ALL_MODELS.index(fitted), rep))
            s = behavioural_summaries(sc, sr, env["advantageous_option"], env["reversal"], pre, post)
            for k in reps:
                reps[k].append(s[k])
        lr = fitted_learning_rates(fitted, fit["params"], c, r)
        out = {
            "generating_model": generating, "fitted_model": fitted, "agent": agent,
            "params": fit["params"], "loss": fit["loss"], "converged": fit["converged"],
            "near_random": fit["near_random"], "any_boundary": fit["any_boundary"],
            "window": [pre, post],
            "observed": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in obs.items()},
            "replicates": {k: np.asarray(v).tolist() for k, v in reps.items()},
            "fitted_alpha_t": None if lr is None else lr.tolist(),
            "reversal": env["reversal"].tolist(),
        }
        save_json(out, ppc_path(generating, agent, fitted))
    return "done"


def _load(generating: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((FITS / "demo_ppc" / generating).glob("*.json"))]


def ppc_tables(generating_models, interval_q=(0.05, 0.95)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-level observed statistics vs the distribution of group-level replicate statistics.

    Replicate r of the group is the mean over agents of each agent's r-th replicate, so the
    interval reflects the sampling variability of a 100-agent dataset under the fitted model.
    """
    curves, blocks, summ, lrs = [], [], [], []
    for gen in generating_models:
        fits = _load(gen)
        for fitted in sorted({f["fitted_model"] for f in fits}):
            fs = [f for f in fits if f["fitted_model"] == fitted]
            pre, post = fs[0]["window"]
            offsets = np.arange(-pre, post)
            obs_curve = np.nanmean([f["observed"]["reversal_curve"] for f in fs], axis=0)
            sim_curve = np.nanmean([f["replicates"]["reversal_curve"] for f in fs], axis=0)  # reps x offsets
            q = np.nanquantile(sim_curve, interval_q, axis=0)
            curves.append(pd.DataFrame({"generating_model": gen, "fitted_model": fitted, "offset": offsets,
                                        "observed": obs_curve, "sim_mean": sim_curve.mean(0),
                                        "sim_q05": q[0], "sim_q95": q[1], "n_agents": len(fs)}))
            obs_b = np.nanmean([f["observed"]["block_curve"] for f in fs], axis=0)
            sim_b = np.nanmean([f["replicates"]["block_curve"] for f in fs], axis=0)
            qb = np.nanquantile(sim_b, interval_q, axis=0)
            blocks.append(pd.DataFrame({"generating_model": gen, "fitted_model": fitted,
                                        "position": np.arange(len(obs_b)), "observed": obs_b,
                                        "sim_mean": sim_b.mean(0), "sim_q05": qb[0], "sim_q95": qb[1]}))
            for s in STATS:
                o = np.nanmean([f["observed"][s] for f in fs])
                sim = np.nanmean([f["replicates"][s] for f in fs], axis=0)
                agent_cover = np.mean([
                    np.nanquantile(f["replicates"][s], interval_q[0]) <= f["observed"][s]
                    <= np.nanquantile(f["replicates"][s], interval_q[1]) for f in fs])
                summ.append({"generating_model": gen, "fitted_model": fitted, "statistic": s, "observed": o,
                             "sim_mean": sim.mean(), "sim_q05": np.quantile(sim, interval_q[0]),
                             "sim_q95": np.quantile(sim, interval_q[1]),
                             "predictive_p": float(np.mean(sim >= o)),
                             "agent_interval_coverage": agent_cover, "n_agents": len(fs)})
            if fs[0]["fitted_alpha_t"] is not None:
                rev = np.flatnonzero(fs[0]["reversal"])
                a = np.array([f["fitted_alpha_t"] for f in fs])
                for o in offsets:
                    idx = [k + o for k in rev if 0 <= k + o < a.shape[1]]
                    lrs.append({"generating_model": gen, "fitted_model": fitted, "offset": int(o),
                                "mean_alpha_t": float(a[:, idx].mean())})
    return pd.concat(curves), pd.concat(blocks), pd.DataFrame(summ), pd.DataFrame(lrs)


def fc_path(generating: str, agent: int, fitted: str):
    return FITS / "demo_forward_chaining" / generating / f"moderate_160_{agent:03d}_{fitted}.json"


def run_forward_chaining_agent(cfg: dict, generating: str, agent: int, fitted_models: tuple[str, ...]) -> str:
    """Sensitivity check: forward-chaining CV on one demo agent (no information flow from held-out data)."""
    from .cross_validation import forward_chaining

    todo = [m for m in fitted_models if not fc_path(generating, agent, m).exists()]
    if not todo:
        return "skipped"
    key = ("demo", generating, "moderate", 160, agent)
    _, df = simulate_agent(cfg, *key[:4], agent)
    c, r = df.choice.to_numpy(), df.reward.to_numpy(float)
    for fitted in todo:
        fc = forward_chaining(fitted, c, r, n_starts=cfg["fitting"]["n_starts_recovery"],
                              block_length=cfg["task"]["cv_block_length"],
                              seed_fn=lambda k, f=fitted: agent_rng(cfg["master_seed"], *key, "fit",
                                                                    300 + ALL_MODELS.index(f), k))
        fc.update({"generating_model": generating, "fitted_model": fitted, "agent": agent, "folds": None})
        save_json(fc, fc_path(generating, agent, fitted))
    return "done"
