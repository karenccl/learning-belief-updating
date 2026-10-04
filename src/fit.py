"""Multi-start maximum-likelihood fitting with the locked diagnostics.

Optimiser: scipy L-BFGS-B on unconstrained parameters, with the JIT-compiled
JAX negative log likelihood and its ``jax.grad`` gradient (``jac=True``).
"""
from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import BETA_MAX, MODEL_PARAMS, TRANSFORM_VARIANTS, _as_jax, nll_and_grad_fn, transform

LOG2 = float(np.log(2.0))
OPTIMISER_OPTIONS = {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6}


def boundary_flags(params: dict, variant: str = "locked") -> dict:
    """Locked boundary rules: (0,1) params <0.01 or >0.99; beta > 20; |rho| > 4.9.

    Bounded variant (post-hoc): beta within 1% of its cap; h < 0.01 or > 0.495.
    """
    flags = {}
    for name, value in params.items():
        kind = TRANSFORM_VARIANTS[variant][name]
        if kind == "bounded_beta":
            flags[name] = bool(value > 0.99 * BETA_MAX)
            continue
        if kind == "half_logistic":
            flags[name] = bool(value < 0.01 or value > 0.495)
            continue
        if kind == "logistic":
            flags[name] = bool(value < 0.01 or value > 0.99)
        elif kind == "softplus":
            flags[name] = bool(value > 20.0)
        elif kind == "tanh5":
            flags[name] = bool(abs(value) > 4.9)
        else:
            flags[name] = False
    return flags


def _objective(model_name, choices, rewards, mask, variant="locked"):
    vg = nll_and_grad_fn(model_name, variant)
    c, r, m = _as_jax(choices, rewards, mask)

    def f(theta):
        value, grad = vg(jnp.asarray(theta, dtype=jnp.float64), c, r, m)
        return float(value), np.asarray(grad, dtype=np.float64)

    return f


def fit_arrays(
    model_name: str,
    choices,
    rewards,
    score_mask=None,
    n_starts: int = 20,
    rng: np.random.Generator | None = None,
    start_sd: float = 1.5,
    grad_norm_tol: float = 1e-3,
    variant: str = "locked",
) -> dict:
    """Fit one model to one agent's arrays; return estimates, all starts and flags.

    ``variant`` selects the parameter transforms: "locked" (primary analysis) or
    "bounded" (post-hoc sensitivity: beta <= 20, Bayesian h < 0.5).
    """
    choices = np.asarray(choices, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    mask = np.ones(len(choices)) if score_mask is None else np.asarray(score_mask, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    names = MODEL_PARAMS[model_name]
    k = len(names)
    n_scored = float(mask.sum())
    random_nll = n_scored * LOG2

    if k == 0:  # random-choice benchmark: nothing to optimise
        return {
            "model": model_name, "n_params": 0, "n_scored": n_scored, "params": {}, "theta": [],
            "loss": random_nll, "random_nll": random_nll, "converged": True, "failed": False,
            "near_random": True, "boundary": {}, "any_boundary": False,
            "n_starts": 0, "n_converged_starts": 0, "prop_starts_within_0.1": 1.0, "starts": [],
        }

    f = _objective(model_name, choices, rewards, mask, variant)
    x0s = rng.normal(0.0, start_sd, size=(n_starts, k))
    starts = []
    for i, x0 in enumerate(x0s):
        rec = {"start": i, "x0": x0.tolist()}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = minimize(f, x0, jac=True, method="L-BFGS-B", options=OPTIMISER_OPTIONS)
            loss, grad = f(res.x)
            params = np.asarray(transform(model_name, jnp.asarray(res.x), variant))
            finite = bool(np.isfinite(loss) and np.all(np.isfinite(res.x)) and np.all(np.isfinite(params)))
            gnorm = float(np.linalg.norm(grad))
            rec.update({
                "theta": res.x.tolist(),
                "params": dict(zip(names, params.tolist())),
                "loss": float(loss),
                "grad_norm": gnorm,
                "status": int(res.status),
                "message": str(res.message),
                "nit": int(res.nit),
                "nfev": int(res.nfev),
                "warnings": [str(w.message) for w in caught],
                "finite": finite,
                "converged": bool(finite and gnorm < grad_norm_tol),
            })
        except Exception as exc:  # recorded, never raised: a start may fail on its own
            rec.update({"theta": None, "params": None, "loss": float("nan"), "grad_norm": float("nan"),
                        "status": -1, "message": repr(exc), "nit": 0, "nfev": 0, "warnings": [],
                        "finite": False, "converged": False})
        starts.append(rec)

    converged = [s for s in starts if s["converged"]]
    finite = [s for s in starts if s["finite"]]
    pool = converged or finite
    if not pool:
        return {
            "model": model_name, "n_params": k, "n_scored": n_scored,
            "params": {n: float("nan") for n in names}, "theta": [float("nan")] * k,
            "loss": float("nan"), "random_nll": random_nll, "converged": False, "failed": True,
            "near_random": False, "boundary": {n: False for n in names}, "any_boundary": False,
            "n_starts": n_starts, "n_converged_starts": 0, "prop_starts_within_0.1": 0.0,
            "starts": starts,
        }
    best = min(pool, key=lambda s: s["loss"])
    flags = boundary_flags(best["params"], variant)
    within = sum(1 for s in finite if s["loss"] <= best["loss"] + 0.1)
    return {
        "model": model_name,
        "variant": variant,
        "n_params": k,
        "n_scored": n_scored,
        "params": best["params"],
        "theta": best["theta"],
        "loss": best["loss"],
        "random_nll": random_nll,
        "converged": bool(best["converged"]),
        "failed": False,
        "near_random": bool(random_nll - best["loss"] < k),  # AIC: no gain over random
        "boundary": flags,
        "any_boundary": any(flags.values()),
        "n_starts": n_starts,
        "n_converged_starts": len(converged),
        "prop_starts_within_0.1": within / n_starts,
        "starts": starts,
    }


def fit_model(
    trials: pd.DataFrame,
    model_name: str,
    n_starts: int = 20,
    seed=1,
    score_mask=None,
    **kwargs,
) -> dict:
    """Fit one model with multi-start optimisation and return diagnostics."""
    trials = trials.sort_values("trial")
    return fit_arrays(
        model_name,
        trials["choice"].to_numpy(),
        trials["reward"].to_numpy(),
        score_mask=score_mask,
        n_starts=n_starts,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def starts_table(fit: dict) -> pd.DataFrame:
    """All starts of one fit as a table (params expanded into columns)."""
    rows = []
    for s in fit["starts"]:
        row = {k: v for k, v in s.items() if k not in ("params", "theta", "x0", "warnings")}
        row.update({f"est_{k}": v for k, v in (s["params"] or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=1), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written fit that would be skipped
