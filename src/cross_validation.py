"""Masked leave-block-out cross-validation (primary) and forward chaining (sensitivity).

Latent states always run through the *full* trial sequence. During training only
training trials are scored; at test time the fitted parameters are fixed and only
held-out trials are scored. Within a held-out block the model still updates after
every observed outcome, and never uses future choices or outcomes.

Caveat (stated in the methods): held-out choices and outcomes act as *inputs* to
state updates that affect later training trials, although they are never scored
during training. Forward chaining removes this information flow.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .fit import fit_arrays
from .models import MODEL_PARAMS, _as_jax, dict_to_array, logps_fn
from .task import cv_block_index


def block_masks(n_trials: int, block_length: int = 40) -> list[np.ndarray]:
    """One held-out mask per fixed block, defined by trial number only."""
    blocks = cv_block_index(n_trials, block_length)
    return [(blocks == b).astype(float) for b in np.unique(blocks)]


def sequential_predictive_log_likelihood(
    parameters: dict,
    choices,
    rewards,
    heldout_mask,
    model_name: str,
) -> float:
    """Run the model through the full trial sequence with fixed parameters
    and return the summed log probability of choices where heldout_mask == 1."""
    c, r, m = _as_jax(choices, rewards, heldout_mask)
    lp = logps_fn(model_name)(dict_to_array(model_name, parameters), c, r)
    return float(np.sum(np.asarray(m) * np.asarray(lp)))


def _compact_fold(fold: int, fit: dict, held: np.ndarray, score: float) -> dict:
    return {
        "fold": fold,
        "n_heldout": int(held.sum()),
        "heldout_ll": score,
        "train_loss": fit["loss"],
        "params": fit["params"],
        "converged": fit["converged"],
        "failed": fit["failed"],
        "near_random": fit["near_random"],
        "boundary": fit["boundary"],
        "any_boundary": fit["any_boundary"],
        "n_converged_starts": fit["n_converged_starts"],
        "prop_starts_within_0.1": fit["prop_starts_within_0.1"],
        "starts": [
            {k: s.get(k) for k in ("start", "loss", "grad_norm", "converged", "status", "nit", "params")}
            for s in fit["starts"]
        ],
    }


def _run_folds(model_name, choices, rewards, splits, n_starts, seed_fn, **fit_kwargs) -> dict:
    choices = np.asarray(choices, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    folds, total, n_scored = [], 0.0, 0
    for k, (train, held) in enumerate(splits):
        rng = seed_fn(k) if seed_fn is not None else np.random.default_rng()
        fit = fit_arrays(model_name, choices, rewards, score_mask=train, n_starts=n_starts, rng=rng,
                         **fit_kwargs)
        score = (float("nan") if fit["failed"]
                 else sequential_predictive_log_likelihood(fit["params"], choices, rewards, held, model_name))
        folds.append(_compact_fold(k, fit, held, score))
        total += score
        n_scored += int(held.sum())
    return {
        "model": model_name,
        "n_params": len(MODEL_PARAMS[model_name]),
        "n_trials": int(len(choices)),
        "n_scored": n_scored,
        "heldout_ll": total,
        "heldout_ll_per_trial": total / n_scored,
        "n_failed_folds": sum(f["failed"] for f in folds),
        "folds": folds,
    }


def cross_validate(
    model_name: str,
    choices,
    rewards,
    n_starts: int = 10,
    block_length: int = 40,
    seed_fn: Callable[[int], np.random.Generator] | None = None,
    **fit_kwargs,
) -> dict:
    """Masked leave-block-out CV. ``seed_fn(fold)`` returns the optimiser RNG for a fold."""
    masks = block_masks(len(choices), block_length)
    splits = [(1.0 - held, held) for held in masks]
    out = _run_folds(model_name, choices, rewards, splits, n_starts, seed_fn, **fit_kwargs)
    out["scheme"] = "leave_block_out_masked"
    return out


def forward_chaining(
    model_name: str,
    choices,
    rewards,
    n_starts: int = 10,
    block_length: int = 40,
    seed_fn: Callable[[int], np.random.Generator] | None = None,
    **fit_kwargs,
) -> dict:
    """Fit blocks < k, predict block k (k >= 1). No information flows from held-out data."""
    masks = block_masks(len(choices), block_length)
    splits = [(np.sum(masks[:k], axis=0), masks[k]) for k in range(1, len(masks))]
    out = _run_folds(model_name, choices, rewards, splits, n_starts, seed_fn, **fit_kwargs)
    out["scheme"] = "forward_chaining"
    return out
