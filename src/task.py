"""Two-armed reversal-learning environment.

Trials are 1-indexed in tables and in schedule definitions: "reversal after
trial k" means the contingencies differ from trial k+1 onwards. Internally,
arrays are 0-indexed, so the first trial with swapped contingencies sits at
array index k.
"""
from __future__ import annotations

import numpy as np

CONDITIONS = ("stable", "moderate", "volatile")


def reversal_schedule(condition: str, n_trials: int, task_cfg: dict) -> list[int]:
    """Return the trials *after which* contingencies reverse (1-indexed).

    Stable: exactly one reversal after ``reversal_after_fraction * n_trials``.
    Periodic: a reversal after every ``interval`` trials, excluding the last trial.
    """
    spec = task_cfg["conditions"][condition]
    if spec["type"] == "single":
        k = int(round(spec["reversal_after_fraction"] * n_trials))
        return [k]
    if spec["type"] == "periodic":
        interval = int(spec["interval"])
        return list(range(interval, n_trials, interval))
    raise ValueError(f"Unknown condition type: {spec['type']}")


def make_environment(
    n_trials: int,
    reward_high: float,
    reward_low: float,
    reversal_trials: list[int],
    initial_advantageous_option: int,
) -> dict:
    """Return trialwise reward probabilities and reversal indicators."""
    if initial_advantageous_option not in (0, 1):
        raise ValueError("initial_advantageous_option must be 0 or 1")
    reversal = np.zeros(n_trials, dtype=int)
    for k in reversal_trials:
        if not 1 <= k < n_trials:
            raise ValueError(f"reversal after trial {k} is outside 1..{n_trials - 1}")
        reversal[k] = 1  # array index k == trial k+1
    # The advantageous option flips once per reversal seen so far.
    advantageous = (initial_advantageous_option + np.cumsum(reversal)) % 2
    p_option_0 = np.where(advantageous == 0, reward_high, reward_low).astype(float)
    p_option_1 = np.where(advantageous == 1, reward_high, reward_low).astype(float)
    return {
        "n_trials": n_trials,
        "trial": np.arange(1, n_trials + 1),
        "advantageous_option": advantageous.astype(int),
        "p_option_0": p_option_0,
        "p_option_1": p_option_1,
        "reversal": reversal,
        "reversal_trials": list(reversal_trials),
        "reward_high": reward_high,
        "reward_low": reward_low,
    }


def environment_for_condition(condition: str, n_trials: int, task_cfg: dict) -> dict:
    """Build the environment for a named volatility condition."""
    env = make_environment(
        n_trials=n_trials,
        reward_high=task_cfg["reward_high"],
        reward_low=task_cfg["reward_low"],
        reversal_trials=reversal_schedule(condition, n_trials, task_cfg),
        initial_advantageous_option=task_cfg["initial_advantageous_option"],
    )
    env["condition"] = condition
    return env


def sample_reward(rng: np.random.Generator, environment: dict, t: int, action: int) -> int:
    """Draw a binary reward for ``action`` on 0-indexed trial ``t``."""
    p = environment["p_option_1"][t] if action == 1 else environment["p_option_0"][t]
    return int(rng.random() < p)


def cv_block_index(n_trials: int, block_length: int = 40) -> np.ndarray:
    """Fixed block index (0, 1, ...) per trial, independent of the reversal schedule."""
    return np.arange(n_trials) // block_length
