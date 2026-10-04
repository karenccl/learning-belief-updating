"""Simulate one agent's choices, rewards, and latent states (plain NumPy)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .models import make_agent, sigmoid_np
from .task import cv_block_index, sample_reward


def simulate_model(
    model_name: str,
    parameters: dict,
    environment: dict,
    seed,
    participant_id: str | int = 0,
    cv_block_length: int = 40,
) -> pd.DataFrame:
    """Simulate one participant's choices, rewards, and latent states.

    ``seed`` may be an int or a ``np.random.SeedSequence`` (the agent's spawned child).
    Latent columns record the state *used for the choice* on that trial, plus the
    prediction error and learning rate produced by that trial's update.
    """
    rng = np.random.default_rng(seed)
    agent = make_agent(model_name, parameters)
    n = environment["n_trials"]
    rows = []
    prev_s = 0.0
    for t in range(n):
        logit = agent.logit(t, prev_s)
        p1 = sigmoid_np(logit)
        latents = dict(agent.latents())
        a = int(rng.random() < p1)
        r = sample_reward(rng, environment, t, a)
        latents.update(agent.update(a, r))
        rows.append({"choice": a, "reward": r, "p_choice_1": p1, **latents})
        prev_s = 2.0 * a - 1.0

    df = pd.DataFrame(rows)
    df.insert(0, "participant_id", participant_id)
    df.insert(1, "trial", environment["trial"])
    df.insert(2, "condition", environment.get("condition", "custom"))
    df.insert(3, "block", np.cumsum(environment["reversal"]))  # stable-contingency period index
    df["p_option_0"] = environment["p_option_0"]
    df["p_option_1"] = environment["p_option_1"]
    df["advantageous_option"] = environment["advantageous_option"]
    df["reversal"] = environment["reversal"]
    df["true_model"] = model_name
    df["true_parameters"] = json.dumps(parameters)
    df["cv_block"] = cv_block_index(n, cv_block_length)
    return df


def simulate_arrays(model_name: str, parameters: dict, environment: dict, seed) -> tuple[np.ndarray, np.ndarray]:
    """Fast variant of ``simulate_model`` returning only (choices, rewards).

    Uses the same agent classes and the same order of random draws, so for a given
    seed it reproduces ``simulate_model`` exactly (unit-tested).
    """
    rng = np.random.default_rng(seed)
    agent = make_agent(model_name, parameters)
    n = environment["n_trials"]
    choices = np.empty(n, dtype=int)
    rewards = np.empty(n, dtype=int)
    prev_s = 0.0
    for t in range(n):
        a = int(rng.random() < sigmoid_np(agent.logit(t, prev_s)))
        r = sample_reward(rng, environment, t, a)
        agent.update(a, r)
        choices[t], rewards[t] = a, r
        prev_s = 2.0 * a - 1.0
    return choices, rewards
