"""Deterministic per-agent seeds.

``SeedSequence(master, spawn_key=(i,))`` is identical to the i-th child of
``SeedSequence(master).spawn(n)`` for any n > i, so children are indexed
directly without materialising the whole spawn. The index is a fixed mixed-radix
encoding of (stage, generating model, condition, trial count, agent), which
makes each agent's streams independent of execution order and of which other
cells are run.
"""
from __future__ import annotations

import numpy as np

from .models import ALL_MODELS
from .task import CONDITIONS

STAGES = ("smoke", "parameter_recovery", "model_recovery", "demo")
TRIAL_COUNTS = (50, 80, 160, 320)
MAX_AGENTS = 1000
STREAMS = ("params", "simulation", "fit")


def agent_index(stage: str, model: str, condition: str, n_trials: int, agent: int) -> int:
    if not 0 <= agent < MAX_AGENTS:
        raise ValueError("agent index out of range")
    idx = STAGES.index(stage)
    idx = idx * len(ALL_MODELS) + ALL_MODELS.index(model)
    idx = idx * len(CONDITIONS) + CONDITIONS.index(condition)
    idx = idx * len(TRIAL_COUNTS) + TRIAL_COUNTS.index(n_trials)
    return idx * MAX_AGENTS + agent


def agent_seed(master_seed: int, stage, model, condition, n_trials, agent) -> np.random.SeedSequence:
    """The agent's spawned child: SeedSequence(master_seed).spawn(N)[agent_index]."""
    return np.random.SeedSequence(
        master_seed, spawn_key=(agent_index(stage, model, condition, n_trials, agent),)
    )


def agent_stream(master_seed: int, stage, model, condition, n_trials, agent, stream: str,
                 *extra: int) -> np.random.SeedSequence:
    """A named sub-stream of the agent's child (grandchild keys), e.g. for fitting fold k."""
    key = (agent_index(stage, model, condition, n_trials, agent), STREAMS.index(stream), *extra)
    return np.random.SeedSequence(master_seed, spawn_key=key)


def agent_rng(*args, **kwargs) -> np.random.Generator:
    return np.random.default_rng(agent_stream(*args, **kwargs))
