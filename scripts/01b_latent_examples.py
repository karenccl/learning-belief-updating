"""Simulate one agent per model (volatile, 160 trials) and save trialwise latents + the task schedule."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import FIGURES, TABLES, load_config  # noqa: E402
from src.plots import latent_trajectories, task_schedule  # noqa: E402
from src.recovery import simulate_agent  # noqa: E402
from src.task import CONDITIONS, environment_for_condition  # noqa: E402

EXAMPLES = {
    "fixed_rl": {"alpha": 0.3, "beta": 5.0},
    "asymmetric_rl": {"alpha_pos": 0.5, "alpha_neg": 0.1, "beta": 5.0},
    "hybrid_ph": {"eta": 0.4, "kappa": 0.8, "beta": 5.0},
    "bayes_cp": {"h": 0.1, "beta": 8.0},
}


def main() -> None:
    cfg = load_config()
    TABLES.mkdir(parents=True, exist_ok=True)
    sched = []
    for cond in CONDITIONS:
        env = environment_for_condition(cond, 160, cfg["task"])
        sched.append(pd.DataFrame({k: env[k] for k in ("trial", "p_option_0", "p_option_1",
                                                          "advantageous_option", "reversal")}).assign(condition=cond))
    pd.concat(sched).to_csv(TABLES / "task_schedule.csv", index=False)
    sims = [simulate_agent(cfg, "smoke", m, "volatile", 160, 0, params=p)[1] for m, p in EXAMPLES.items()]
    pd.concat(sims).to_csv(TABLES / "latent_examples.csv", index=False)
    task_schedule(TABLES / "task_schedule.csv", FIGURES / "fig1_task_schedule.png")
    latent_trajectories(TABLES / "latent_examples.csv", FIGURES / "latent_trajectories.png")
    print("saved task schedule and latent trajectories")


if __name__ == "__main__":
    main()
