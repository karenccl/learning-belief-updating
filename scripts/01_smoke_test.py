"""Minimal end-to-end smoke test.

1. Simulate 10 agents x 50 trials of fixed RL (alpha = 0.30, beta = 4.0), moderate schedule.
2. Fit fixed RL with the single-fit multi-start setting (20 starts; locked decision).
3. Confirm every selected fit is finite.
4. Plot true versus recovered alpha and beta (from the saved table).
5. Model-recovery row: fit fixed RL and asymmetric RL to each dataset; score with
   masked leave-block-out held-out log likelihood (the locked comparison score).
6. Save everything under results/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DATA_SIM, FIGURES, FITS, TABLES, load_config  # noqa: E402
from src.cross_validation import cross_validate  # noqa: E402
from src.fit import fit_arrays, save_json  # noqa: E402
from src.models import ALL_MODELS  # noqa: E402
from src.plots import smoke_true_vs_recovered  # noqa: E402
from src.seeds import agent_rng, agent_stream  # noqa: E402
from src.simulate import simulate_model  # noqa: E402
from src.task import environment_for_condition  # noqa: E402

N_AGENTS, N_TRIALS, CONDITION = 10, 50, "moderate"
TRUTH = {"alpha": 0.30, "beta": 4.0}


def main() -> None:
    cfg = load_config()
    seed = cfg["master_seed"]
    n_starts = cfg["fitting"]["n_starts_single"]
    block = cfg["task"]["cv_block_length"]
    env = environment_for_condition(CONDITION, N_TRIALS, cfg["task"])
    key = ("smoke", "fixed_rl", CONDITION, N_TRIALS)

    t0 = time.perf_counter()
    sims, rec_rows, mr_rows = [], [], []
    for i in range(N_AGENTS):
        df = simulate_model("fixed_rl", TRUTH, env, seed=agent_stream(seed, *key, i, "simulation"),
                            participant_id=i, cv_block_length=block)
        sims.append(df)
        c, r = df.choice.to_numpy(), df.reward.to_numpy(float)

        # Steps 2-3: full-data fit of the generating model
        path = FITS / "smoke" / "fixed_rl" / f"{CONDITION}_{N_TRIALS}_{i:03d}_fixed_rl.json"
        if path.exists():
            fit = json.loads(path.read_text(encoding="utf-8"))
        else:
            fit = fit_arrays("fixed_rl", c, r, n_starts=n_starts,
                             rng=agent_rng(seed, *key, i, "fit", ALL_MODELS.index("fixed_rl")))
            fit.update({"true_model": "fixed_rl", "true_params": TRUTH, "agent": i})
            save_json(fit, path)
        assert not fit["failed"] and np.isfinite(fit["loss"]), f"agent {i}: non-finite fit"
        rec_rows.append({"agent": i, "n_trials": N_TRIALS, "loss": fit["loss"],
                         "converged": fit["converged"], "near_random": fit["near_random"],
                         **{f"true_{k}": v for k, v in TRUTH.items()},
                         **{f"est_{k}": v for k, v in fit["params"].items()}})

        # Step 5: one model-recovery row scored by held-out log likelihood
        row = {"agent": i, "true_model": "fixed_rl"}
        for fitted in ("fixed_rl", "asymmetric_rl"):
            cv_path = FITS / "smoke_cv" / "fixed_rl" / f"{CONDITION}_{N_TRIALS}_{i:03d}_{fitted}.json"
            if cv_path.exists():
                cv = json.loads(cv_path.read_text(encoding="utf-8"))
            else:
                cv = cross_validate(fitted, c, r, n_starts=n_starts, block_length=block,
                                    seed_fn=lambda fold, f=fitted: agent_rng(
                                        seed, *key, i, "fit", 100 + ALL_MODELS.index(f), fold))
                save_json(cv, cv_path)
            row[f"heldout_ll_per_trial_{fitted}"] = cv["heldout_ll_per_trial"]
        mr_rows.append(row)

    DATA_SIM.mkdir(parents=True, exist_ok=True)
    pd.concat(sims).to_csv(DATA_SIM / "smoke_fixed_rl.csv", index=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(TABLES / "smoke_fixed_rl_recovery.csv", index=False)
    mr = pd.DataFrame(mr_rows)
    mr["winner"] = np.where(mr.heldout_ll_per_trial_fixed_rl >= mr.heldout_ll_per_trial_asymmetric_rl,
                            "fixed_rl", "asymmetric_rl")
    mr.to_csv(TABLES / "smoke_model_recovery_row.csv", index=False)

    smoke_true_vs_recovered(TABLES / "smoke_fixed_rl_recovery.csv",
                            FIGURES / "smoke_true_vs_recovered.png")
    print(rec[["agent", "est_alpha", "est_beta", "loss", "converged", "near_random"]].round(3).to_string())
    print(f"\nmedian recovered alpha = {rec.est_alpha.median():.3f} (true 0.30), "
          f"beta = {rec.est_beta.median():.2f} (true 4.0)")
    print(f"all finite: {bool(np.isfinite(rec[['loss', 'est_alpha', 'est_beta']]).all().all())}")
    print("\nmodel-recovery row (held-out LL per trial):")
    print(mr.round(4).to_string())
    print(f"\nwinner counts: {mr.winner.value_counts().to_dict()}")
    print(f"elapsed {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
