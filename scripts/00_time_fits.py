"""Time JIT-compiled multi-start fits per model and project the recovery workload.

Writes results/tables/timing.csv. Numbers feed docs/compute_budget.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import TABLES, load_config  # noqa: E402
from src.cross_validation import cross_validate  # noqa: E402
from src.fit import fit_arrays  # noqa: E402
from src.models import ALL_MODELS, PRIMARY_MODELS  # noqa: E402
from src.recovery import simulate_agent  # noqa: E402

REPS = 5


def main() -> None:
    cfg = load_config()
    rows = []
    for model in ALL_MODELS:
        gen = model if model in PRIMARY_MODELS else "fixed_rl"
        for n_trials in (80, 160, 320):
            # agent 999 of the demo stage is never used by any analysis
            _, df = simulate_agent(cfg, "demo", gen, "moderate", n_trials, 999)
            c, r = df.choice.to_numpy(), df.reward.to_numpy(float)
            t0 = time.perf_counter()
            fit_arrays(model, c, r, n_starts=1, rng=np.random.default_rng(0))  # includes compilation
            compile_s = time.perf_counter() - t0
            for n_starts in (10, 20):
                ts, nfev = [], []
                for rep in range(REPS):
                    t0 = time.perf_counter()
                    fit = fit_arrays(model, c, r, n_starts=n_starts, rng=np.random.default_rng(rep))
                    ts.append(time.perf_counter() - t0)
                    nfev.append(np.mean([s["nfev"] for s in fit["starts"]]) if fit["starts"] else 0)
                rows.append({"model": model, "n_trials": n_trials, "task": f"fit_{n_starts}_starts",
                             "seconds": float(np.median(ts)), "first_call_incl_compile_s": compile_s,
                             "mean_nfev_per_start": float(np.mean(nfev))})
            if n_trials == 160:
                ts = []
                for rep in range(REPS):
                    t0 = time.perf_counter()
                    cross_validate(model, c, r, n_starts=10, seed_fn=lambda k: np.random.default_rng(k))
                    ts.append(time.perf_counter() - t0)
                rows.append({"model": model, "n_trials": 160, "task": "cv_4folds_10_starts",
                             "seconds": float(np.median(ts)), "first_call_incl_compile_s": np.nan,
                             "mean_nfev_per_start": np.nan})
        print(model, "done", flush=True)
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "timing.csv", index=False)
    print(out.round(4).to_string())

    # Projection (single core) at the configured cell sizes
    pr = cfg["recovery"]["parameter"]
    mr = cfg["recovery"]["model"]
    t = out.set_index(["model", "n_trials", "task"])["seconds"]
    param_s = sum(t[(m, n, "fit_10_starts")] * pr["agents_per_cell"] * 3
                  for m in PRIMARY_MODELS for n in pr["trial_counts"])
    model_s = sum(t[(f, 160, "cv_4folds_10_starts")] * mr["agents_per_cell"] * 3 * len(PRIMARY_MODELS)
                  for f in ALL_MODELS)
    print(f"\nprojected single-core: parameter recovery {param_s / 3600:.2f} h, "
          f"model recovery {model_s / 3600:.2f} h (all 6 fitted models)")


if __name__ == "__main__":
    main()
