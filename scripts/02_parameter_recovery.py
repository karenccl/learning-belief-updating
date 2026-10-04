"""Parameter recovery: simulate agents from each primary model and refit the same model.

Each agent's fit is written to
results/fits/parameter_recovery/{model}/{condition}_{ntrials}_{agent:03d}_{model}.json
and skipped if it exists, so runs resume after interruption.

Examples
    python scripts/02_parameter_recovery.py                       # full grid, config agents
    python scripts/02_parameter_recovery.py --agents 100          # final figures
    python scripts/02_parameter_recovery.py --models fixed_rl asymmetric_rl --trial-counts 160
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.models import PRIMARY_MODELS  # noqa: E402
from src.recovery import run_parallel, run_parameter_recovery_agent  # noqa: E402
from src.task import CONDITIONS  # noqa: E402


def main() -> None:
    cfg = load_config()
    pr = cfg["recovery"]["parameter"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--trial-counts", nargs="+", type=int, default=pr["trial_counts"])
    ap.add_argument("--agents", type=int, default=pr["agents_per_cell"])
    ap.add_argument("--n-jobs", type=int, default=cfg["compute"]["n_jobs"])
    args = ap.parse_args()

    jobs = [(cfg, m, c, n, a) for m in args.models for n in args.trial_counts
            for c in args.conditions for a in range(args.agents)]
    print(f"{len(jobs)} agent fits", flush=True)
    t0 = time.perf_counter()
    status = run_parallel(run_parameter_recovery_agent, jobs, n_jobs=args.n_jobs, verbose=2)
    print(f"done={status.count('done')} skipped={status.count('skipped')} "
          f"wall={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
