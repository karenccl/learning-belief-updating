"""Model recovery: simulate from each primary model, fit every model with masked
leave-block-out CV (4 x 40-trial folds, 10 starts per fold), and save held-out scores.

Benchmarks (random, perseveration) are fitted too; they are reported but are not
candidates in the confusion matrix.

Examples
    python scripts/03_model_recovery.py                 # 160 trials x 3 conditions, config agents
    python scripts/03_model_recovery.py --agents 100    # final figures
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.models import ALL_MODELS, PRIMARY_MODELS  # noqa: E402
from src.recovery import run_model_recovery_agent, run_parallel  # noqa: E402
from src.task import CONDITIONS  # noqa: E402


def main() -> None:
    cfg = load_config()
    mr = cfg["recovery"]["model"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS), help="generating models")
    ap.add_argument("--fitted", nargs="+", default=list(ALL_MODELS), help="fitted models")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--agents", type=int, default=mr["agents_per_cell"])
    ap.add_argument("--n-jobs", type=int, default=cfg["compute"]["n_jobs"])
    args = ap.parse_args()

    jobs = [(cfg, m, c, a, tuple(args.fitted)) for m in args.models for c in args.conditions
            for a in range(args.agents)]
    print(f"{len(jobs)} agents x {len(args.fitted)} fitted models", flush=True)
    t0 = time.perf_counter()
    status = run_parallel(run_model_recovery_agent, jobs, n_jobs=args.n_jobs, verbose=2)
    print(f"done={status.count('done')} skipped={status.count('skipped')} "
          f"wall={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
