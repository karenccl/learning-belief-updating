"""Day 5: held-out comparison on the demonstration datasets and parametric predictive checks.

For each primary generating model, a 100-agent demonstration dataset (moderate, 160 trials)
is treated as observed data (true parameters are not used):
  1. masked leave-block-out CV of all six models (4 folds x 10 starts);
  2. the selected model = highest mean held-out LL per trial on that dataset;
  3. the selected model and the true generating model are fitted to each agent's full data
     (20 starts) and 100 replicate datasets are simulated per agent;
  4. observed vs simulated behavioural summaries are tabulated;
  5. sensitivity: forward-chaining CV of all models on the same datasets.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.config import DATA_SIM, FITS, TABLES, load_config  # noqa: E402
from src.models import ALL_MODELS, PRIMARY_MODELS  # noqa: E402
from src.predictive_checks import ppc_tables, run_forward_chaining_agent, run_ppc_agent  # noqa: E402
from src.recovery import collect_model_recovery, run_model_recovery_agent, run_parallel, simulate_agent  # noqa: E402


def summarise_scores(long: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for gen, g in long.groupby("generating_model"):
        wide = g.pivot(index="agent", columns="fitted_model", values="heldout_ll_per_trial")
        best = wide.mean().idxmax()
        for m in wide.columns:
            x = wide[m]
            d = x - wide[best]
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"scheme": label, "generating_model": gen, "fitted_model": m, "n_agents": len(x),
                         "mean": x.mean(), "median": x.median(), "se": x.std(ddof=1) / np.sqrt(len(x)),
                         "diff_vs_selected_mean": d.mean(),
                         "diff_ci_low": d.mean() - stats.t.ppf(0.975, len(d) - 1) * se if m != best else 0.0,
                         "diff_ci_high": d.mean() + stats.t.ppf(0.975, len(d) - 1) * se if m != best else 0.0,
                         "diff_median": d.median(), "prop_agents_better_than_selected": (d > 0).mean(),
                         "selected": m == best, "rank": int(wide.mean().rank(ascending=False)[m])})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=cfg["predictive_checks"]["demo_agents"])
    ap.add_argument("--n-jobs", type=int, default=cfg["compute"]["n_jobs"])
    ap.add_argument("--skip-forward-chaining", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    DATA_SIM.mkdir(parents=True, exist_ok=True)
    for gen in PRIMARY_MODELS:
        pd.concat([simulate_agent(cfg, "demo", gen, "moderate", 160, a)[1] for a in range(args.agents)]).to_csv(
            DATA_SIM / f"demo_{gen}.csv", index=False)

    # 1. CV of all models on each demo dataset
    jobs = [(cfg, gen, "moderate", a, tuple(ALL_MODELS), "demo", 160) for gen in PRIMARY_MODELS
            for a in range(args.agents)]
    run_parallel(run_model_recovery_agent, jobs, n_jobs=args.n_jobs, verbose=0)
    long = collect_model_recovery("demo")
    long = long[long.agent < args.agents]
    long.to_csv(TABLES / "demo_cv_scores.csv", index=False)
    summary = summarise_scores(long, "leave_block_out_masked")
    selected = summary[summary.selected].set_index("generating_model")["fitted_model"].to_dict()
    print("selected per demo dataset:", selected, flush=True)

    # 2-4. Predictive checks for the selected model and the true model
    jobs = [(cfg, gen, a, tuple(dict.fromkeys([selected[gen], gen]))) for gen in PRIMARY_MODELS
            for a in range(args.agents)]
    run_parallel(run_ppc_agent, jobs, n_jobs=args.n_jobs, verbose=0)
    curves, blocks, summ, lrs = ppc_tables(PRIMARY_MODELS)
    for df_, name in [(curves, "ppc_reversal_curves"), (blocks, "ppc_block_curves"), (summ, "ppc_summaries"),
                      (lrs, "ppc_fitted_learning_rates")]:
        df_.assign(selected_model=lambda d: d.generating_model.map(selected)).to_csv(
            TABLES / f"{name}.csv", index=False)

    # 5. Forward-chaining sensitivity check
    if not args.skip_forward_chaining:
        jobs = [(cfg, gen, a, tuple(ALL_MODELS)) for gen in PRIMARY_MODELS for a in range(args.agents)]
        run_parallel(run_forward_chaining_agent, jobs, n_jobs=args.n_jobs, verbose=0)
        fc = pd.DataFrame([json.loads(p.read_text(encoding="utf-8"))
                           for p in (FITS / "demo_forward_chaining").rglob("*.json")])
        fc = fc[fc.agent < args.agents][["generating_model", "fitted_model", "agent", "heldout_ll_per_trial",
                                         "n_failed_folds"]]
        fc.to_csv(TABLES / "demo_forward_chaining_scores.csv", index=False)
        summary = pd.concat([summary, summarise_scores(fc, "forward_chaining")])
    summary.to_csv(TABLES / "demo_cv_summary.csv", index=False)
    print(summary.round(4).to_string())
    print(f"wall {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
