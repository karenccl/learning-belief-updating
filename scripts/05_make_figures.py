"""Make all figures from results/tables/*.csv (never from fits or in-memory objects)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plots  # noqa: E402
from src.config import FIGURES, TABLES  # noqa: E402


def main() -> None:
    T, F = TABLES, FIGURES
    made = []

    def run(name, fn, *inputs, **kw):
        if all((T / i).exists() for i in inputs):
            fn(*[T / i for i in inputs], F / name, **kw)
            made.append(name)

    run("fig1_task_schedule.png", plots.task_schedule, "task_schedule.csv")
    run("latent_trajectories.png", plots.latent_trajectories, "latent_examples.csv")
    run("smoke_true_vs_recovered.png", plots.smoke_true_vs_recovered, "smoke_fixed_rl_recovery.csv")
    for n in (80, 160, 320):
        run(f"fig2_true_vs_recovered_{n}.png", plots.true_vs_recovered, "parameter_recovery_agents.csv",
            n_trials=n)
    run("fig3_recovery_vs_trial_count.png", plots.recovery_vs_trial_count, "parameter_recovery_metrics.csv")
    run("bias_mae_by_condition.png", plots.bias_mae_by_condition, "parameter_recovery_metrics.csv")
    run("parameter_tradeoffs.png", plots.parameter_tradeoffs, "parameter_recovery_agents.csv")
    run("near_random_by_beta.png", plots.near_random_vs_beta, "near_random_by_beta.csv")
    run("fig4_confusion_matrix.png", plots.confusion_matrices, "model_recovery_confusion.csv")
    run("margin_distribution.png", plots.margin_distribution, "model_recovery_winners.csv")
    run("fig5_heldout_comparison.png", plots.heldout_comparison, "model_recovery_scores.csv")
    run("fig5b_paired_differences.png", plots.paired_differences, "model_recovery_scores.csv")
    run("demo_paired_differences.png", plots.paired_differences, "demo_cv_scores.csv")
    # post-hoc sensitivity analysis (bounded beta and h)
    run("sensitivity_confusion_matrix_bounded.png", plots.confusion_matrices, "model_recovery_confusion_bounded.csv")
    run("sensitivity_bounded.png", plots.sensitivity_bounded, "sensitivity_model_recovery.csv",
        "sensitivity_catastrophic_folds.csv", "sensitivity_parameter_recovery.csv")
    if hasattr(plots, "ppc_reversal_curves"):
        run("fig6_ppc_reversal_curves.png", plots.ppc_reversal_curves, "ppc_reversal_curves.csv")
        run("ppc_summaries.png", plots.ppc_summaries, "ppc_summaries.csv")
    print("figures:", *made, sep="\n  ")


if __name__ == "__main__":
    main()
