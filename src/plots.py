"""Figures. Every function reads CSV tables from results/tables, never in-memory fits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .models import MODEL_LABELS, MODEL_PARAMS, PRIMARY_MODELS  # noqa: E402

sns.set_theme(context="paper", style="ticks", font_scale=1.1)
PARAM_LABELS = {
    "alpha": r"$\alpha$", "alpha_pos": r"$\alpha^+$", "alpha_neg": r"$\alpha^-$", "beta": r"$\beta$",
    "eta": r"$\eta$", "kappa": r"$\kappa$", "h": r"$h$", "rho": r"$\rho$", "bias": r"$b$",
}
CONDITION_ORDER = ["stable", "moderate", "volatile"]
CONDITION_COLORS = dict(zip(CONDITION_ORDER, ["#4C72B0", "#DD8452", "#C44E52"]))


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def smoke_true_vs_recovered(table: Path, out: Path) -> None:
    df = pd.read_csv(table)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, p, lim in zip(axes, ["alpha", "beta"], [(0, 1), (0, 12)]):
        ax.scatter(df[f"true_{p}"], df[f"est_{p}"], alpha=0.7, s=30, zorder=3)
        ax.plot(lim, lim, "k--", lw=1)
        ax.axhline(df[f"true_{p}"].iloc[0], color="grey", lw=0.8, ls=":")
        ax.set(xlim=lim, ylim=lim, xlabel=f"true {PARAM_LABELS[p]}", ylabel=f"recovered {PARAM_LABELS[p]}")
    fig.suptitle(f"Smoke test: fixed RL, {len(df)} agents x {int(df.n_trials.iloc[0])} trials")
    fig.tight_layout()
    _save(fig, out)


def task_schedule(table: Path, out: Path) -> None:
    """Figure 1: reward probability of option 0 by trial in each condition (160 trials)."""
    df = pd.read_csv(table)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 4.6), sharex=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        g = df[df.condition == cond]
        ax.step(g.trial, g.p_option_0, where="post", color=CONDITION_COLORS[cond], lw=2, label="option 0")
        ax.step(g.trial, g.p_option_1, where="post", color="grey", lw=1.2, ls="--", label="option 1")
        for t in g.trial[g.reversal == 1]:
            ax.axvline(t - 0.5, color="k", lw=0.6, alpha=0.4)
        n_rev = int(g.reversal.sum())
        ax.set(ylim=(0, 1), yticks=[0.2, 0.8], ylabel="P(reward)")
        ax.set_title(f"{cond} ({n_rev} reversal{'s' if n_rev > 1 else ''})", loc="left", fontsize=10)
    axes[0].legend(loc="center left", ncol=2, frameon=False, fontsize=8)
    for b in range(0, 161, 40):
        axes[-1].axvline(b + 0.5, color="purple", lw=0.8, ls=":", ymax=0.15)
    axes[-1].set_xlabel("trial (dotted purple ticks: 40-trial CV block boundaries)")
    fig.suptitle("Two-armed reversal task: reward schedules", y=1.0)
    fig.tight_layout()
    _save(fig, out)


def latent_trajectories(table: Path, out: Path) -> None:
    """Latent states of one simulated agent per primary model (volatile condition)."""
    df = pd.read_csv(table)
    models = [m for m in PRIMARY_MODELS if m in df.true_model.unique()]
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.1 * len(models)), sharex=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        g = df[df.true_model == m]
        for t in g.trial[g.reversal == 1]:
            ax.axvline(t - 0.5, color="k", lw=0.6, alpha=0.3)
        if m == "bayes_cp":
            ax.plot(g.trial, g.belief_option_0, color="C3", label="P(option 0 best)")
        else:
            ax.plot(g.trial, g.Q0, color="C0", label="$Q_0$")
            ax.plot(g.trial, g.Q1, color="C1", label="$Q_1$")
        if m in ("asymmetric_rl", "hybrid_ph"):
            ax.plot(g.trial, g.alpha_t, color="C2", lw=1, label=r"$\alpha_t$")
        if m == "hybrid_ph":
            ax.plot(g.trial, g.associability, color="C4", lw=1, ls="--", label="$A_t$")
        ax.scatter(g.trial, np.where(g.choice == 0, 1.04, -0.04), c=np.where(g.reward == 1, "k", "w"),
                   edgecolors="k", s=9, lw=0.5, clip_on=False, zorder=3)
        ax.set(ylim=(-0.08, 1.08), ylabel="value / prob.")
        params = ", ".join(f"{k}={v:.2f}" for k, v in json.loads(g.true_parameters.iloc[0]).items())
        ax.set_title(f"{MODEL_LABELS[m]} ({params})", loc="left", fontsize=9)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("trial (dots: choice 0 top / 1 bottom; filled = rewarded)")
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------
BETA_CLIP = (0.1, 100.0)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b)[0])


def _param_label(model: str, p: str, log_beta: bool = False) -> str:
    lab = PARAM_LABELS[p]
    return f"{MODEL_LABELS[model]}\n{'log ' + lab if (log_beta and p == 'beta') else lab}"


def true_vs_recovered(agents_csv: Path, out: Path, n_trials: int = 160) -> None:
    """Figure 2: true vs recovered, every parameter of every primary model, coloured by condition."""
    df = pd.read_csv(agents_csv)
    df = df[df.n_trials == n_trials]
    ncol = max(len(MODEL_PARAMS[m]) for m in PRIMARY_MODELS)
    fig, axes = plt.subplots(len(PRIMARY_MODELS), ncol, figsize=(3.0 * ncol, 2.8 * len(PRIMARY_MODELS)))
    for i, m in enumerate(PRIMARY_MODELS):
        g = df[df.model == m]
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(MODEL_PARAMS[m]):
                ax.axis("off")
                continue
            p = MODEL_PARAMS[m][j]
            for cond in CONDITION_ORDER:
                h = g[g.condition == cond]
                t, e = h[f"true_{p}"], h[f"est_{p}"]
                if p == "beta":
                    clipped = e > BETA_CLIP[1]
                    e = e.clip(*BETA_CLIP)
                    ax.scatter(t[clipped], e[clipped], marker="^", s=22, color=CONDITION_COLORS[cond],
                               edgecolors="k", lw=0.4, zorder=4)
                    ax.scatter(t[~clipped], e[~clipped], s=10, alpha=0.6, color=CONDITION_COLORS[cond],
                               label=cond)
                else:
                    ax.scatter(t, e, s=10, alpha=0.6, color=CONDITION_COLORS[cond], label=cond)
            if p == "beta":
                ax.set(xscale="log", yscale="log", xlim=(0.4, 10), ylim=BETA_CLIP)
                ax.plot([0.4, 10], [0.4, 10], "k--", lw=0.8)
            else:
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.set(xlim=(0, 1), ylim=(-0.02, 1.02))
            rho = _spearman(g[f"true_{p}"], g[f"est_{p}"])
            ax.text(0.04, 0.96, f"Spearman {rho:.2f}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlabel(f"true {PARAM_LABELS[p]}")
            ax.set_ylabel(f"recovered {PARAM_LABELS[p]}")
            if j == 0:
                ax.set_title(MODEL_LABELS[m], loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Parameter recovery, {n_trials} trials (triangles: runaway estimates above "
                 f"{BETA_CLIP[1]:.0f}, drawn at the edge)", y=1.0)
    fig.tight_layout()
    _save(fig, out)


def recovery_vs_trial_count(metrics_csv: Path, out: Path) -> None:
    """Figure 3: Spearman recovery against trial count (dotted: agents with true beta >= 2)."""
    m = pd.read_csv(metrics_csv)
    ncol = max(len(MODEL_PARAMS[x]) for x in PRIMARY_MODELS)
    fig, axes = plt.subplots(len(PRIMARY_MODELS), ncol, figsize=(2.9 * ncol, 2.4 * len(PRIMARY_MODELS)),
                             sharey=True)
    for i, mod in enumerate(PRIMARY_MODELS):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(MODEL_PARAMS[mod]):
                ax.axis("off")
                continue
            p = MODEL_PARAMS[mod][j]
            g = m[(m.model == mod) & (m.parameter == p)]
            for cond in CONDITION_ORDER:
                h = g[g.condition == cond].sort_values("n_trials")
                ax.plot(h.n_trials, h.spearman_rho, "o-", color=CONDITION_COLORS[cond], label=cond)
                ax.plot(h.n_trials, h.spearman_rho_high_beta, "o:", color=CONDITION_COLORS[cond], alpha=0.5,
                        ms=3)
            ax.set(xscale="log", ylim=(-0.1, 1.0))
            ax.set_xticks([80, 160, 320], ["80", "160", "320"])
            ax.minorticks_off()
            ax.axhline(0.8, color="grey", lw=0.6, ls="--")
            ax.set_title(f"{MODEL_LABELS[mod]}: {PARAM_LABELS[p]}", fontsize=9, loc="left")
            ax.set_xlabel("trials")
            if j == 0:
                ax.set_ylabel("Spearman (true, est.)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Recovery vs trial count (solid: all agents; dotted: true beta >= 2)", y=1.0)
    fig.tight_layout()
    _save(fig, out)


def bias_mae_by_condition(metrics_csv: Path, out: Path, n_trials: int = 160) -> None:
    """Bias and MAE by volatility condition (beta in log units)."""
    m = pd.read_csv(metrics_csv)
    m = m[m.n_trials == n_trials].copy()
    is_beta = m.parameter == "beta"
    m["bias_plot"] = np.where(is_beta, m.bias_log, m.bias)
    m["mae_plot"] = np.where(is_beta, m.mae_log, m.mae)
    m["label"] = [_param_label(a, b, log_beta=True) for a, b in zip(m.model, m.parameter)]
    order = [_param_label(a, b, log_beta=True) for a in PRIMARY_MODELS for b in MODEL_PARAMS[a]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, col, lab in zip(axes, ["bias_plot", "mae_plot"], ["mean signed bias", "mean absolute error"]):
        sns.barplot(data=m, x="label", y=col, hue="condition", hue_order=CONDITION_ORDER, order=order,
                    palette=CONDITION_COLORS, ax=ax)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(ylabel=lab, xlabel="")
        ax.legend(frameon=False, fontsize=8, ncol=3)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Recovery error by volatility condition ({n_trials} trials; beta errors in log units)")
    fig.tight_layout()
    _save(fig, out)


def parameter_tradeoffs(agents_csv: Path, out: Path, n_trials: int = 160) -> None:
    """Spearman correlations among recovered parameters (true parameters are sampled independently)."""
    df = pd.read_csv(agents_csv)
    df = df[df.n_trials == n_trials]
    fig, axes = plt.subplots(1, len(PRIMARY_MODELS) + 1, figsize=(17, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.2]})
    for ax, m in zip(axes, PRIMARY_MODELS):
        g = df[df.model == m]
        est = pd.DataFrame({PARAM_LABELS[n]: (np.log(g[f"est_{n}"]) if n == "beta" else g[f"est_{n}"])
                            for n in MODEL_PARAMS[m]})
        sns.heatmap(est.corr(method="spearman"), vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f",
                    cbar=False, ax=ax, square=True)
        ax.set_title(MODEL_LABELS[m], fontsize=10)
    g = df[df.model == "fixed_rl"]
    ax = axes[-1]
    for cond in CONDITION_ORDER:
        h = g[g.condition == cond]
        ax.scatter(h.est_alpha, h.est_beta.clip(*BETA_CLIP), s=10, alpha=0.6, color=CONDITION_COLORS[cond],
                   label=cond)
    ax.set(yscale="log", ylim=BETA_CLIP, xlabel=r"recovered $\alpha$", ylabel=r"recovered $\beta$",
           title="Fixed RL estimates")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Trade-offs among recovered parameters ({n_trials} trials, conditions pooled; "
                 "Spearman, beta on log scale)", y=1.02)
    fig.tight_layout()
    _save(fig, out)


def near_random_vs_beta(table: Path, out: Path) -> None:
    d = pd.read_csv(table)
    fig, axes = plt.subplots(1, len(PRIMARY_MODELS), figsize=(13, 3), sharey=True)
    for ax, m in zip(axes, PRIMARY_MODELS):
        g = d[d.model == m]
        for n, h in g.groupby("n_trials"):
            ax.plot(range(len(h)), h.prop_near_random, "o-", label=f"{n} trials")
            ax.set_xticks(range(len(h)), h.true_beta_bin, rotation=30, fontsize=7)
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.set_xlabel(r"true $\beta$")
    axes[0].set_ylabel("P(near_random flag)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Model recovery and predictive comparison
# ---------------------------------------------------------------------------
def confusion_matrices(conf_csv: Path, out: Path) -> None:
    """Figure 4: row-normalised confusion matrices with a tie column."""
    c = pd.read_csv(conf_csv)
    panels = ["all"] + CONDITION_ORDER
    cols = list(PRIMARY_MODELS) + ["tie"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, cond in zip(axes, panels):
        g = c[c.condition == cond]
        prop = g.pivot(index="generating_model", columns="selected", values="proportion").reindex(
            index=list(PRIMARY_MODELS), columns=cols)
        cnt = g.pivot(index="generating_model", columns="selected", values="count").reindex(
            index=list(PRIMARY_MODELS), columns=cols)
        annot = prop.round(2).astype(str) + "\n(" + cnt.astype(int).astype(str) + ")"
        sns.heatmap(prop, vmin=0, vmax=1, cmap="Blues", annot=annot, fmt="", cbar=False, ax=ax,
                    annot_kws={"fontsize": 8})
        ax.set_xticklabels([MODEL_LABELS.get(x, "tie") for x in cols], rotation=30, ha="right")
        ax.set_yticklabels([MODEL_LABELS[x] for x in PRIMARY_MODELS], rotation=0)
        ax.set(xlabel="selected (highest held-out LL)", ylabel="generating model" if cond == "all" else "")
        acc = np.mean([prop.loc[m, m] for m in PRIMARY_MODELS])
        ax.set_title(f"{cond} (mean sensitivity {acc:.2f})", fontsize=10)
    fig.suptitle("Model recovery at 160 trials: P(selected | generating); tie = margin < 0.01 LL/trial", y=1.02)
    fig.tight_layout()
    _save(fig, out)


def margin_distribution(winners_csv: Path, out: Path, tie: float = 0.01) -> None:
    w = pd.read_csv(winners_csv)
    fig, axes = plt.subplots(1, len(PRIMARY_MODELS), figsize=(14, 3), sharey=True)
    bins = np.linspace(0, 0.15, 31)
    for ax, m in zip(axes, PRIMARY_MODELS):
        g = w[w.generating_model == m]
        for cond in CONDITION_ORDER:
            ax.hist(g[g.condition == cond].margin_per_trial.clip(upper=bins[-1]), bins=bins, histtype="step",
                    lw=1.5, color=CONDITION_COLORS[cond], label=cond)
        ax.axvline(tie, color="k", ls="--", lw=0.8)
        ax.set_title(f"generated by {MODEL_LABELS[m]}", fontsize=10)
        ax.set_xlabel("winner minus runner-up (LL/trial)")
    axes[0].set_ylabel("agents")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, out)


def heldout_comparison(scores_csv: Path, out: Path, ylim=(-0.85, -0.2), group_col: str = "generating_model",
                       title: str | None = None) -> None:
    """Figure 5: held-out LL per trial for every fitted model, by generating model and condition."""
    d = pd.read_csv(scores_csv)
    order = list(PRIMARY_MODELS) + ["perseveration", "random"]
    fig, axes = plt.subplots(1, len(PRIMARY_MODELS), figsize=(17, 3.8), sharey=True)
    for ax, m in zip(axes, PRIMARY_MODELS):
        g = d[d[group_col] == m].copy()
        n_clip = int((g.heldout_ll_per_trial < ylim[0]).sum())
        g["score"] = g.heldout_ll_per_trial.clip(lower=ylim[0])
        hue_order = [c for c in CONDITION_ORDER if c in set(g.condition)]
        sns.boxplot(data=g, x="fitted_model", y="score", hue="condition", order=order, hue_order=hue_order,
                    palette=CONDITION_COLORS, ax=ax, fliersize=1.5, linewidth=0.8)
        ax.axhline(np.log(0.5), color="k", ls=":", lw=0.8)
        ax.set_xticks(range(len(order)), [MODEL_LABELS[x] for x in order], rotation=40, ha="right", fontsize=8)
        ax.set(ylim=ylim, xlabel="fitted model", ylabel="held-out LL per trial" if m == PRIMARY_MODELS[0] else "")
        ax.set_title(f"generated by {MODEL_LABELS[m]}" + (f"\n({n_clip} scores below {ylim[0]} drawn at floor)"
                                                           if n_clip else ""), fontsize=9)
        if ax is not axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(title or "Held-out sequential log likelihood per trial (masked leave-block-out; "
                 "dotted line: random = log 0.5)", y=1.03)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Predictive checks and demo-dataset comparison
# ---------------------------------------------------------------------------
def ppc_reversal_curves(curves_csv: Path, out: Path) -> None:
    """Figure 6: observed reversal-locked accuracy vs 90% interval of fitted-model simulations."""
    d = pd.read_csv(curves_csv)
    fig, axes = plt.subplots(1, len(PRIMARY_MODELS), figsize=(16, 3.6), sharey=True)
    for ax, gen in zip(axes, PRIMARY_MODELS):
        g = d[d.generating_model == gen]
        sel = g.selected_model.iloc[0]
        fitted = list(dict.fromkeys([sel, gen]))
        for k, m in enumerate(fitted):
            h = g[g.fitted_model == m].sort_values("offset")
            col = f"C{k}"
            tag = "selected & true" if m == sel == gen else ("selected" if m == sel else "true model")
            ax.fill_between(h.offset, h.sim_q05, h.sim_q95, color=col, alpha=0.25, lw=0)
            ax.plot(h.offset, h.sim_mean, color=col, lw=1.2, label=f"{MODEL_LABELS[m]} ({tag})")
        h = g[g.fitted_model == fitted[0]].sort_values("offset")
        ax.plot(h.offset, h.observed, "ko", ms=3, label="observed")
        ax.axvline(-0.5, color="grey", ls="--", lw=0.8)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set(xlabel="trial relative to reversal", ylim=(0, 1),
               title=f"demo data from {MODEL_LABELS[gen]}")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_ylabel("P(choose currently advantageous)")
    fig.suptitle("Parametric predictive check: post-reversal adaptation (moderate, 160 trials; "
                 "bands = 90% interval of 100 simulated 100-agent datasets)", y=1.03)
    fig.tight_layout()
    _save(fig, out)


def ppc_summaries(summ_csv: Path, out: Path) -> None:
    d = pd.read_csv(summ_csv)
    stats_ = ["accuracy", "win_stay", "lose_switch", "p_repeat"]
    fig, axes = plt.subplots(1, len(stats_), figsize=(15, 3.4))
    for ax, s in zip(axes, stats_):
        g = d[d.statistic == s].reset_index(drop=True)
        y = np.arange(len(g))
        labels = [f"{MODEL_LABELS[a]} data / {MODEL_LABELS[b]} fit" for a, b in zip(g.generating_model, g.fitted_model)]
        ax.errorbar(g.sim_mean, y, xerr=[g.sim_mean - g.sim_q05, g.sim_q95 - g.sim_mean], fmt="s", color="C0",
                    ms=4, capsize=2, label="simulated (90%)")
        ax.plot(g.observed, y, "k|", ms=12, mew=2, label="observed")
        ax.set_yticks(y, labels if s == stats_[0] else [""] * len(y), fontsize=7)
        ax.set_title(s.replace("_", " "), fontsize=10)
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Behavioural summaries: observed vs simulated from fitted parameters", y=1.02)
    fig.tight_layout()
    _save(fig, out)


def paired_differences(scores_csv: Path, out: Path) -> None:
    """Figure 5b: per-agent held-out LL/trial of each fitted model minus the generating model
    (median and interquartile range; robust to rare catastrophic folds)."""
    d = pd.read_csv(scores_csv)
    order = list(PRIMARY_MODELS) + ["perseveration", "random"]
    fig, axes = plt.subplots(1, len(PRIMARY_MODELS), figsize=(17, 3.8), sharey=True)
    for ax, gen in zip(axes, PRIMARY_MODELS):
        g = d[d.generating_model == gen]
        conds = [c for c in CONDITION_ORDER if c in set(g.condition)]
        for k, cond in enumerate(conds):
            w = g[g.condition == cond].pivot(index="agent", columns="fitted_model", values="heldout_ll_per_trial")
            diff = w.sub(w[gen], axis=0)
            x = np.arange(len(order)) + (k - (len(conds) - 1) / 2) * 0.22
            med = diff[order].median()
            q1, q3 = diff[order].quantile(0.25), diff[order].quantile(0.75)
            ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o", ms=4, capsize=2,
                        color=CONDITION_COLORS[cond], label=cond)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhspan(-0.01, 0.01, color="grey", alpha=0.15, lw=0)
        ax.set_xticks(range(len(order)), [MODEL_LABELS[o] for o in order], rotation=40, ha="right", fontsize=8)
        ax.set_title(f"generated by {MODEL_LABELS[gen]}", fontsize=10)
        ax.set_ylim(-0.2, 0.05)
    axes[0].set_ylabel("held-out LL/trial minus\ngenerating model (median, IQR)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Paired held-out score differences (grey band: +-0.01 LL/trial tie zone)", y=1.03)
    fig.tight_layout()
    _save(fig, out)


# ---------------------------------------------------------------------------
# Post-hoc sensitivity analysis: locked vs bounded (beta <= 20, h < 0.5)
# ---------------------------------------------------------------------------
VARIANT_COLORS = {"locked": "#7f7f7f", "bounded": "#2ca02c"}


def sensitivity_bounded(mr_csv: Path, cat_csv: Path, pr_csv: Path, out: Path) -> None:
    mr = pd.read_csv(mr_csv)
    cat = pd.read_csv(cat_csv)
    pr = pd.read_csv(pr_csv)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    g = mr[(mr.condition == "all") & (mr.generating_model != "all")]
    x = np.arange(len(PRIMARY_MODELS))
    for k, v in enumerate(["locked", "bounded"]):
        h = g[g.variant == v].set_index("generating_model").reindex(list(PRIMARY_MODELS))
        ax.bar(x + (k - 0.5) * 0.38, h.best_raw_correct, width=0.36, color=VARIANT_COLORS[v], label=f"{v}: raw winner correct")
        ax.plot(x + (k - 0.5) * 0.38, h.tie_rate, "k" + ("o" if v == "locked" else "D"), ms=5,
                label=f"{v}: tie rate")
    ax.axhline(0.25, color="k", ls=":", lw=0.8)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in PRIMARY_MODELS], fontsize=8)
    ax.set(ylim=(0, 1), ylabel="proportion of agents", title="Model recovery, 160 trials (dotted: chance)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1]
    c = cat[cat.dataset == "model_recovery"]
    order = list(PRIMARY_MODELS) + ["perseveration"]
    x = np.arange(len(order))
    for k, v in enumerate(["locked", "bounded"]):
        h = c[c.variant == v].set_index("fitted_model").reindex(order)
        ax.bar(x + (k - 0.5) * 0.38, h.n_catastrophic, width=0.36, color=VARIANT_COLORS[v], label=v)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in order], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="agents with held-out LL/trial < -2", title="Catastrophic held-out scores (1,200 agents)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    p = pr[pr.n_trials == 160]
    for model, mk in zip(PRIMARY_MODELS, "osD^"):
        q = p[p.model == model]
        ax.scatter(q.spearman_rho_locked, q.spearman_rho_bounded, marker=mk, s=28, label=MODEL_LABELS[model])
    for _, r in p[p.model == "bayes_cp"].iterrows():
        ax.annotate(f"{r.parameter[:4]}/{r.condition[:3]}", (r.spearman_rho_locked, r.spearman_rho_bounded),
                    fontsize=6, xytext=(3, -8), textcoords="offset points")
    ax.plot([-0.1, 1], [-0.1, 1], "k--", lw=0.8)
    ax.set(xlim=(-0.1, 1), ylim=(-0.1, 1), xlabel="Spearman, locked", ylabel="Spearman, bounded",
           title="Parameter recovery, 160 trials (each point: parameter x condition)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("POST-HOC sensitivity: locked transforms vs bounded beta <= 20 and h < 0.5", y=1.03)
    fig.tight_layout()
    _save(fig, out)
