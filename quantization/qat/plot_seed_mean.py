"""INT8 QAT vs FP32, averaged over the three training seeds (42, 1, 2): one summary figure.

Left:   change in each performance metric vs FP32 (percentage points): one dot per seed, the mean as a large marker.
Middle: FP32 -> QAT mean, the mean change, and in how many seeds it is significant (paired bootstrap, p < 0.05).
Right:  mean % change vs FP32 in size, latency (1 and 4 threads), peak RAM and estimated energy.

Reads quantization/qat/results/evaluation_seeds/comparison/*/table_validation.csv (evaluation/evaluate.py,
--models fp32 qat qat_seed1 qat_seed2). Writes quantization/qat/results/evaluation_seeds/
qat_mean_of_3_seeds_validation.{png,pdf}.

Usage: python quantization/qat/plot_seed_mean.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "evaluation_seeds"
RUNS = ["qat", "qat_seed1", "qat_seed2"]  # training seeds 42, 1, 2
QAT_COLOR, FP32_COLOR = "#f47230", "#257de0"  # pair validated for colour-blind separation (dataviz validator)
INK, MUTED, GRID = "#1f1e1c", "#5f6368", "#e6e6e6"
PERFORMANCE = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
               ("precision_macro", "Macro precision"), ("specificity_macro", "Macro specificity"),
               ("f1_macro", "Macro F1"), ("auc_macro", "Macro AUC"), ("melanoma_recall", "Melanoma recall")]


def load(kind):
    return pd.read_csv(RESULTS / "comparison" / kind / "table_validation.csv").set_index("model")


def main():
    perf, eff, env = load("performance"), load("efficiency"), load("environment")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.edgecolor": "#b0b0b0", "xtick.color": MUTED, "ytick.color": INK})
    fig = plt.figure(figsize=(15, 5.4))
    ax1 = fig.add_axes([0.12, 0.16, 0.29, 0.66])   # performance dots
    axt = fig.add_axes([0.42, 0.16, 0.22, 0.66])   # numbers
    ax2 = fig.add_axes([0.79, 0.16, 0.19, 0.66])   # efficiency bars
    n = len(PERFORMANCE)
    y = np.arange(n)[::-1]
    ylim = (-0.6, n - 0.4)

    # ── Left: performance change vs FP32 ──
    changes_all = perf.loc[RUNS, [f"{k}_change_vs_fp32" for k, _ in PERFORMANCE]].to_numpy() * 100
    lim = max(abs(changes_all).max(), 2) * 1.2
    for yi, (key, _) in zip(y, PERFORMANCE):
        changes = perf.loc[RUNS, f"{key}_change_vs_fp32"].to_numpy() * 100
        ax1.plot([changes.min(), changes.max()], [yi, yi], color=QAT_COLOR, linewidth=2, alpha=0.45,
                 solid_capstyle="round", zorder=2)
        ax1.scatter(changes, [yi] * len(changes), s=36, color=QAT_COLOR, alpha=0.6, edgecolor="white",
                    linewidth=1, zorder=3)
        ax1.scatter([changes.mean()], [yi], s=150, color=QAT_COLOR, edgecolor=INK, linewidth=1.3, zorder=4)
    ax1.axvline(0, color=FP32_COLOR, linewidth=1.6, zorder=1)
    ax1.text(0.15, ylim[0] + 0.05, "FP32 baseline", color=FP32_COLOR, fontsize=8.5, va="bottom")
    ax1.set_yticks(y, [label for _, label in PERFORMANCE])
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(*ylim)
    ax1.set_xlabel("Change vs FP32 (percentage points)\nright of the blue line = QAT higher")
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("Performance", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=26)
    # key, above the plot
    ax1.scatter([0.02], [1.035], s=36, color=QAT_COLOR, alpha=0.6, edgecolor="white", transform=ax1.transAxes,
                clip_on=False)
    ax1.text(0.05, 1.035, "one training seed", transform=ax1.transAxes, va="center", fontsize=8.5, color=MUTED)
    ax1.scatter([0.42], [1.035], s=110, color=QAT_COLOR, edgecolor=INK, linewidth=1.2, transform=ax1.transAxes,
                clip_on=False)
    ax1.text(0.46, 1.035, "mean of the 3 seeds", transform=ax1.transAxes, va="center", fontsize=8.5, color=MUTED)

    # ── Middle: the numbers ──
    axt.set_xlim(0, 1)
    axt.set_ylim(*ylim)
    axt.axis("off")
    cols = (0.02, 0.50, 0.75)
    for x, head in zip(cols, ("FP32 → QAT mean (%)", "Change", "Significant")):
        axt.text(x, n - 0.45, head, fontsize=8.8, color=MUTED, va="bottom")
    for yi, (key, _) in zip(y, PERFORMANCE):
        fp32 = perf.loc["fp32", key] * 100
        qat = perf.loc[RUNS, key].mean() * 100
        sig = int((perf.loc[RUNS, f"{key}_paired_bootstrap_p"] < 0.05).sum())
        axt.text(cols[0], yi, f"{fp32:4.1f} → {qat:4.1f}", va="center", fontsize=9.8, color=INK,
                 fontfamily="DejaVu Sans Mono")
        axt.text(cols[1], yi, f"{qat - fp32:+.1f} pts", va="center", fontsize=9.8, color=INK,
                 fontfamily="DejaVu Sans Mono")
        axt.text(cols[2], yi, f"{sig}/3 seeds", va="center", fontsize=9.8, color=INK if sig else MUTED,
                 fontweight="bold" if sig else "normal")

    # ── Right: efficiency and energy ──
    def pct(table, col):
        return (table.loc[RUNS, col] / table.loc["fp32", col] - 1).to_numpy() * 100

    items = [("File size", pct(eff, "size_mb"), "4× smaller"),
             ("Latency, 1 thread", pct(eff, "latency_median_ms_1t"),
              f"{eff.loc[RUNS, 'speedup_vs_fp32_1t'].mean():.1f}× faster"),
             ("Latency, 4 threads", pct(eff, "latency_median_ms_4t"),
              f"{eff.loc[RUNS, 'speedup_vs_fp32_4t'].mean():.1f}× faster"),
             ("Peak RAM", pct(eff, "peak_memory_mb_1t"), ""),
             ("Energy / 1,000 images", pct(env, "energy_wh_per_1000"), "estimate")]
    y2 = np.arange(len(items))[::-1]
    for yi, (label, values, note) in zip(y2, items):
        ax2.barh(yi, values.mean(), height=0.52, color=QAT_COLOR, edgecolor="white", linewidth=2)
        ax2.text(3, yi, f"{values.mean():.0f}%" + (f"  ({note})" if note else ""), va="center", fontsize=9.5,
                 color=INK)
    ax2.axvline(0, color=FP32_COLOR, linewidth=1.6)
    ax2.set_yticks(y2, [label for label, _, _ in items])
    ax2.set_xlim(-100, 0)
    ax2.set_xticks([-100, -75, -50, -25, 0])
    ax2.set_ylim(-0.6, len(items) - 0.4)
    ax2.set_xlabel("Mean change vs FP32 (%)\nlower is better")
    ax2.grid(axis="x", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Efficiency and energy", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=26)

    f1q, f1f = perf.loc[RUNS, "f1_macro"].mean() * 100, perf.loc["fp32", "f1_macro"] * 100
    fig.text(0.02, 0.97,
             f"INT8 QAT, mean of 3 training seeds: same macro F1 as FP32 ({f1q:.1f} vs {f1f:.1f}), "
             f"{eff.loc[RUNS, 'speedup_vs_fp32_1t'].mean():.1f}× faster and "
             f"{-pct(env, 'energy_wh_per_1000').mean():.0f}% less energy", fontsize=13.5, fontweight="bold",
             color=INK, va="top")
    fig.text(0.02, 0.0,
             "Validation split (1,532 images), CPU (Intel Xeon Gold 6348), fused TorchScript. Seeds 42, 1 and 2 change "
             "only the QAT training randomness; all three start from the same static-PTQ model.\nSignificant: paired "
             "bootstrap vs FP32 (1,000 resamples), p < 0.05. The three seeds share one architecture, so their size, "
             "speed, RAM and energy agree within about 1%. Energy is a CodeCarbon estimate (no power counter), so it "
             "follows run time.", fontsize=8.3, color=MUTED, va="top")

    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"qat_mean_of_3_seeds_validation.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", RESULTS / "qat_mean_of_3_seeds_validation.png")


if __name__ == "__main__":
    main()
