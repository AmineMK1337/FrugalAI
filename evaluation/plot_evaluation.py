"""Plots of the evaluate.py results: performance, efficiency, environment (nothing is re-run).

Bars of run-based metrics are medians of the repeats; the thin line is the min-max range.
Writes PNG (300 dpi) and PDF to evaluation/results/<split>/plots/.

Usage: python evaluation/plot_evaluation.py [--split validation|test]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1e1c", "#52514e", "#e6e5e0"
MODELS = {  # the colour follows the model in every figure; add PTQ and QAT here when they exist
    "FP32": ("FP32", "#2a78d6"),
    "weight_only_per_channel": ("Weight-only per-channel", "#eb6834"),
    "weight_only_per_tensor": ("Weight-only per-tensor", "#1baf7a"),
}
SCORES = {"accuracy": "Accuracy", "precision_macro": "Precision", "recall_macro": "Recall",
          "specificity_macro": "Specificity", "f1_macro": "F1-score", "auc_macro": "AUC"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titlesize": 10.5, "figure.titlesize": 12, "figure.titleweight": "semibold",
})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_performance(results, out):
    table = pd.read_csv(results / "performance.csv").set_index("model")
    x, width = np.arange(len(SCORES)), 0.26
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for offset, (key, (label, colour)) in zip((-1, 0, 1), MODELS.items()):
        bars = ax.bar(x + offset * width, table.loc[key, list(SCORES)].to_numpy(dtype=float) * 100, width * 0.92,
                      label=label, color=colour)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5, color=INK)
    ax.set_xticks(x, SCORES.values())
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%), macro average except accuracy")
    ax.grid(axis="x", visible=False)
    gap = (table[list(SCORES)] - table.loc["FP32", list(SCORES)]).abs().max().max() * 100
    ax.set_title(f"Performance: every score stays within {np.ceil(gap):.0f} points of FP32", loc="left", pad=26,
                 fontsize=12, fontweight="semibold")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    save(fig, out, "performance")


def panels(title, specs, efficiency, runs, out, name, width):
    """One small panel per metric: the units differ, so they never share an axis."""
    keys = list(MODELS)[::-1]
    fig, axes = plt.subplots(1, len(specs), figsize=(width, 2.6), sharey=True)
    for ax, (column, label, decimals) in zip(np.atleast_1d(axes), specs):
        from_runs = column in runs.columns
        values = [(runs[runs["model"] == k][column].median() if from_runs else efficiency.loc[k, column]) for k in keys]
        ax.barh([MODELS[k][0] for k in keys], values, height=0.6, color=[MODELS[k][1] for k in keys])
        ends = values
        if from_runs:
            lows = [runs[runs["model"] == k][column].min() for k in keys]
            highs = [runs[runs["model"] == k][column].max() for k in keys]
            ax.hlines(range(len(keys)), lows, highs, color=INK, lw=1.2)
            ends = highs
        for i, (end, value) in enumerate(zip(ends, values)):
            ax.text(end + max(values) * 0.03, i, f"{value:.{decimals}f}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, max(values) * 1.38)
        ax.set_title(label, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, name)


def main(split):
    results = Path(__file__).resolve().parent / "results" / split
    out = results / "plots"
    efficiency = pd.read_csv(results / "efficiency.csv").set_index("model")
    runs = pd.read_csv(results / "runs.csv")
    plot_performance(results, out)
    panels("Efficiency: quantizing only the stored weights shrinks the file, nothing else improves", [
        ("size_mb", "File size (MB)", 1), ("gflops", "FLOPs (G)", 2), ("latency_median_ms", "Latency (ms)", 1),
        ("throughput_img_s", "Throughput (img/s)", 1), ("peak_memory_mb", "Peak memory (MB)", 1),
    ], efficiency, runs, out, "efficiency", 13)
    panels("Environment (estimated, not measured): no real difference between the models", [
        ("energy_wh_per_1000", "Energy (Wh per 1,000 images)", 2), ("co2e_g_per_1000", "CO₂e (g per 1,000 images)", 2),
    ], efficiency, runs, out, "environment", 8.5)
    print(f"Wrote 3 figures to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    main(parser.parse_args().split)
