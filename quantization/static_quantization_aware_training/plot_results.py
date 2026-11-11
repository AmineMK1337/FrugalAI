"""Plots for the Static QAT INT8 results (validation split).

Reads the CSVs in results/ and writes PNG (300 dpi) and PDF to results/plots/:
- file_size
- validation_metrics
- weight_error

Usage:
    python quantization/static_quantization_aware_training/plot_results.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "plots"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1e1c", "#52514e", "#e6e5e0"
MODELS = {
    "FP32": ("FP32 (original)", "#2a78d6"),
    "QAT_INT8": ("Static QAT INT8", "#9c27b0"),
}
METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "melanoma_recall": "Melanoma recall",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titlesize": 12, "axes.titleweight": "semibold",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_file_size():
    csv_path = RESULTS / "file_sizes.csv"
    if not csv_path.exists():
        return
    sizes = pd.read_csv(csv_path).set_index("variant")
    keys = [k for k in ("QAT_INT8", "FP32") if k in sizes.index]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.barh([MODELS[k][0] for k in keys], sizes.loc[keys, "MB"], height=0.5, color=[MODELS[k][1] for k in keys])
    for i, k in enumerate(keys):
        ax.text(sizes.loc[k, "MB"] + 0.6, i, f"{sizes.loc[k, 'MB']:.2f} MB ({sizes.loc[k, 'pct_of_fp32']:.1f}%)",
                va="center", color=INK)
    ax.set_xlim(0, sizes["MB"].max() * 1.3)
    ax.set_xlabel("File size (MB)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Static QAT INT8 Model File Size vs FP32", loc="left")
    save(fig, "file_size")


def plot_validation_metrics():
    csv_path = RESULTS / "validation_results.csv"
    if not csv_path.exists():
        return
    results = pd.read_csv(csv_path).set_index("model")
    x, width = np.arange(len(METRICS)), 0.32
    fig, ax = plt.subplots(figsize=(8, 4.2))
    keys = [k for k in ("FP32", "QAT_INT8") if k in results.index]
    for offset, key in zip((-0.5, 0.5), keys):
        label, colour = MODELS[key]
        values = results.loc[key, list(METRICS)].to_numpy(dtype=float) * 100
        bars = ax.bar(x + offset * width, values, width * 0.9, label=label, color=colour)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK)
    ax.set_xticks(x, list(METRICS.values()))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score on validation split (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Validation Metrics: FP32 vs Static QAT INT8", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    save(fig, "validation_metrics")


def plot_weight_error():
    csv_path = RESULTS / "weight_errors.csv"
    if not csv_path.exists():
        return
    errors = pd.read_csv(csv_path)
    if errors.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.4))
    layers = errors["layer"].str.replace(".weight", "", regex=False)
    ax.plot(layers, errors["relative_error"] * 100, "-o", lw=2, ms=6, color=MODELS["QAT_INT8"][1],
            label="Static QAT INT8", markeredgecolor=SURFACE, markeredgewidth=1)
    ax.tick_params(axis="x", rotation=60)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Relative weight error (%)")
    ax.set_xlabel("Layer")
    ax.grid(axis="x", visible=False)
    ax.set_title("Relative Weight Error Per Layer (INT8 vs FP32)", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    save(fig, "weight_error")


if __name__ == "__main__":
    plot_file_size()
    plot_validation_metrics()
    plot_weight_error()
    print(f"Wrote figures to {OUT}")

