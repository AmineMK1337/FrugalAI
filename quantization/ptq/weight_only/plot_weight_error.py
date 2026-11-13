"""Per-layer weight error of the weight-only INT8 models (nothing is re-run).

Reads results/weight_errors.csv and writes PNG (300 dpi) and PDF to results/plots/weight_error.
Size and validation metrics are plotted under evaluation/ (evaluate.py, plot_results.py), not here.

Usage: python quantization/ptq/weight_only/plot_weight_error.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "plots"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1e1c", "#52514e", "#e6e5e0"
MODELS = {  # the colour follows the model in every figure
    "FP32": ("FP32 (original)", "#2a78d6"),
    "per_channel": ("INT8 per-channel", "#eb6834"),
    "per_tensor": ("INT8 per-tensor", "#1baf7a"),
}
METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1",
           "melanoma_recall": "Melanoma recall"}

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


def plot_weight_error():
    errors = pd.read_csv(RESULTS / "weight_errors.csv")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for key in ("per_channel", "per_tensor"):
        rows = errors[errors["variant"] == key]
        layers = rows["layer"].str.replace(".weight", "", regex=False)
        ax.plot(layers, rows["relative_error"] * 100, "-o", lw=2, ms=6, color=MODELS[key][1], label=MODELS[key][0],
                markeredgecolor=SURFACE, markeredgewidth=1)
    ax.tick_params(axis="x", rotation=60)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Relative weight error (%)")
    ax.set_xlabel("Layer")
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-channel loses less precision in every layer", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    save(fig, "weight_error")


if __name__ == "__main__":
    plot_weight_error()
    print(f"Wrote weight_error to {OUT}")
