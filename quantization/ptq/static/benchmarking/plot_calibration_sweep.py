"""Plot every calibration run of the sweep, one panel per validation metric.

Reads calibration_sweep_runs.csv (and the FP32 row of ../results/validation_results.csv for the
reference lines); nothing is re-run. Writes plots/calibration_sweep.png and .pdf.

Usage: python quantization/ptq/static/benchmarking/plot_calibration_sweep.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1e1c", "#52514e", "#e6e5e0"
RUN, MEAN, SAVED, FP32 = "#86b6ef", "#2a78d6", "#eda100", "#52514e"
SAVED_RUN = (1000, 42)  # the calibration behind resnet18_static_int8.pt
PANELS = [  # column, title, FP32 reference column in validation_results.csv (x100) or None
    ("accuracy", "Accuracy (%)", "accuracy"),
    ("f1_macro", "Macro F1 (%)", "f1_macro"),
    ("mel_recall", "Melanoma recall (%)", "melanoma_recall"),
    ("identical_to_fp32", "Predictions identical to FP32 (%)", None),
    ("softmax_mad_x1000", "Mean |softmax INT8 - FP32| (x1000)", None),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "pdf.fonttype": 42, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "semibold",
})


def main():
    runs = pd.read_csv(HERE / "calibration_sweep_runs.csv")
    fp32 = pd.read_csv(HERE.parent / "results/validation_results.csv").set_index("model").loc["FP32"]
    means = runs.groupby("n_calibration").mean(numeric_only=True)
    sizes = sorted(runs["n_calibration"].unique())
    saved = runs[(runs["n_calibration"] == SAVED_RUN[0]) & (runs["seed"] == SAVED_RUN[1])]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for ax, (column, title, reference) in zip(axes.flat, PANELS):
        if reference:
            ax.axhline(fp32[reference] * 100, color=FP32, lw=1.5, ls=(0, (4, 3)))
        ax.plot(means.index, means[column], color=MEAN, lw=2, zorder=2)
        ax.scatter(runs["n_calibration"], runs[column], s=40, color=RUN, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(saved["n_calibration"], saved[column], s=90, marker="D", color=SAVED, edgecolor=SURFACE,
                   linewidth=1.5, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes], rotation=45, ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_title(title, loc="left")
        ax.margins(y=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Calibration images (log scale)")
    axes[0, 0].set_xlabel("")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=7, color=RUN, label="One calibration sample\n(seeds 42, 1, 2; one run at 6,981)"),
        Line2D([], [], color=MEAN, lw=2, label="Mean per size"),
        Line2D([], [], ls="", marker="D", ms=8, color=SAVED, label="Saved model\n(1,000 images, seed 42)"),
        Line2D([], [], color=FP32, lw=1.5, ls=(0, (4, 3)), label="FP32"),
    ], loc="center left", labelspacing=1.4, fontsize=10)
    fig.suptitle("Static INT8 PTQ: validation metrics barely change with calibration size", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "calibration_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "calibration_sweep.pdf", bbox_inches="tight")
    print(f"Wrote {OUT / 'calibration_sweep.png'} and .pdf")


if __name__ == "__main__":
    main()
