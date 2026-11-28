"""Simple FP32 vs INT8 QAT plots: plain side-by-side bars, real values on every bar, the answer in the title.

QAT = mean of the three training seeds (42, 1, 2); FP32 = the same session's measurement.
One question per plot:
  1_accuracy  -> does QAT keep the accuracy?
  2_size      -> how much smaller is the file?
  3_speed     -> how much faster is one prediction?
  4_energy    -> how much less energy?
  overview    -> the four together

Reads quantization/qat/results/evaluation_seeds/comparison/*/table_validation.csv.
Writes quantization/qat/results/evaluation_seeds/simple/<name>_validation.{png,pdf}.

Usage: python quantization/qat/plot_simple.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "evaluation_seeds"
OUT = RESULTS / "simple"
RUNS = ["qat", "qat_seed1", "qat_seed2"]
FP32_COLOR, QAT_COLOR = "#257de0", "#f47230"  # pair validated for colour-blind separation (dataviz validator)
INK, MUTED, GRID = "#1f1e1c", "#5f6368", "#e6e6e6"
FP32_LABEL, QAT_LABEL = "FP32 (original)", "INT8 QAT (mean of 3 seeds)"


def load(kind):
    return pd.read_csv(RESULTS / "comparison" / kind / "table_validation.csv").set_index("model")


PERF, EFF, ENV = load("performance"), load("efficiency"), load("environment")


def fp32(table, col):
    return float(table.loc["fp32", col])


def qat(table, col):
    return float(table.loc[RUNS, col].mean())


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, colors=MUTED)
    ax.tick_params(axis="x", length=0, colors=INK, labelsize=10.5)


def pair_bars(ax, groups, fp32_values, qat_values, fmt, ymax):
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, fp32_values, w, color=FP32_COLOR, label=FP32_LABEL)
    b2 = ax.bar(x + w / 2 + 0.01, qat_values, w, color=QAT_COLOR, label=QAT_LABEL)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.015, fmt(bar.get_height()),
                    ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ymax)
    return x, w


def draw_accuracy(ax):
    metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced\naccuracy"), ("f1_macro", "Macro F1"),
               ("auc_macro", "AUC"), ("melanoma_recall", "Melanoma\nrecall")]
    f = [fp32(PERF, k) * 100 for k, _ in metrics]
    q = [qat(PERF, k) * 100 for k, _ in metrics]
    pair_bars(ax, [label for _, label in metrics], f, q, lambda v: f"{v:.1f}", 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("%  (higher is better)", color=MUTED)
    style(ax, "Accuracy: QAT is as good as FP32\n(melanoma recall is higher)")


def draw_size(ax):
    f, q = fp32(EFF, "size_mb"), qat(EFF, "size_mb")
    ax.bar([0], [f], 0.5, color=FP32_COLOR, label=FP32_LABEL)
    ax.bar([1], [q], 0.5, color=QAT_COLOR, label=QAT_LABEL)
    for xi, v in ((0, f), (1, q)):
        ax.text(xi, v + f * 0.02, f"{v:.1f} MB", ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_xticks([0, 1], ["FP32", "INT8 QAT"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, f * 1.18)
    ax.set_ylabel("MB  (lower is better)", color=MUTED)
    style(ax, f"File size: {f / q:.0f}× smaller")


def draw_speed(ax):
    groups = ["1 CPU thread", "4 CPU threads"]
    f = [fp32(EFF, "latency_median_ms_1t"), fp32(EFF, "latency_median_ms_4t")]
    q = [qat(EFF, "latency_median_ms_1t"), qat(EFF, "latency_median_ms_4t")]
    ymax = f[0] * 1.2
    x, w = pair_bars(ax, groups, f, q, lambda v: f"{v:.1f} ms", ymax)
    for xi, fi, qi in zip(x, f, q):
        ax.text(xi + w / 2 + 0.01, qi + ymax * 0.09, f"{fi / qi:.1f}×\nfaster", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=INK)
    ax.set_ylabel("ms per image  (lower is better)", color=MUTED)
    style(ax, f"Time per image: {f[0] / q[0]:.1f}× faster")


def draw_energy(ax):
    f, q = fp32(ENV, "energy_wh_per_1000"), qat(ENV, "energy_wh_per_1000")
    ax.bar([0], [f], 0.5, color=FP32_COLOR, label=FP32_LABEL)
    ax.bar([1], [q], 0.5, color=QAT_COLOR, label=QAT_LABEL)
    for xi, v in ((0, f), (1, q)):
        ax.text(xi, v + f * 0.02, f"{v:.2f} Wh", ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_xticks([0, 1], ["FP32", "INT8 QAT"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, f * 1.18)
    ax.set_ylabel("Wh per 1,000 images  (lower is better)", color=MUTED)
    style(ax, f"Energy: {(1 - q / f) * 100:.0f}% less (estimate)")


NOTE = ("Validation set (1,532 images), CPU (Intel Xeon Gold 6348), one image at a time. "
        "QAT = mean of 3 training seeds. Energy is a CodeCarbon estimate.")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    singles = [("1_accuracy", draw_accuracy, (8.5, 4.6)), ("2_size", draw_size, (4.6, 4.4)),
               ("3_speed", draw_speed, (6.2, 4.6)), ("4_energy", draw_energy, (4.6, 4.4))]
    notes = {"1_accuracy": "Validation set (1,532 images). QAT = mean of 3 training seeds.",
             "2_size": "Size of the saved model file.\nQAT: the same for all 3 seeds.",
             "3_speed": "CPU (Intel Xeon Gold 6348), one image at a time.\nQAT = mean of 3 training seeds.",
             "4_energy": "CodeCarbon estimate, 1 CPU thread.\nQAT = mean of 3 training seeds."}
    for name, draw, size in singles:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        grouped = name in ("1_accuracy", "3_speed")  # only grouped bars need a legend
        if grouped:
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False, fontsize=9.5)
        fig.text(0.01, -0.12 if grouped else -0.02, notes[name], fontsize=8.5, color=MUTED, va="top")
        save(fig, name)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5), gridspec_kw={"width_ratios": [2.4, 1, 1.5, 1], "wspace": 0.38})
    for ax, (_, draw, _) in zip(axes, singles):
        draw(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.12, 1.06), ncol=2, frameon=False, fontsize=11)
    fig.text(0.12, -0.02, NOTE, fontsize=9, color=MUTED, va="top")
    save(fig, "overview")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
