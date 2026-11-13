"""Plots for evaluation/results/: one set per model and metric type, plus the cross-model comparison.

Only the evaluation metrics are plotted: performance (accuracy, precision, recall, specificity, F1, AUC),
efficiency (size, FLOPs, latency, throughput, memory) and environment (energy, CO2e). Reads only the CSVs
written by evaluate.py (nothing is re-run). Every plot is saved as PDF and 300-dpi PNG next to its CSV.
Each model keeps the same colour, hatch and marker in every plot, so the figures stay readable in black and white.

Usage: python evaluation/plot_results.py [--split validation|test]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
INK, NOTE = "#1f1e1c", "#555555"
STYLE = {  # the same look for a model in every plot
    "fp32": dict(label="FP32", short="FP32", color="#2a78d6", hatch=""),
    "weight_only_per_channel": dict(label="Weight-only INT8 per-channel", short="Weight-only\nper-channel",
                                    color="#eb6834", hatch="///"),
    "weight_only_per_tensor": dict(label="Weight-only INT8 per-tensor", short="Weight-only\nper-tensor",
                                   color="#1baf7a", hatch="\\\\\\"),
    "static_ptq": dict(label="Static INT8 PTQ", short="Static\nINT8 PTQ", color="#eda100", hatch="xxx"),
}
MODELS = list(STYLE)
THREADS = (1, 4)
FORMAT = "fused + TorchScript"
PERFORMANCE = {"accuracy": "Accuracy", "precision_macro": "Precision", "recall_macro": "Recall",
               "specificity_macro": "Specificity", "f1_macro": "F1-score", "auc_macro": "AUC (x100)"}
ESTIMATE_NOTE = "CodeCarbon estimate: no power counter on this VM, so it is roughly an assumed CPU power x run time."

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "#b0b0b0", "axes.grid": True, "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False, "pdf.fonttype": 42,
    "hatch.linewidth": 0.8, "xtick.major.size": 0,
})


def bar_style(model):
    return dict(color=STYLE[model]["color"], hatch=STYLE[model]["hatch"], edgecolor=INK, linewidth=0.6)


def save(fig, folder, name, split):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}_{split}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}_{split}.pdf", bbox_inches="tight")
    plt.close(fig)


def read(model, kind, name, split):
    return pd.read_csv(RESULTS / model / kind / f"{name}_{split}.csv")


def thread_labels():
    return [f"{t} thread{'s' if t > 1 else ''}" for t in THREADS]


def ci_errors(row):
    """95% bootstrap CI as error bars (x100)."""
    return np.array([[(row[m] - row[f"{m}_ci_low"]) * 100 for m in PERFORMANCE],
                     [(row[f"{m}_ci_high"] - row[m]) * 100 for m in PERFORMANCE]])


def thread_bars(ax, model, values, fmt):
    bars = ax.bar(range(len(THREADS)), values, 0.6, **bar_style(model))
    ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=2, fontsize=8, color=INK)
    ax.set_xticks(range(len(THREADS)), thread_labels(), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)


def grouped_by_thread(ax, table, prefix, fmt):
    """One group per thread count, one bar per model."""
    x, width = np.arange(len(THREADS)), 0.8 / len(MODELS)
    for k, model in enumerate(MODELS):
        values = [table.loc[model, f"{prefix}_{t}t"] for t in THREADS]
        bars = ax.bar(x + (k - (len(MODELS) - 1) / 2) * width, values, width * 0.92, label=STYLE[model]["label"],
                      **bar_style(model))
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=2, fontsize=7, rotation=90, color=INK)
    ax.set_xticks(x, thread_labels())
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)


def model_bars(ax, values, fmt):
    """One bar per model."""
    for k, model in enumerate(MODELS):
        bars = ax.bar(k, values[model], 0.6, **bar_style(model))
        ax.bar_label(bars, labels=[fmt.format(values[model])], padding=2, fontsize=8.5, color=INK)
    ax.set_xticks(range(len(MODELS)), [STYLE[m]["short"] for m in MODELS])
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)


def box(ax, data, models, labels):
    parts = ax.boxplot(data, patch_artist=True, widths=0.55, flierprops=dict(marker=".", markersize=3, alpha=0.5),
                       medianprops=dict(color=INK, linewidth=1.5))
    for patch, model in zip(parts["boxes"], models):
        patch.set(facecolor=STYLE[model]["color"], hatch=STYLE[model]["hatch"], edgecolor=INK, linewidth=0.8)
    top = max(v.max() for v in data)
    for i, values in enumerate(data, start=1):
        ax.text(i, values.max() + top * 0.03, f"{np.median(values):.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(1, len(data) + 1), labels, fontsize=8.5)
    ax.set_ylim(0, top * 1.15)
    ax.grid(axis="x", visible=False)


# ── Per model ────────────────────────────────────────────────────────────────

def model_plots(model, split):
    label = STYLE[model]["label"]

    row = read(model, "performance", "metrics", split).iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    values = np.array([row[m] * 100 for m in PERFORMANCE])
    bars = ax.bar(np.arange(len(PERFORMANCE)), values, 0.6, yerr=ci_errors(row),
                  error_kw=dict(ecolor=INK, elinewidth=1, capsize=4), **bar_style(model))
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=8, fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(PERFORMANCE)), PERFORMANCE.values())
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%), macro average except accuracy")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{label}: performance ({FORMAT}, {split})", loc="left")
    ax.text(0, -0.14, "Error bars: 95% bootstrap CI (1,000 resamples).", transform=ax.transAxes, fontsize=8,
            color=NOTE)
    save(fig, RESULTS / model / "performance", "overall_metrics", split)

    folder = RESULTS / model / "efficiency"
    samples = read(model, "efficiency", "latency_samples", split)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    box(ax, [samples.loc[samples["threads"] == t, "latency_ms"] for t in THREADS], [model] * len(THREADS),
        thread_labels())
    ax.set_ylabel("Latency per image, batch 1 (ms)")
    ax.set_title(f"{label}: latency ({split} image; number = median, ms)", loc="left")
    ax.text(0, -0.14, f"Each box: 600 timed runs (3 repeats x 200). Format: {FORMAT}.", transform=ax.transAxes, fontsize=8, color=NOTE)
    save(fig, folder, "latency_distribution", split)

    row = read(model, "efficiency", "metrics", split).iloc[0]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw={"width_ratios": [0.6, 0.6, 1.4, 1.4]})
    for ax, (title, value, fmt) in zip(axes[:2], [("Size (MB)", row["size_mb"], "{:.1f}"),
                                                  ("FLOPs (G)", row["gflops"], "{:.2f}")]):
        bars = ax.bar([0], [value], 0.5, **bar_style(model))
        ax.bar_label(bars, labels=[fmt.format(value)], padding=2, fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    thread_bars(axes[2], model, [row[f"throughput_img_s_{t}t"] for t in THREADS], "{:.1f}")
    axes[2].set_title("Throughput, batch 32 (images/s)", loc="left", fontsize=10)
    thread_bars(axes[3], model, [row[f"peak_memory_mb_{t}t"] for t in THREADS], "{:.0f}")
    axes[3].set_title("Peak RAM (MB)", loc="left", fontsize=10)
    fig.suptitle(f"{label}: efficiency ({FORMAT}, {split})", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save(fig, folder, "efficiency_summary", split)

    row = read(model, "environment", "metrics", split).iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, (title, value) in zip(axes, [("Energy per 1,000 images (Wh)", row["energy_wh_per_1000"]),
                                         ("CO2e per 1,000 images (g)", row["co2e_g_per_1000"])]):
        bars = ax.bar([0], [value], 0.5, **bar_style(model))
        ax.bar_label(bars, labels=[f"{value:.3f}"], padding=2, fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle(f"{label}: energy and CO2e (estimate, 1 thread, {split})", x=0.01, ha="left",
                 fontweight="semibold")
    fig.text(0.01, -0.06, f"{ESTIMATE_NOTE}\nAt least 60 s per run. Grid {row['grid_g_per_kwh']:.0f} gCO2e/kWh ({row['country']}).",
             fontsize=8, color=NOTE)
    fig.tight_layout()
    save(fig, RESULTS / model / "environment", "energy_co2e", split)


# ── Comparison ───────────────────────────────────────────────────────────────

def comparison(split):
    out = RESULTS / "comparison"
    perf = pd.read_csv(out / "performance" / f"table_{split}.csv").set_index("model")
    eff = pd.read_csv(out / "efficiency" / f"table_{split}.csv").set_index("model")
    env = pd.read_csv(out / "environment" / f"table_{split}.csv").set_index("model")

    fig, ax = plt.subplots(figsize=(12, 5))
    x, width = np.arange(len(PERFORMANCE)), 0.8 / len(MODELS)
    for k, model in enumerate(MODELS):
        values = np.array([perf.loc[model, m] * 100 for m in PERFORMANCE])
        bars = ax.bar(x + (k - (len(MODELS) - 1) / 2) * width, values, width * 0.92, yerr=ci_errors(perf.loc[model]),
                      error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5), label=STYLE[model]["label"],
                      **bar_style(model))
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=6, fontsize=7, rotation=90, color=INK)
    ax.set_xticks(x, PERFORMANCE.values())
    ax.set_ylim(0, 118)
    ax.set_ylabel("Score (%), macro average except accuracy")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Performance, all models ({FORMAT}, {split})", loc="left", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
    ax.text(0, -0.12, "Error bars: 95% bootstrap CI (1,000 resamples). Static INT8 PTQ is the saved seed-42 model.",
            transform=ax.transAxes, fontsize=8, color=NOTE)
    save(fig, out / "performance", "metrics_all_models", split)

    for name, column, ylabel, title, fmt, note in [
            ("size", "size_mb", "File size (MB)", "Model file size", "{:.1f}", None),
            ("flops", "gflops", "FLOPs per image (G)", "FLOPs per image", "{:.2f}",
             "FLOPs = 2 x MACs (ptflops). Static INT8 PTQ counted on the FP32 architecture; its operations are INT8.")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        model_bars(ax, eff[column], fmt)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if note:
            ax.text(0, -0.24, note, transform=ax.transAxes, fontsize=8, color=NOTE)
        save(fig, out / "efficiency", name, split)

    samples = pd.concat([read(m, "efficiency", "latency_samples", split) for m in MODELS])
    fig, axes = plt.subplots(1, len(THREADS), figsize=(12, 4.6))
    for ax, threads, title in zip(axes, THREADS, thread_labels()):
        box(ax, [samples.loc[(samples["model"] == m) & (samples["threads"] == threads), "latency_ms"] for m in MODELS],
            MODELS, [STYLE[m]["short"] for m in MODELS])
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_ylabel("Latency per image, batch 1 (ms)")
    fig.suptitle(f"Latency, all models ({FORMAT}, {split} image; number = median, ms)", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    save(fig, out / "efficiency", "latency", split)

    for folder, table, name, prefix, ylabel, title, fmt, note in [
            ("efficiency", eff, "throughput", "throughput_img_s", "Throughput, batch 32 (images/s)", "Throughput",
             "{:.1f}", None),
            ("efficiency", eff, "memory", "peak_memory_mb", "Peak RAM (MB)", "Peak RAM (psutil; load + batch-1 warm-up)",
             "{:.0f}", None),
]:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        grouped_by_thread(ax, table, prefix, fmt)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=46)
        if note:
            ax.text(0, -0.13, note, transform=ax.transAxes, fontsize=8, color=NOTE)
        save(fig, out / folder, name, split)

    for name, column, ylabel, title in [
            ("energy_per_1000", "energy_wh_per_1000", "Energy per 1,000 images (Wh)", "Energy per 1,000 images"),
            ("co2e_per_1000", "co2e_g_per_1000", "CO2e per 1,000 images (g)", "CO2e per 1,000 images")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        model_bars(ax, env[column], "{:.3f}")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (estimate, 1 thread, {FORMAT})", loc="left")
        ax.text(0, -0.24, ESTIMATE_NOTE + "\nAt least 60 s per run; median of 3 repeats.", transform=ax.transAxes,
                fontsize=8, color=NOTE)
        save(fig, out / "environment", name, split)


def main(split="validation"):
    for model in MODELS:
        model_plots(model, split)
    comparison(split)
    print(f"Plots written under {RESULTS}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    main(parser.parse_args().split)
