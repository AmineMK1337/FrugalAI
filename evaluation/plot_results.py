"""Plots for evaluation/results/: one set per model and metric type, plus the cross-model comparison.

Only the evaluation metrics are plotted: performance (accuracy, precision, recall, specificity, F1, AUC),
efficiency (size, FLOPs, latency, throughput, memory) and environment (energy, CO2e). Every quantized model is
compared with the FP32 baseline: its per-model plots show FP32 next to it, plus change_vs_fp32 (performance, paired
bootstrap) and vs_fp32_<split>.csv tables in each metric folder; the all-model plots put FP32 first in each group
and draw its value as a dashed reference line. Reads only the CSVs
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
    "qat": dict(label="INT8 QAT", short="INT8\nQAT", color="#8b5cc7", hatch="..."),
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
    fp32_reference(ax, x, [table.loc["fp32", f"{prefix}_{t}t"] for t in THREADS])
    ax.set_xticks(x, thread_labels())
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)


def fp32_reference(ax, centers, values, half_width=0.42):
    """Dashed FP32 level across each group, so every quantized bar is read against the baseline."""
    for c, v in zip(centers, values):
        ax.hlines(v, c - half_width, c + half_width, colors=STYLE["fp32"]["color"], linewidth=1, linestyles="--",
                  zorder=3)


def model_bars(ax, values, fmt):
    """One bar per model."""
    for k, model in enumerate(MODELS):
        bars = ax.bar(k, values[model], 0.6, **bar_style(model))
        ax.bar_label(bars, labels=[fmt.format(values[model])], padding=2, fontsize=8.5, color=INK)
    ax.axhline(values["fp32"], color=STYLE["fp32"]["color"], linewidth=1, linestyle="--", zorder=0.5)
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


# ── Per model: every quantized model is drawn next to the FP32 baseline ──────────

def shown(model):
    """Models in a per-model plot: FP32 alone, or FP32 then the quantized model, side by side."""
    return ["fp32"] if model == "fp32" else ["fp32", model]


def paired_bars(ax, models, groups, values, fmt, yerr=None, rotation=0, fontsize=8.5):
    """One group per entry of `groups`, one bar per model inside it (FP32 first), bars touching."""
    x, width = np.arange(len(groups)), 0.8 / len(models)
    for k, model in enumerate(models):
        bars = ax.bar(x + (k - (len(models) - 1) / 2) * width, values[model], width * 0.95,
                      yerr=None if yerr is None else yerr[model], label=STYLE[model]["label"],
                      error_kw=dict(ecolor=INK, elinewidth=1, capsize=3), **bar_style(model))
        ax.bar_label(bars, labels=[fmt.format(v) for v in values[model]], padding=6 if yerr is not None else 2,
                     fontsize=fontsize, rotation=rotation, color=INK)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def title_of(model, what, split):
    return (f"{STYLE[model]['label']}: {what} ({split})" if model == "fp32"
            else f"{STYLE[model]['label']} vs FP32: {what} ({split})")


def model_plots(model, split):
    models = shown(model)
    legend = dict(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)

    # Performance: FP32 and the model side by side for each metric, 95% bootstrap CIs
    rows = {m: read(m, "performance", "metrics", split).iloc[0] for m in models}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    paired_bars(ax, models, list(PERFORMANCE.values()),
                {m: np.array([rows[m][k] * 100 for k in PERFORMANCE]) for m in models}, "{:.1f}",
                yerr={m: ci_errors(rows[m]) for m in models}, rotation=90 if len(models) > 1 else 0, fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Score (%), macro average except accuracy")
    ax.set_title(title_of(model, f"performance, {FORMAT}", split), loc="left", pad=30 if len(models) > 1 else 6)
    if len(models) > 1:
        ax.legend(**legend)
    ax.text(0, -0.14, "Error bars: 95% bootstrap CI (1,000 resamples).", transform=ax.transAxes, fontsize=8,
            color=NOTE)
    save(fig, RESULTS / model / "performance", "overall_metrics", split)

    if model != "fp32":  # change vs FP32 with the paired-bootstrap CI and p-value
        row = rows[model]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        y = np.arange(len(PERFORMANCE))[::-1]
        change = np.array([row[f"{k}_change_vs_fp32"] * 100 for k in PERFORMANCE])
        low = np.array([row[f"{k}_change_ci_low"] * 100 for k in PERFORMANCE])
        high = np.array([row[f"{k}_change_ci_high"] * 100 for k in PERFORMANCE])
        ax.axvline(0, color=STYLE["fp32"]["color"], linewidth=1.5, label="FP32 baseline (0)")
        ax.errorbar(change, y, xerr=[change - low, high - change], fmt="o", color=INK, markersize=7,
                    markerfacecolor=STYLE[model]["color"], ecolor=INK, elinewidth=1.2, capsize=4,
                    label=f"{STYLE[model]['label']} - FP32")
        span = max(abs(low).max(), abs(high).max())
        for yi, k, c in zip(y, PERFORMANCE, change):
            ax.text(span * 1.08, yi, f"{c:+.2f} pts  (p = {row[f'{k}_paired_bootstrap_p']:.2f})", va="center",
                    fontsize=8.5, color=INK)
        ax.set_xlim(-span * 1.1, span * 2.1)
        ax.set_yticks(y, list(PERFORMANCE.values()))
        ax.set_xlabel("Change vs FP32 (percentage points)")
        ax.grid(axis="y", visible=False)
        ax.set_title(title_of(model, "change in performance", split), loc="left", pad=24)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
        ax.text(0, -0.2, "Bars: 95% paired bootstrap CI of the difference (same 1,000 resamples for both models); "
                "p: two-sided paired bootstrap.", transform=ax.transAxes, fontsize=8, color=NOTE)
        save(fig, RESULTS / model / "performance", "change_vs_fp32", split)

    # Latency distribution: FP32 box next to the model box, per thread count
    folder = RESULTS / model / "efficiency"
    samples = {m: read(m, "efficiency", "latency_samples", split) for m in models}
    fig, ax = plt.subplots(figsize=(7.5 if len(models) == 1 else 9, 4.4))
    data, owners, labels = [], [], []
    for t, tl in zip(THREADS, thread_labels()):
        for m in models:
            data.append(samples[m].loc[samples[m]["threads"] == t, "latency_ms"])
            owners.append(m)
            labels.append(f"{tl}\n{STYLE[m]['short'].replace(chr(10), ' ')}" if len(models) > 1 else tl)
    box(ax, data, owners, labels)
    ax.set_ylabel("Latency per image, batch 1 (ms)")
    ax.set_title(title_of(model, "latency (number = median, ms)", split), loc="left")
    ax.text(0, -0.2 if len(models) > 1 else -0.14, f"Each box: 600 timed runs (3 repeats x 200). Format: {FORMAT}.",
            transform=ax.transAxes, fontsize=8, color=NOTE)
    save(fig, folder, "latency_distribution", split)

    # Efficiency summary: size, FLOPs, throughput, RAM, FP32 next to the model
    rows = {m: read(m, "efficiency", "metrics", split).iloc[0] for m in models}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), gridspec_kw={"width_ratios": [0.7, 0.7, 1.3, 1.3]})
    for ax, (title, column, fmt) in zip(axes[:2], [("Size (MB)", "size_mb", "{:.1f}"), ("FLOPs (G)", "gflops", "{:.2f}")]):
        paired_bars(ax, models, [""], {m: [rows[m][column]] for m in models}, fmt)
        ax.set_xticks([])
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    for ax, (title, prefix, fmt) in zip(axes[2:], [("Throughput, batch 32 (images/s)", "throughput_img_s", "{:.1f}"),
                                                   ("Peak RAM (MB)", "peak_memory_mb", "{:.0f}")]):
        paired_bars(ax, models, thread_labels(), {m: [rows[m][f"{prefix}_{t}t"] for t in THREADS] for m in models},
                    fmt, fontsize=8)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle(title_of(model, f"efficiency, {FORMAT}", split), x=0.01, ha="left", fontweight="semibold")
    if len(models) > 1:
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.tight_layout()
    save(fig, folder, "efficiency_summary", split)

    # Energy and CO2e: FP32 next to the model
    rows = {m: read(m, "environment", "metrics", split).iloc[0] for m in models}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
    for ax, (title, column) in zip(axes, [("Energy per 1,000 images (Wh)", "energy_wh_per_1000"),
                                          ("CO2e per 1,000 images (g)", "co2e_g_per_1000")]):
        paired_bars(ax, models, [""], {m: [rows[m][column]] for m in models}, "{:.3f}")
        ax.set_xticks([])
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle(title_of(model, "energy and CO2e (estimate, 1 thread)", split), x=0.01, ha="left",
                 fontweight="semibold")
    row = rows[model]
    if len(models) > 1:
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.text(0.01, -0.14 if len(models) > 1 else -0.06,
             f"{ESTIMATE_NOTE}\nAt least 60 s per run. Grid {row['grid_g_per_kwh']:.0f} gCO2e/kWh ({row['country']}).",
             fontsize=8, color=NOTE)
    fig.tight_layout()
    save(fig, RESULTS / model / "environment", "energy_co2e", split)

    if model != "fp32":
        vs_fp32_tables(model, split)


def vs_fp32_tables(model, split):
    """One CSV per metric type: FP32 value, model value, and the change (ratio / % / paired-bootstrap CI)."""
    perf = {m: read(m, "performance", "metrics", split).iloc[0] for m in ("fp32", model)}
    rows = [{"metric": k, "fp32": perf["fp32"][k], model: perf[model][k],
             "change_pts": perf[model][f"{k}_change_vs_fp32"] * 100,
             "change_ci_low_pts": perf[model][f"{k}_change_ci_low"] * 100,
             "change_ci_high_pts": perf[model][f"{k}_change_ci_high"] * 100,
             "paired_bootstrap_p": perf[model][f"{k}_paired_bootstrap_p"]} for k in PERFORMANCE]
    pd.DataFrame(rows).to_csv(RESULTS / model / "performance" / f"vs_fp32_{split}.csv", index=False)

    for kind, columns, lower_is_better in [
            ("efficiency", ["size_mb", "gflops"] + [f"{p}_{t}t" for p in ("latency_median_ms", "latency_p95_ms",
                                                                          "throughput_img_s", "peak_memory_mb")
                                                    for t in THREADS], None),
            ("environment", ["energy_wh_per_1000", "co2e_g_per_1000"], None)]:
        eff = {m: read(m, kind, "metrics", split).iloc[0] for m in ("fp32", model)}
        rows = []
        for c in columns:
            base, value = float(eff["fp32"][c]), float(eff[model][c])
            rows.append({"metric": c, "fp32": base, model: value, "change_pct": (value / base - 1) * 100,
                         "ratio_fp32_over_model": base / value,
                         "better_direction": "higher" if c.startswith("throughput") else "lower"})
        pd.DataFrame(rows).to_csv(RESULTS / model / kind / f"vs_fp32_{split}.csv", index=False)


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
    fp32_reference(ax, x, [perf.loc["fp32", m] * 100 for m in PERFORMANCE])
    ax.set_xticks(x, PERFORMANCE.values())
    ax.set_ylim(0, 118)
    ax.set_ylabel("Score (%), macro average except accuracy")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Performance, all models ({FORMAT}, {split})", loc="left", pad=46)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    ax.text(0, -0.12, "Error bars: 95% bootstrap CI (1,000 resamples). Dashed line: FP32 level. Static INT8 PTQ and "
            "INT8 QAT are the saved seed-42 models.",
            transform=ax.transAxes, fontsize=8, color=NOTE)
    save(fig, out / "performance", "metrics_all_models", split)

    for name, column, ylabel, title, fmt, note in [
            ("size", "size_mb", "File size (MB)", "Model file size", "{:.1f}", None),
            ("flops", "gflops", "FLOPs per image (G)", "FLOPs per image", "{:.2f}",
             "FLOPs = 2 x MACs (ptflops). INT8 models counted on the FP32 architecture; their operations are INT8.")]:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        model_bars(ax, eff[column], fmt)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        if note:
            ax.text(0, -0.24, note, transform=ax.transAxes, fontsize=8, color=NOTE)
        save(fig, out / "efficiency", name, split)

    samples = pd.concat([read(m, "efficiency", "latency_samples", split) for m in MODELS])
    fig, axes = plt.subplots(1, len(THREADS), figsize=(14, 4.6))
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
        fig, ax = plt.subplots(figsize=(9, 4.4))
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
