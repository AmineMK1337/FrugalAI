"""One clear latency figure for INT8 QAT vs FP32 (CPU, batch 1), from the evaluation of the three QAT seeds.

Left panel:  median latency per image at 1 and 4 threads, with the speed-up.  -> "how much faster?"
Right panel: spread of the timed runs around each model's own median.          -> "is it stable?"

Reads quantization/qat/results/evaluation_seeds/<model>/efficiency/latency_samples_validation.csv
(written by evaluation/evaluate.py --models fp32 qat qat_seed1 qat_seed2 --results-dir ...evaluation_seeds).
Writes quantization/qat/results/evaluation_seeds/latency_summary_validation.{png,pdf}.

Usage: python quantization/qat/plot_latency.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "evaluation_seeds"
QAT_RUNS = ["qat", "qat_seed1", "qat_seed2"]  # training seeds 42, 1, 2
THREADS = (1, 4)
# Categorical pair checked with the dataviz validator (CVD and normal-vision separation pass); orange is below 3:1
# contrast on white, so every bar carries a text label.
FP32_COLOR, QAT_COLOR = "#257de0", "#f47230"
INK, MUTED, GRID = "#1f1e1c", "#5f6368", "#e6e6e6"


def samples(model):
    return pd.read_csv(RESULTS / model / "efficiency" / "latency_samples_validation.csv")


def summarize():
    fp32 = samples("fp32")
    qat = {m: samples(m) for m in QAT_RUNS}
    out = {}
    for t in THREADS:
        f = fp32.loc[fp32.threads == t, "latency_ms"].to_numpy()
        q_runs = [d.loc[d.threads == t, "latency_ms"].to_numpy() for d in qat.values()]
        q_medians = [np.median(q) for q in q_runs]
        # spread: every run relative to its own model's median (seeds pooled after normalizing)
        f_rel = (f / np.median(f) - 1) * 100
        q_rel = np.concatenate([(q / np.median(q) - 1) * 100 for q in q_runs])
        out[t] = dict(fp32_median=np.median(f), qat_median=np.mean(q_medians), qat_seed_medians=q_medians,
                      fp32_rel=f_rel, qat_rel=q_rel, n_fp32=len(f), n_qat=sum(len(q) for q in q_runs),
                      fp32_min=f.min(), qat_max=max(q.max() for q in q_runs))
    return out


def main():
    s = summarize()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b0b0b0", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.42})

    # rows, top to bottom: 1T FP32, 1T QAT, (gap), 4T FP32, 4T QAT
    ys = {(1, "fp32"): 3.6, (1, "qat"): 3.0, (4, "fp32"): 1.4, (4, "qat"): 0.8}
    group_y = {1: 3.3, 4: 1.1}
    h = 0.46

    # ── Panel A: how much faster ──
    xmax = s[1]["fp32_median"] * 1.32
    for t in THREADS:
        f, q = s[t]["fp32_median"], s[t]["qat_median"]
        ax1.barh(ys[(t, "fp32")], f, height=h, color=FP32_COLOR, edgecolor="white", linewidth=2)
        ax1.barh(ys[(t, "qat")], q, height=h, color=QAT_COLOR, edgecolor="white", linewidth=2)
        ax1.text(f + xmax * 0.012, ys[(t, "fp32")], f"{f:.1f} ms", va="center", fontsize=10, color=INK)
        ax1.text(q + xmax * 0.012, ys[(t, "qat")], f"{q:.1f} ms", va="center", fontsize=10, color=INK)
        ax1.text(q + xmax * 0.13, ys[(t, "qat")], f"{f / q:.1f}× faster", va="center", fontsize=11,
                 fontweight="bold", color=INK)
    ax1.set_yticks([ys[k] for k in ys], ["FP32", "INT8 QAT"] * 2)
    for t, y in group_y.items():
        ax1.text(-0.2, y, f"{t} thread{'s' if t > 1 else ''}", transform=ax1.get_yaxis_transform(), ha="right",
                 va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(0.3, 4.1)
    ax1.set_xlabel("Median latency per image, batch 1 (ms)  ·  lower is better")
    ax1.grid(axis="x", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("How much faster?", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)

    # ── Panel B: is it stable? ──
    for t in THREADS:
        for key, rel, color in (("fp32", s[t]["fp32_rel"], FP32_COLOR), ("qat", s[t]["qat_rel"], QAT_COLOR)):
            y = ys[(t, key)]
            p5, p25, p75, p95 = np.percentile(rel, [5, 25, 75, 95])
            ax2.plot([p5, p95], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax2.plot([p25, p75], [y, y], color=color, linewidth=7, solid_capstyle="round")
            ax2.plot([0], [y], marker="o", markersize=8, color="white", markeredgecolor=color, markeredgewidth=2)
            ax2.text(6.4, y, f"{p5:+.1f}% to {p95:+.1f}%", va="center", fontsize=9.5, color=INK,
                     fontfamily="DejaVu Sans Mono")
    ax2.axvline(0, color="#b0b0b0", linewidth=1, zorder=0)
    ax2.set_yticks([ys[k] for k in ys], ["FP32", "INT8 QAT"] * 2)
    for t, y in group_y.items():
        ax2.text(-0.2, y, f"{t} thread{'s' if t > 1 else ''}", transform=ax2.get_yaxis_transform(), ha="right",
                 va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax2.set_xlim(-3, 10.5)
    ax2.set_ylim(0.3, 4.1)
    ax2.set_xlabel("Difference from the model's own median latency (%)")
    ax2.grid(axis="x", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Is it stable from run to run?", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    # key for the interval marks
    ax2.plot([-2.6, -0.6], [4.0, 4.0], color=MUTED, linewidth=2)
    ax2.plot([-2.0, -1.2], [4.0, 4.0], color=MUTED, linewidth=7, solid_capstyle="round")
    ax2.plot([-1.6], [4.0], marker="o", markersize=7, color="white", markeredgecolor=MUTED, markeredgewidth=2)
    ax2.text(-0.2, 4.0, "thin line = 90% of runs, thick = middle 50%, ○ = median", fontsize=8.3, color=MUTED,
             va="center")

    one = s[1]
    fig.suptitle(f"INT8 QAT is {one['fp32_median'] / one['qat_median']:.1f}× faster than FP32 per image on CPU, "
                 "with run-to-run variation of only a few percent", x=0.06, y=1.02, ha="left", fontsize=13.5, fontweight="bold",
                 color=INK)
    seeds = ", ".join(f"{m:.2f}" for m in one["qat_seed_medians"])
    fig.text(0.06, -0.08,
             f"Intel Xeon Gold 6348 CPU, batch 1, fused TorchScript, validation images. FP32: {one['n_fp32']} timed "
             f"runs per thread count (3 repeats × 200). INT8 QAT: {one['n_qat']:,} runs (3 training seeds × 3 repeats × "
             f"200); bar = mean of the seed medians\n({seeds} ms at 1 thread). No overlap: the slowest QAT run "
             f"({one['qat_max']:.1f} ms) is still {one['fp32_min'] / one['qat_max']:.0f}× faster than the fastest "
             f"FP32 run ({one['fp32_min']:.1f} ms) at 1 thread. A few rare spikes beyond the 95th percentile "
             "(operating-system scheduling) are not drawn.",
             fontsize=8.3, color=MUTED, ha="left", va="top")

    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"latency_summary_validation.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", RESULTS / "latency_summary_validation.png")


if __name__ == "__main__":
    main()
