"""Three comparisons for unstructured pruning (validation set), from existing results only: no training, no inference.

  1. FP32 baseline vs unstructured 30/50/70/90% BEFORE fine-tuning   (immediate one-shot pruning damage)
  2. FP32 baseline vs unstructured 30/50/70/90% AFTER fine-tuning    (final compressed model vs original baseline)
  3. FP32 control vs unstructured 70% AFTER fine-tuning, by seed     (pruning's own effect, extra fine-tuning removed)

Sources:
  pruning/unstructured/results/report/stage1_cpu_validation.csv                    before fine-tuning (report.py)
  pruning/unstructured/results/evaluation/comparison/performance/table_validation.csv   after fine-tuning (evaluate.py)
  pruning/structured/filter/results/evaluation/comparison/performance/table_validation.csv           FP32 controls (structured session)
Comparisons 1 and 2 use seed 42 (before fine-tuning is identical for every seed). The 95% CI and p-value of the macro F1
difference are paired bootstrap vs FP32 (1,000 resamples); none exist for pruned - control ("not available").

Writes pruning/unstructured/results/comparisons/: comparisons.md, table{1,2,3}_*.csv/.pdf/.png, and one figure\nper metric group of the evaluation (performance before / after fine-tuning, efficiency, environment), PDF + PNG.
Usage: python pruning/unstructured/comparisons.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "pruning/unstructured/results/report/stage1_cpu_validation.csv"
AFTER = ROOT / "pruning/unstructured/results/evaluation/comparison/performance/table_validation.csv"
CONTROL = ROOT / "pruning/structured/filter/results/evaluation/comparison/performance/table_validation.csv"
OUT = ROOT / "pruning/unstructured/results/comparisons"

METRICS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("precision_macro", "Macro precision"),
           ("specificity_macro", "Macro specificity"), ("f1_macro", "Macro F1"), ("auc_macro", "Macro AUC"),
           ("melanoma_recall", "Melanoma recall")]
SPARSITIES = [30, 50, 70, 90]
SEEDS = [42, 1, 2]
NA = "not available"

# Palette: FP32 reference in gray, every pruned model in one series colour (blue)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE, GRAY = "#2a78d6", "#898781"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42})


def load():
    before = pd.read_csv(BEFORE).set_index("run")
    after = pd.read_csv(AFTER).set_index("model")
    control = pd.read_csv(CONTROL).set_index("model")
    for m, _ in METRICS:  # accuracy metrics are deterministic: FP32 must be identical in both sessions
        if abs(after.loc["fp32", m] - control.loc["fp32", m]) > 1e-12:
            raise SystemExit(f"FP32 {m} differs between the two evaluation sessions")
    return before, after, control


def diff(value, ref):
    """Absolute difference (percentage points) and relative change (%)."""
    d = (value - ref) * 100
    return d, d / (ref * 100) * 100


def fmt(value, ref):
    d, rel = diff(value, ref)
    return f"{value * 100:.2f} ({d:+.2f}; {rel:+.1f}%)"


def ci_text(row):
    if "f1_macro_change_ci_low" not in row or pd.isna(row["f1_macro_change_ci_low"]):
        return NA
    p = row["f1_macro_paired_bootstrap_p"]
    return (f"{row['f1_macro_change_ci_low'] * 100:+.2f} to {row['f1_macro_change_ci_high'] * 100:+.2f}, "
            + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}"))


def vs_fp32_table(df, fp32):
    """Rows = metrics, columns = FP32 then each sparsity; long CSV alongside."""
    runs = [f"sparse{s}_seed42" for s in SPARSITIES]
    wide = {"Metric (%)": [l for _, l in METRICS] + ["Δ macro F1 vs FP32: 95% CI and p"],
            "FP32 baseline": [f"{fp32[m] * 100:.2f}" for m, _ in METRICS] + ["–"]}
    long = []
    for s, r in zip(SPARSITIES, runs):
        wide[f"{s}% sparse"] = [fmt(df.loc[r, m], fp32[m]) for m, _ in METRICS] + [ci_text(df.loc[r])]
        for m, l in METRICS:
            d, rel = diff(df.loc[r, m], fp32[m])
            long.append({"sparsity_pct": s, "metric": l, "fp32": fp32[m] * 100, "pruned": df.loc[r, m] * 100,
                         "abs_diff_pp": d, "rel_change_pct": rel,
                         "ci_low_pp": df.loc[r, "f1_macro_change_ci_low"] * 100 if m == "f1_macro" else np.nan,
                         "ci_high_pp": df.loc[r, "f1_macro_change_ci_high"] * 100 if m == "f1_macro" else np.nan,
                         "p": df.loc[r, "f1_macro_paired_bootstrap_p"] if m == "f1_macro" else np.nan})
    return pd.DataFrame(wide), pd.DataFrame(long)


def vs_control_table(after, control):
    wide = {"Metric (%)": [l for _, l in METRICS] + ["Δ macro F1 vs control: 95% CI and p"]}
    long = []
    for s in SEEDS:
        c, p = control.loc[f"control_seed{s}"], after.loc[f"sparse70_seed{s}"]
        wide[f"Control, seed {s}"] = [f"{c[m] * 100:.2f}" for m, _ in METRICS] + ["–"]
        wide[f"70% sparse, seed {s}"] = [fmt(p[m], c[m]) for m, _ in METRICS] + [NA]
        for m, l in METRICS:
            d, rel = diff(p[m], c[m])
            long.append({"seed": s, "metric": l, "control": c[m] * 100, "pruned": p[m] * 100, "abs_diff_pp": d,
                         "rel_change_pct": rel, "ci": NA, "p": NA})
    cm = {m: np.mean([control.loc[f"control_seed{s}", m] for s in SEEDS]) for m, _ in METRICS}
    pm = {m: np.mean([after.loc[f"sparse70_seed{s}", m] for s in SEEDS]) for m, _ in METRICS}
    wide["Control, mean"] = [f"{cm[m] * 100:.2f}" for m, _ in METRICS] + ["–"]
    wide["70% sparse, mean"] = [fmt(pm[m], cm[m]) for m, _ in METRICS] + [NA]
    for m, l in METRICS:
        d, rel = diff(pm[m], cm[m])
        long.append({"seed": "mean", "metric": l, "control": cm[m] * 100, "pruned": pm[m] * 100, "abs_diff_pp": d,
                     "rel_change_pct": rel, "ci": NA, "p": NA})
    return pd.DataFrame(wide), pd.DataFrame(long)


def table_pdf(wide, title, note, path, col_width=1.0):
    n_cols = len(wide.columns)
    fig_w = 2.0 + col_width * 1.55 * (n_cols - 1)
    fig, ax = plt.subplots(figsize=(fig_w, 0.32 * (len(wide) + 1) + 1.0))
    ax.axis("off")
    cells = [[str(v).replace(" (", "\n(").replace(", p", "\np").replace(": 95%", ":\n95%") for v in row]
             for row in wide.values]
    first = 2.0 / fig_w
    widths = [first] + [(1 - first) / (n_cols - 1)] * (n_cols - 1)
    t = ax.table(cellText=cells, colLabels=[c.replace(", ", ",\n") for c in wide.columns], loc="center",
                 cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 2.1)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(weight="bold", color=INK)
        elif c == 0:
            cell.set_text_props(ha="left", color=INK)
            cell._loc = "left"
        if r > 0 and wide.iloc[r - 1, 0] in ("Macro F1", "Melanoma recall"):
            cell.set_text_props(weight="bold")
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    fig.text(0.01, 0.01, note, fontsize=6.5, color=MUTED, wrap=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


EFFICIENCY = ROOT / "pruning/unstructured/results/evaluation/comparison/efficiency/table_validation.csv"
ENVIRONMENT = ROOT / "pruning/unstructured/results/evaluation/comparison/environment/table_validation.csv"
MODELS = ["fp32"] + [f"sparse{s}_seed42" for s in SPARSITIES]
NAMES = ["FP32\nbaseline"] + [f"{s}%\nsparse" for s in SPARSITIES]
COLORS = [GRAY] + [BLUE] * len(SPARSITIES)  # FP32 = gray reference, pruned models = one series colour


def bar_panel(ax, values, title, unit, decimals=1, ylim=None):
    """One metric: FP32 (gray) and the pruned models (blue), value on each bar, dashed FP32 reference."""
    x = np.arange(len(values))
    ax.bar(x, values, width=0.62, color=COLORS, edgecolor="white", linewidth=2)
    ax.axhline(values[0], color=MUTED, lw=0.9, ls="--", zorder=0)
    top = ylim[1] if ylim else max(values) * 1.18
    for xi, v in zip(x, values):
        ax.text(xi, v + top * 0.02, f"{v:.{decimals}f}", ha="center", va="bottom", fontsize=8, color=INK, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
    ax.set_xticks(x, NAMES, fontsize=7.5)
    ax.set_ylim(*(ylim or (0, top)))
    ax.set_title(title, loc="left", fontsize=10, color=INK, weight="bold")
    ax.set_ylabel(unit)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def performance(df, fp32, stage, name, subtitle):
    """Performance metrics of the evaluation diagram: accuracy, precision, recall, specificity, F1, AUC (macro)."""
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
               ("specificity_macro", "Specificity (macro)"), ("f1_macro", "F1-score (macro)"),
               ("auc_macro", "AUC (macro)")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    for ax, (m, title) in zip(axes.flat, metrics):
        bar_panel(ax, [fp32[m] * 100] + [df.loc[r, m] * 100 for r in MODELS[1:]], title, "%", ylim=(0, 110))
        ax.set_yticks(range(0, 101, 20))
    fig.suptitle(f"{stage}\n{subtitle}", x=0.01, ha="left", fontsize=12,
                 color=INK)
    save(fig, name)


def performance_vs_control(after, control):
    """Comparison 3: FP32 control vs 70% sparse after fine-tuning, matched by seed (3 seeds + mean)."""
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
               ("specificity_macro", "Specificity (macro)"), ("f1_macro", "F1-score (macro)"),
               ("auc_macro", "AUC (macro)")]
    groups = ["Seed 42", "Seed 1", "Seed 2", "Mean"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    x, w = np.arange(len(groups)), 0.38
    for ax, (m, title) in zip(axes.flat, metrics):
        ctrl = [control.loc[f"control_seed{s}", m] * 100 for s in SEEDS]
        prun = [after.loc[f"sparse70_seed{s}", m] * 100 for s in SEEDS]
        ctrl.append(np.mean(ctrl))
        prun.append(np.mean(prun))
        for off, vals, color, label in ((-w / 2, ctrl, GRAY, "FP32 control (same fine-tuning, not pruned)"),
                                        (w / 2, prun, BLUE, "70% sparse (pruned + fine-tuned)")):
            ax.bar(x + off, vals, width=w, color=color, edgecolor="white", linewidth=2, label=label)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK, zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.axvline(2.5, color=GRID, lw=1)
        ax.set_xticks(x, groups, fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("%")
        ax.set_title(title, loc="left", fontsize=10, color=INK, weight="bold")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    f1 = [(after.loc[f"sparse70_seed{s}", "f1_macro"] - control.loc[f"control_seed{s}", "f1_macro"]) * 100
          for s in SEEDS]
    fig.suptitle("Comparison 3. Performance: FP32 control vs 70% sparse, both after the same 10-epoch fine-tuning\n"
                 f"Isolates the effect of pruning: F1 {np.mean(f1):+.2f} points vs the control on average "
                 f"({', '.join(f'{v:+.2f}' for v in f1)} for seeds 42, 1, 2)", x=0.01, ha="left", fontsize=12,
                 color=INK)
    save(fig, "fig3_performance_control_vs_pruned70")


def efficiency(eff):
    panels = [("state_dict_gzip_mb", "Model size (compressed, gzip)", "MB", 1),
              ("gflops", "FLOPs per image", "GFLOPs", 2), ("latency_median_ms_1t", "Latency (1 image, 1 thread)", "ms", 1),
              ("throughput_img_s_1t", "Throughput (1 thread)", "images / s", 1),
              ("peak_memory_mb_1t", "Peak memory (RAM)", "MB", 0)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), constrained_layout=True)
    for ax, (col, title, unit, d) in zip(axes.flat, panels):
        bar_panel(ax, [eff.loc[r, col] for r in MODELS], title, unit, decimals=d)
    fig.suptitle("Efficiency: unstructured pruning after fine-tuning\nZeros compress well, so the compressed size "
                 "shrinks; but they are still computed, so FLOPs, latency, throughput and memory stay at the FP32 level",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "fig_efficiency")


def environment(env):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    bar_panel(axes[0], [env.loc[r, "energy_wh_per_1000"] for r in MODELS], "Energy per 1,000 images", "Wh",
              decimals=2)
    bar_panel(axes[1], [env.loc[r, "co2e_g_per_1000"] for r in MODELS], "CO\u2082e per 1,000 images", "g CO\u2082e",
              decimals=2)
    fig.suptitle("Environment: unstructured pruning after fine-tuning (CodeCarbon estimate, 1 thread)\n"
                 "Within the \u00b15% measurement noise: no energy saving", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "fig_environment")


def markdown(t1, t2, t3):
    def md(df):
        head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
        return head + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.values)
    cell = "Cells: value (absolute difference in percentage points; relative change in %) vs the reference model."
    return f"""# Unstructured pruning: three comparisons (validation set, 1,532 images)

Generated by `pruning/unstructured/comparisons.py` from existing results; no model was retrained or re-evaluated.
{cell} 95% CI and p-value: paired bootstrap of the macro F1 difference vs FP32 (1,000 resamples).

## Table 1: FP32 baseline vs unstructured pruning, before fine-tuning (immediate one-shot pruning damage)

Pruning is deterministic: these values are the same for seeds 42, 1 and 2.

{md(t1)}
## Table 2: FP32 baseline vs unstructured pruning, after fine-tuning (final compressed model vs original baseline)

Seed 42, epoch with the best validation macro F1.

{md(t2)}
## Table 3: FP32 control vs unstructured 70%, after fine-tuning, matched by seed (pruning's own effect)

The control is the unpruned FP32 model fine-tuned with the same 10-epoch recipe and seed. The paired bootstrap was only
run against FP32, so no CI or p-value exists for pruned − control.

{md(t3)}
## Figures

- `fig1_performance_fp32_vs_pruned_before_finetuning`: comparison 1, accuracy, precision, recall, specificity, F1, AUC
  (macro), FP32 vs 30/50/70/90% before fine-tuning.
- `fig2_performance_fp32_vs_pruned_after_finetuning`: comparison 2, the same metrics after fine-tuning (seed 42).
- `fig3_performance_control_vs_pruned70`: comparison 3, FP32 control vs 70% sparse by seed (42, 1, 2) and mean.
- `fig_efficiency`: compressed model size (gzip), FLOPs, latency, throughput, peak memory. The uncompressed file is
  44.8 MB for every model (zeros are stored like any weight), so it is not plotted.
- `fig_environment`: energy and CO2e per 1,000 images (CodeCarbon estimate).
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before, after, control = load()
    fp32 = after.loc["fp32"]
    t1, l1 = vs_fp32_table(before, fp32)
    t2, l2 = vs_fp32_table(after, fp32)
    t3, l3 = vs_control_table(after, control)
    cell = "Cells: value (absolute difference in percentage points; relative change in %) vs the reference model."
    for t, l, name, title, note in (
            (t1, l1, "table1_fp32_vs_pruned_before_finetuning",
             "Table 1. FP32 baseline vs unstructured pruning BEFORE fine-tuning (one-shot pruning damage)",
             cell + " Same for every seed. 95% CI / p: paired bootstrap vs FP32."),
            (t2, l2, "table2_fp32_vs_pruned_after_finetuning",
             "Table 2. FP32 baseline vs unstructured pruning AFTER fine-tuning (final model vs original baseline)",
             cell + " Seed 42. 95% CI / p: paired bootstrap vs FP32."),
            (t3, l3, "table3_control_vs_pruned70_after_finetuning",
             "Table 3. FP32 control vs unstructured 70% AFTER fine-tuning, by seed (pruning's own effect)",
             cell + " Control = unpruned FP32 with the same 10-epoch fine-tuning and seed. "
                    "No paired bootstrap vs the control exists: CI / p not available.")):
        t.to_csv(OUT / f"{name}.csv", index=False)
        l.to_csv(OUT / f"{name}_long.csv", index=False)
        table_pdf(t, title, note, OUT / f"{name}.pdf", col_width=0.8 if t is t3 else 1.0)
    (OUT / "comparisons.md").write_text(markdown(t1, t2, t3))
    performance(before, fp32, "Comparison 1. Performance: FP32 baseline vs unstructured pruning BEFORE fine-tuning",
                "fig1_performance_fp32_vs_pruned_before_finetuning",
                "One-shot pruning damage: 30% is almost unchanged; at 50% recall starts to drop (-5.3); at 70-90% the model breaks")
    performance(after, fp32, "Comparison 2. Performance: FP32 baseline vs unstructured pruning AFTER fine-tuning",
                "fig2_performance_fp32_vs_pruned_after_finetuning",
                "10 epochs of fine-tuning bring every sparsity level back to the FP32 level (seed 42)")
    performance_vs_control(after, control)
    efficiency(pd.read_csv(EFFICIENCY).set_index("model"))
    environment(pd.read_csv(ENVIRONMENT).set_index("model"))
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
