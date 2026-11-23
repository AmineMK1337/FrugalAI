"""Three comparisons for structured (filter) pruning (validation set), the same evaluation as
pruning/unstructured/comparisons.py. No training: only the pruning step is re-run (deterministic) and models are
evaluated on the CPU.

  1. FP32 baseline vs structured 10/20/30/40% BEFORE fine-tuning   (immediate one-shot pruning damage)
  2. FP32 baseline vs structured 10/20/30/40% AFTER fine-tuning    (final compressed model vs original baseline)
  3. FP32 control vs structured 10% AFTER fine-tuning, by seed     (pruning's own effect, extra fine-tuning removed)

Ratio = share of output filters removed in every prunable layer (not a share of all weights).

Sources:
  before fine-tuning: the models are not saved, so they are rebuilt here from the FP32 checkpoint with prune.py's own
      prune() (deterministic), then evaluated like evaluate.py (CPU, fused + TorchScript, 1,000 bootstrap resamples
      paired with FP32). Checks: parameter count = the evaluated pruned model; accuracy within 2 images of prune.py's
      GPU validation before fine-tuning. Cached in results/comparaison/stage1_cpu_validation.csv.
  after fine-tuning, controls, efficiency, environment: pruning/structured/filter/results/evaluation/comparison/ (evaluate.py session).
Comparisons 1 and 2 use seed 42 (before fine-tuning is identical for every seed).

Writes pruning/structured/filter/results/comparaison/: comparisons.md, table{1,2,3}_*.csv/.pdf/.png, figures (PDF + PNG).
Usage: python pruning/structured/filter/comparisons.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("evaluation_evaluate", ROOT / "evaluation" / "evaluate.py")          # CPU evaluation protocol
sp = _load("structured_prune", HERE / "prune.py")                               # prune() and load_fp32()
uc = _load("unstructured_comparisons", ROOT / "pruning/unstructured/comparisons.py")  # table helpers and style

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch_pruning as tp  # noqa: E402

EVAL = ROOT / "pruning/structured/filter/results/evaluation/comparison"
OUT = HERE / "results/comparaison"
RATIOS = [10, 20, 30, 40]
BEST, SEEDS = 10, [42, 1, 2]
MODELS = ["fp32"] + [f"pruned{r}_seed42" for r in RATIOS]
NAMES = ["FP32\nbaseline"] + [f"{r}%\npruned" for r in RATIOS]
COLORS = [uc.GRAY] + [uc.BLUE] * len(RATIOS)
METRICS, NA, INK, MUTED, GRID = uc.METRICS, uc.NA, uc.INK, uc.MUTED, uc.GRID
PERF = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
        ("specificity_macro", "Specificity (macro)"), ("f1_macro", "F1-score (macro)"), ("auc_macro", "AUC (macro)")]
CELL = "Cells: value (absolute difference in percentage points; relative change in %) vs the reference model."
RATIO_NOTE = "Ratio = share of filters removed per layer."


# ── Comparison 1 data: rebuild and evaluate the pruned models before fine-tuning ──────────────────────────────────

def stage1(params_after):
    cache = OUT / "stage1_cpu_validation.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index("run")
    _, loader, _ = ev.build_dataloaders(ev.load_config(ev.CONFIG))
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    probs = {"fp32": ev.logits(ev.fused_torchscript(ev.fp32_state()), images).softmax(1).numpy()}
    example = torch.zeros(1, 3, 224, 224)
    names = [f"pruned{r}_seed42" for r in RATIOS]
    for r, name in zip(RATIOS, names):
        model = sp.load_fp32()
        sp.prune(model, r / 100, example)
        model.eval()
        _, n_params = tp.utils.count_ops_and_params(model, example)
        if n_params != params_after[name]:
            sys.exit(f"CHECK FAILED: {name} rebuilt with {n_params:,} parameters, evaluated model has "
                     f"{params_after[name]:,}")
        unfused = ev.logits(model, images)
        fused = ev.logits(ev.fused_module_torchscript(model), images)
        if not torch.equal(unfused.argmax(1), fused.argmax(1)):
            sys.exit(f"CHECK FAILED: {name} before fine-tuning: fused and unfused predictions differ")
        probs[name] = fused.softmax(1).numpy()
        saved = pd.read_csv(HERE / "results" / name / "metrics.csv").set_index("stage").loc["before_finetuning"]
        acc = (probs[name].argmax(1) == y).mean()
        if abs(acc - saved["accuracy"]) * len(y) > 2:
            sys.exit(f"CHECK FAILED: {name} before fine-tuning: accuracy {acc:.4f} vs prune.py {saved['accuracy']:.4f}")
        print(f"{name} before fine-tuning: {n_params:,} params, accuracy {acc:.4f} (prune.py {saved['accuracy']:.4f})")
    boot = ev.bootstrap(probs, y, ev.NUM_CLASSES)
    fp32_point, _ = ev.scores(y, probs["fp32"], ev.NUM_CLASSES)
    fp32_pred = probs["fp32"].argmax(1)
    rows = []
    for name in names:
        point, _ = ev.scores(y, probs[name], ev.NUM_CLASSES)
        row = {"run": name, "identical_to_fp32_pct": (probs[name].argmax(1) == fp32_pred).mean() * 100}
        for m in ev.PERFORMANCE:
            d = boot[name][m] - boot["fp32"][m]
            row.update({m: point[m], f"{m}_ci_low": ev.ci(boot[name][m])[0], f"{m}_ci_high": ev.ci(boot[name][m])[1],
                        f"{m}_change_vs_fp32": point[m] - fp32_point[m], f"{m}_change_ci_low": ev.ci(d)[0],
                        f"{m}_change_ci_high": ev.ci(d)[1],
                        f"{m}_paired_bootstrap_p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(cache, index=False)
    return table.set_index("run")


# ── Tables ───────────────────────────────────────────────────────────────────────────────────────────────────────

def vs_fp32_table(df, fp32):
    runs = MODELS[1:]
    wide = {"Metric (%)": [l for _, l in METRICS] + ["Δ macro F1 vs FP32: 95% CI and p"],
            "FP32 baseline": [f"{fp32[m] * 100:.2f}" for m, _ in METRICS] + ["–"]}
    long = []
    for r, run in zip(RATIOS, runs):
        wide[f"{r}% pruned"] = [uc.fmt(df.loc[run, m], fp32[m]) for m, _ in METRICS] + [uc.ci_text(df.loc[run])]
        for m, l in METRICS:
            d, rel = uc.diff(df.loc[run, m], fp32[m])
            f1 = m == "f1_macro"
            long.append({"ratio_pct": r, "metric": l, "fp32": fp32[m] * 100, "pruned": df.loc[run, m] * 100,
                         "abs_diff_pp": d, "rel_change_pct": rel,
                         "ci_low_pp": df.loc[run, "f1_macro_change_ci_low"] * 100 if f1 else np.nan,
                         "ci_high_pp": df.loc[run, "f1_macro_change_ci_high"] * 100 if f1 else np.nan,
                         "p": df.loc[run, "f1_macro_paired_bootstrap_p"] if f1 else np.nan})
    return pd.DataFrame(wide), pd.DataFrame(long)


def vs_control_table(after):
    wide = {"Metric (%)": [l for _, l in METRICS] + ["Δ macro F1 vs control: 95% CI and p"]}
    long = []
    for s in SEEDS:
        c, p = after.loc[f"control_seed{s}"], after.loc[f"pruned{BEST}_seed{s}"]
        wide[f"Control, seed {s}"] = [f"{c[m] * 100:.2f}" for m, _ in METRICS] + ["–"]
        wide[f"{BEST}% pruned, seed {s}"] = [uc.fmt(p[m], c[m]) for m, _ in METRICS] + [NA]
        for m, l in METRICS:
            d, rel = uc.diff(p[m], c[m])
            long.append({"seed": s, "metric": l, "control": c[m] * 100, "pruned": p[m] * 100, "abs_diff_pp": d,
                         "rel_change_pct": rel, "ci": NA, "p": NA})
    cm = {m: np.mean([after.loc[f"control_seed{s}", m] for s in SEEDS]) for m, _ in METRICS}
    pm = {m: np.mean([after.loc[f"pruned{BEST}_seed{s}", m] for s in SEEDS]) for m, _ in METRICS}
    wide["Control, mean"] = [f"{cm[m] * 100:.2f}" for m, _ in METRICS] + ["–"]
    wide[f"{BEST}% pruned, mean"] = [uc.fmt(pm[m], cm[m]) for m, _ in METRICS] + [NA]
    for m, l in METRICS:
        d, rel = uc.diff(pm[m], cm[m])
        long.append({"seed": "mean", "metric": l, "control": cm[m] * 100, "pruned": pm[m] * 100, "abs_diff_pp": d,
                     "rel_change_pct": rel, "ci": NA, "p": NA})
    return pd.DataFrame(wide), pd.DataFrame(long)


# ── Figures ──────────────────────────────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def bar_panel(ax, values, title, unit, decimals=1, ylim=None):
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


def performance(df, fp32, title, name, subtitle):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    for ax, (m, t) in zip(axes.flat, PERF):
        bar_panel(ax, [fp32[m] * 100] + [df.loc[r, m] * 100 for r in MODELS[1:]], t, "%", ylim=(0, 110))
        ax.set_yticks(range(0, 101, 20))
    fig.suptitle(f"{title}\n{subtitle}", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, name)


def performance_vs_control(after):
    groups = ["Seed 42", "Seed 1", "Seed 2", "Mean"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    x, w = np.arange(len(groups)), 0.38
    for ax, (m, title) in zip(axes.flat, PERF):
        ctrl = [after.loc[f"control_seed{s}", m] * 100 for s in SEEDS]
        prun = [after.loc[f"pruned{BEST}_seed{s}", m] * 100 for s in SEEDS]
        ctrl.append(np.mean(ctrl))
        prun.append(np.mean(prun))
        for off, vals, color, label in ((-w / 2, ctrl, uc.GRAY, "FP32 control (same fine-tuning, not pruned)"),
                                        (w / 2, prun, uc.BLUE, f"{BEST}% pruned (pruned + fine-tuned)")):
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
    f1 = [(after.loc[f"pruned{BEST}_seed{s}", "f1_macro"] - after.loc[f"control_seed{s}", "f1_macro"]) * 100
          for s in SEEDS]
    fig.suptitle(f"Comparison 3. Performance: FP32 control vs {BEST}% pruned, both after the same 10-epoch "
                 f"fine-tuning\nIsolates the effect of pruning: F1 {np.mean(f1):+.2f} points vs the control on average "
                 f"({', '.join(f'{v:+.2f}' for v in f1)} for seeds 42, 1, 2: the sign changes with the seed)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "fig3_performance_control_vs_pruned10")


def efficiency(eff):
    panels = [("size_mb", "Model size (file)", "MB", 1), ("gflops", "FLOPs per image", "GFLOPs", 2),
              ("latency_median_ms_1t", "Latency (1 image, 1 thread)", "ms", 1),
              ("throughput_img_s_1t", "Throughput (1 thread)", "images / s", 1),
              ("peak_memory_mb_1t", "Peak memory (RAM)", "MB", 0)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), constrained_layout=True)
    for ax, (col, title, unit, d) in zip(axes.flat, panels):
        bar_panel(ax, [eff.loc[r, col] for r in MODELS], title, unit, decimals=d)
    fig.suptitle("Efficiency: structured pruning after fine-tuning\nFilters are physically removed: size, FLOPs, "
                 "latency and memory all fall with the ratio (latency less than FLOPs)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    save(fig, "fig_efficiency")


def environment(env):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    bar_panel(axes[0], [env.loc[r, "energy_wh_per_1000"] for r in MODELS], "Energy per 1,000 images", "Wh",
              decimals=2)
    bar_panel(axes[1], [env.loc[r, "co2e_g_per_1000"] for r in MODELS], "CO₂e per 1,000 images", "g CO₂e",
              decimals=2)
    e = [(env.loc[r, "energy_wh_per_1000"] / env.loc["fp32", "energy_wh_per_1000"] - 1) * 100 for r in MODELS[1:]]
    fig.suptitle("Environment: structured pruning after fine-tuning (CodeCarbon estimate, 1 thread)\n"
                 f"Energy {', '.join(f'{v:+.0f}%' for v in e)} at 10/20/30/40% (noise ±5%)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    save(fig, "fig_environment")


def markdown(t1, t2, t3):
    def md(df):
        head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
        return head + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.values)
    return f"""# Structured pruning: three comparisons (validation set, 1,532 images)

Generated by `pruning/structured/filter/comparisons.py`; no model was retrained. {RATIO_NOTE}
{CELL} 95% CI and p-value: paired bootstrap of the macro F1 difference vs FP32 (1,000 resamples).

## Table 1: FP32 baseline vs structured pruning, before fine-tuning (immediate one-shot pruning damage)

The pruned models before fine-tuning were not saved by prune.py; they are rebuilt here with the same deterministic
pruning (parameter count and prune.py's logged accuracy checked) and evaluated on the CPU. Same for every seed.

{md(t1)}
## Table 2: FP32 baseline vs structured pruning, after fine-tuning (final compressed model vs original baseline)

Seed 42, epoch with the best validation macro F1.

{md(t2)}
## Table 3: FP32 control vs structured {BEST}%, after fine-tuning, matched by seed (pruning's own effect)

The control is the unpruned FP32 model fine-tuned with the same 10-epoch recipe and seed. The paired bootstrap was only
run against FP32, so no CI or p-value exists for pruned − control.

{md(t3)}
## Figures

- `fig1_performance_fp32_vs_pruned_before_finetuning`: comparison 1, accuracy, precision, recall, specificity, F1, AUC
  (macro), FP32 vs 10/20/30/40% before fine-tuning.
- `fig2_performance_fp32_vs_pruned_after_finetuning`: comparison 2, the same metrics after fine-tuning (seed 42).
- `fig3_performance_control_vs_pruned{BEST}`: comparison 3, FP32 control vs {BEST}% pruned by seed (42, 1, 2) and mean.
- `fig_efficiency`: model size (file), FLOPs, latency, throughput, peak memory (after fine-tuning; the models before
  fine-tuning have the same architecture, so the same efficiency).
- `fig_environment`: energy and CO2e per 1,000 images (CodeCarbon estimate).
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    after = pd.read_csv(EVAL / "performance/table_validation.csv").set_index("model")
    eff = pd.read_csv(EVAL / "efficiency/table_validation.csv").set_index("model")
    env = pd.read_csv(EVAL / "environment/table_validation.csv").set_index("model")
    fp32 = after.loc["fp32"]
    before = stage1(eff["params"].to_dict())
    t1, l1 = vs_fp32_table(before, fp32)
    t2, l2 = vs_fp32_table(after, fp32)
    t3, l3 = vs_control_table(after)
    for t, l, name, title, note in (
            (t1, l1, "table1_fp32_vs_pruned_before_finetuning",
             "Table 1. FP32 baseline vs structured pruning BEFORE fine-tuning (one-shot pruning damage)",
             f"{CELL} {RATIO_NOTE} Same for every seed. 95% CI / p: paired bootstrap vs FP32."),
            (t2, l2, "table2_fp32_vs_pruned_after_finetuning",
             "Table 2. FP32 baseline vs structured pruning AFTER fine-tuning (final model vs original baseline)",
             f"{CELL} {RATIO_NOTE} Seed 42. 95% CI / p: paired bootstrap vs FP32."),
            (t3, l3, f"table3_control_vs_pruned{BEST}_after_finetuning",
             f"Table 3. FP32 control vs structured {BEST}% AFTER fine-tuning, by seed (pruning's own effect)",
             f"{CELL} Control = unpruned FP32 with the same 10-epoch fine-tuning and seed. "
             "No paired bootstrap vs the control exists: CI / p not available.")):
        t.to_csv(OUT / f"{name}.csv", index=False)
        l.to_csv(OUT / f"{name}_long.csv", index=False)
        uc.table_pdf(t, title, note, OUT / f"{name}.pdf", col_width=0.8 if t is t3 else 1.0)
    (OUT / "comparisons.md").write_text(markdown(t1, t2, t3))
    performance(before, fp32, "Comparison 1. Performance: FP32 baseline vs structured pruning BEFORE fine-tuning",
                "fig1_performance_fp32_vs_pruned_before_finetuning", "One-shot pruning damage: even 10% of filters breaks the model (F1 71.0 \u2192 38.0); "
                "from 30% it predicts \"nv\" for almost every image")
    performance(after, fp32, "Comparison 2. Performance: FP32 baseline vs structured pruning AFTER fine-tuning",
                "fig2_performance_fp32_vs_pruned_after_finetuning", "Fine-tuning recovers most of the loss: 10% is close to FP32 (F1 \u22120.7, not significant); "
                "the loss then grows with the ratio (\u221210.4 at 40%, seed 42)")
    performance_vs_control(after)
    efficiency(eff)
    environment(env)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
