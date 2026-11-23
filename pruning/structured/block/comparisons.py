"""Three comparisons for block (depth) pruning (validation set), the same evaluation as pruning/structured/filter and
pruning/unstructured comparisons.py. No training.

  1. FP32 baseline vs 1/2/3/4 blocks removed BEFORE fine-tuning   (immediate one-shot pruning damage)
  2. FP32 baseline vs 1/2/3/4 blocks removed AFTER fine-tuning    (final compressed model vs original baseline)
  3. FP32 control vs the best configuration AFTER fine-tuning, by seed (pruning's own effect)

Sources:
  before fine-tuning: pruning/structured/block/models/blocks<k>_seed42_before.pt, evaluated here like evaluate.py (CPU, fused +
      TorchScript, 1,000 bootstrap resamples paired with FP32). Cached in results/comparisons/stage1_cpu_validation.csv.
  after fine-tuning, efficiency, environment: pruning/structured/block/results/evaluation/comparison/ (evaluate.py session).
  FP32 controls: pruning/structured/filter/results/evaluation/comparison/performance/ (structured session; accuracy metrics are
      deterministic, FP32 is checked identical in both sessions).
Comparisons 1 and 2 use seed 42 (before fine-tuning is identical for every seed).

Writes pruning/structured/block/results/comparisons/: comparisons.md, table{1,2,3}_*.csv/.pdf/.png, figures (PDF + PNG).
Usage: python pruning/structured/block/comparisons.py [--best K]
"""
import argparse
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


ev = _load("evaluation_evaluate", ROOT / "evaluation" / "evaluate.py")
sc = _load("structured_comparisons", ROOT / "pruning/structured/filter/comparisons.py")  # tables, bars, style
uc = sc.uc

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

EVAL = HERE / "results/evaluation/comparison"
CONTROL = ROOT / "pruning/structured/filter/results/evaluation/comparison/performance/table_validation.csv"
OUT = HERE / "results/comparisons"
KS, SEEDS = [1, 2, 3, 4], [42, 1, 2]
METRICS, NA, INK, MUTED, GRID = uc.METRICS, uc.NA, uc.INK, uc.MUTED, uc.GRID
CELL = sc.CELL


def configure(best):
    """Point the structured comparison helpers (tables, bars) at the block-pruning runs."""
    sc.OUT = OUT
    sc.RATIOS = KS
    sc.BEST = best
    sc.MODELS = ["fp32"] + [f"blocks{k}_seed42" for k in KS]
    sc.NAMES = ["FP32\nbaseline"] + [f"{k} block{'s' if k > 1 else ''}\nremoved" for k in KS]
    sc.COLORS = [uc.GRAY] + [uc.BLUE] * len(KS)


def stage1():
    cache = OUT / "stage1_cpu_validation.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index("run")
    _, loader, _ = ev.build_dataloaders(ev.load_config(ev.CONFIG))
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    probs = {"fp32": ev.logits(ev.fused_torchscript(ev.fp32_state()), images).softmax(1).numpy()}
    names = [f"blocks{k}_seed42" for k in KS]
    for name in names:
        module = torch.load(HERE / "models" / f"{name}_before.pt", map_location="cpu", weights_only=False).eval()
        unfused = ev.logits(module, images)
        fused = ev.logits(ev.fused_module_torchscript(module), images)
        if not torch.equal(unfused.argmax(1), fused.argmax(1)):
            sys.exit(f"CHECK FAILED: {name} before fine-tuning: fused and unfused predictions differ")
        probs[name] = fused.softmax(1).numpy()
        saved = pd.read_csv(HERE / "results" / name / "metrics.csv").set_index("stage").loc["before_finetuning"]
        if abs((probs[name].argmax(1) == y).mean() - saved["accuracy"]) * len(y) > 2:
            sys.exit(f"CHECK FAILED: {name} before fine-tuning differs from prune_blocks.py by > 2 images")
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


def rename(wide):
    """Column headers of the structured tables say "<k>% pruned": rewrite them for blocks."""
    cols = {}
    for c in wide.columns:
        for k in KS:
            if c.startswith(f"{k}% pruned"):
                cols[c] = c.replace(f"{k}% pruned", blocks_label(k))
                break
    return wide.rename(columns=cols)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    perf = pd.read_csv(EVAL / "performance/table_validation.csv").set_index("model")
    eff = pd.read_csv(EVAL / "efficiency/table_validation.csv").set_index("model")
    env = pd.read_csv(EVAL / "environment/table_validation.csv").set_index("model")
    control = pd.read_csv(CONTROL).set_index("model")
    for m, _ in METRICS:
        if abs(perf.loc["fp32", m] - control.loc["fp32", m]) > 1e-12:
            sys.exit(f"FP32 {m} differs between the two evaluation sessions")
    best = args.best or max(KS, key=lambda k: perf.loc[f"blocks{k}_seed42", "f1_macro"])
    if not all(f"blocks{best}_seed{s}" in perf.index for s in SEEDS):
        sys.exit(f"blocks{best}: seeds {SEEDS} not all evaluated")
    configure(best)
    after = pd.concat([perf, control.loc[[f"control_seed{s}" for s in SEEDS]]])
    aliases = perf.loc[[f"blocks{best}_seed{s}" for s in SEEDS]].copy()
    aliases.index = [f"pruned{best}_seed{s}" for s in SEEDS]  # names the structured table helper expects
    after = pd.concat([after, aliases])
    fp32 = perf.loc["fp32"]
    before = stage1()
    removed = json.loads((HERE / "results/settings.json").read_text())["removal_order"]
    t1, l1 = sc.vs_fp32_table(before, fp32)
    t2, l2 = sc.vs_fp32_table(after, fp32)
    t3, l3 = sc.vs_control_table(after)
    t1, t2, t3 = rename(t1), rename(t2), rename(t3)
    order_note = "Removal order (least important first): " + ", ".join(removed) + "."
    for t, l, name, title, note in (
            (t1, l1, "table1_fp32_vs_blocks_before_finetuning",
             "Table 1. FP32 baseline vs block pruning BEFORE fine-tuning (one-shot pruning damage)",
             f"{CELL} {order_note} Same for every seed. 95% CI / p: paired bootstrap vs FP32."),
            (t2, l2, "table2_fp32_vs_blocks_after_finetuning",
             "Table 2. FP32 baseline vs block pruning AFTER fine-tuning (final model vs original baseline)",
             f"{CELL} {order_note} Seed 42. 95% CI / p: paired bootstrap vs FP32."),
            (t3, l3, f"table3_control_vs_blocks{best}_after_finetuning",
             f"Table 3. FP32 control vs {best} block(s) removed AFTER fine-tuning, by seed (pruning's own effect)",
             f"{CELL} Control = unpruned FP32 with the same 10-epoch fine-tuning and seed. "
             "No paired bootstrap vs the control exists: CI / p not available.")):
        t.to_csv(OUT / f"{name}.csv", index=False)
        l.to_csv(OUT / f"{name}_long.csv", index=False)
        uc.table_pdf(t, title, note, OUT / f"{name}.pdf", col_width=0.8 if t is t3 else 1.0)

    f1b = [before.loc[f"blocks{k}_seed42", "f1_macro"] * 100 for k in KS]
    f1a = [after.loc[f"blocks{k}_seed42", "f1_macro"] * 100 for k in KS]
    sc.performance(before, fp32, "Comparison 1. Performance: FP32 baseline vs block pruning BEFORE fine-tuning",
                   "fig1_performance_fp32_vs_blocks_before_finetuning",
                   f"One-shot damage: macro F1 {fp32['f1_macro'] * 100:.1f} → "
                   + ", ".join(f"{v:.1f}" for v in f1b) + " with 1, 2, 3, 4 blocks removed")
    sc.performance(after, fp32, "Comparison 2. Performance: FP32 baseline vs block pruning AFTER fine-tuning",
                   "fig2_performance_fp32_vs_blocks_after_finetuning",
                   f"After 10 epochs: macro F1 " + ", ".join(f"{v:.1f}" for v in f1a)
                   + f" with 1, 2, 3, 4 blocks removed (FP32 {fp32['f1_macro'] * 100:.1f}; seed 42)")
    performance_vs_control(after, best)
    efficiency(eff)
    environment(env)
    (OUT / "comparisons.md").write_text(markdown(t1, t2, t3, best, order_note))
    print(f"written to {OUT} (best = {best} block(s))")


def blocks_label(k):
    return f"{k} block{'s' if k > 1 else ''} removed"


def performance_vs_control(after, best):
    groups = ["Seed 42", "Seed 1", "Seed 2", "Mean"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    x, w = np.arange(len(groups)), 0.38
    for ax, (m, title) in zip(axes.flat, sc.PERF):
        ctrl = [after.loc[f"control_seed{s}", m] * 100 for s in SEEDS]
        prun = [after.loc[f"blocks{best}_seed{s}", m] * 100 for s in SEEDS]
        ctrl.append(np.mean(ctrl))
        prun.append(np.mean(prun))
        for off, vals, color, label in ((-w / 2, ctrl, uc.GRAY, "FP32 control (same fine-tuning, not pruned)"),
                                        (w / 2, prun, uc.BLUE, f"{blocks_label(best)} (pruned + fine-tuned)")):
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
    f1 = [(after.loc[f"blocks{best}_seed{s}", "f1_macro"] - after.loc[f"control_seed{s}", "f1_macro"]) * 100
          for s in SEEDS]
    same_sign = all(v > 0 for v in f1) or all(v < 0 for v in f1)
    fig.suptitle(f"Comparison 3. Performance: FP32 control vs {blocks_label(best)}, both after the same 10-epoch "
                 f"fine-tuning\nIsolates the effect of pruning: F1 {np.mean(f1):+.2f} points vs the control on average "
                 f"({', '.join(f'{v:+.2f}' for v in f1)} for seeds 42, 1, 2"
                 + ("" if same_sign else ": the sign changes with the seed") + ")",
                 x=0.01, ha="left", fontsize=12, color=INK)
    sc.save(fig, f"fig3_performance_control_vs_blocks{best}")


def markdown(t1, t2, t3, best, order_note):
    def md(df):
        head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
        return head + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.values)
    return f"""# Block pruning: three comparisons (validation set, 1,532 images)

Generated by `pruning/structured/block/comparisons.py`; no model was retrained. {order_note}
{CELL} 95% CI and p-value: paired bootstrap of the macro F1 difference vs FP32 (1,000 resamples).

## Table 1: FP32 baseline vs block pruning, before fine-tuning (immediate one-shot pruning damage)

The models right after block removal are saved by prune_blocks.py and evaluated here on the CPU. Same for every seed.

{md(t1)}
## Table 2: FP32 baseline vs block pruning, after fine-tuning (final compressed model vs original baseline)

Seed 42, epoch with the best validation macro F1.

{md(t2)}
## Table 3: FP32 control vs {blocks_label(best)}, after fine-tuning, matched by seed (pruning's own effect)

The control is the unpruned FP32 model fine-tuned with the same 10-epoch recipe and seed. The paired bootstrap was only
run against FP32, so no CI or p-value exists for pruned − control.

{md(t3)}
## Figures

- `fig1_performance_fp32_vs_blocks_before_finetuning`: comparison 1, accuracy, precision, recall, specificity, F1, AUC
  (macro), FP32 vs 1/2/3/4 blocks removed before fine-tuning.
- `fig2_performance_fp32_vs_blocks_after_finetuning`: comparison 2, the same metrics after fine-tuning (seed 42).
- `fig3_performance_control_vs_blocks{best}`: comparison 3, FP32 control vs {blocks_label(best)} by seed and mean.
- `fig_efficiency`: model size (file), FLOPs, latency, throughput, peak memory (the models before fine-tuning have the
  same architecture, so the same efficiency).
- `fig_environment`: energy and CO2e per 1,000 images (CodeCarbon estimate).
"""


def efficiency(eff):
    panels = [("size_mb", "Model size (file)", "MB", 1), ("gflops", "FLOPs per image", "GFLOPs", 2),
              ("latency_median_ms_1t", "Latency (1 image, 1 thread)", "ms", 1),
              ("throughput_img_s_1t", "Throughput (1 thread)", "images / s", 1),
              ("peak_memory_mb_1t", "Peak memory (RAM)", "MB", 0)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9), constrained_layout=True)
    for ax, (col, title, unit, d) in zip(axes.flat, panels):
        sc.bar_panel(ax, [eff.loc[r, col] for r in sc.MODELS], title, unit, decimals=d)
    lat = [eff.loc["fp32", "latency_median_ms_1t"] / eff.loc[r, "latency_median_ms_1t"] for r in sc.MODELS[1:]]
    fig.suptitle("Efficiency: block pruning after fine-tuning\nWhole blocks are removed: every block costs the same "
                 "FLOPs, but very different parameters; speed-up " + ", ".join(f"{v:.2f}×" for v in lat)
                 + " with 1-4 blocks", x=0.01, ha="left", fontsize=12, color=INK)
    sc.save(fig, "fig_efficiency")


def environment(env):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    sc.bar_panel(axes[0], [env.loc[r, "energy_wh_per_1000"] for r in sc.MODELS], "Energy per 1,000 images", "Wh",
                 decimals=2)
    sc.bar_panel(axes[1], [env.loc[r, "co2e_g_per_1000"] for r in sc.MODELS], "CO₂e per 1,000 images",
                 "g CO₂e", decimals=2)
    e = [(env.loc[r, "energy_wh_per_1000"] / env.loc["fp32", "energy_wh_per_1000"] - 1) * 100 for r in sc.MODELS[1:]]
    fig.suptitle("Environment: block pruning after fine-tuning (CodeCarbon estimate, 1 thread)\n"
                 f"Energy {', '.join(f'{v:+.0f}%' for v in e)} with 1, 2, 3, 4 blocks removed (noise ±5%)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    sc.save(fig, "fig_environment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=int, help="configuration repeated with 3 seeds (default: best seed-42 F1)")
    main(parser.parse_args())
