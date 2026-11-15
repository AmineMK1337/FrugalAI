"""One figure per pruning ratio: FP32 baseline vs pruned (before fine-tuning) vs pruned + fine-tuned, validation split.

The pruned-before-fine-tuning models were not saved. Pruning is deterministic (L1 filter norms of the FP32 checkpoint),
so they are rebuilt here with prune.py's own function and checked: same architecture as the saved fine-tuned model and
same validation accuracy as prune.py recorded before fine-tuning (up to 2 images, GPU vs CPU). They are then evaluated
exactly like evaluate.py does: CPU, fused + TorchScript, same metrics, same 1,000 bootstrap resamples (seed 0).
FP32 and the fine-tuned models are read from evaluation/results_pruning/ (evaluate.py). Fine-tuning does not change
the architecture, so size, GFLOPs, latency and energy are the same before and after fine-tuning: the efficiency panel
shows the measured values of the fine-tuned model for the pruned architecture.

Writes pruning/structured/results/report/stages/: stages_pruned<ratio>_seed42.{png,pdf}, stages_validation.csv.
Usage: python pruning/structured/plot_stages.py [--ratios 10 20 30 40]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = load("evaluation_evaluate", ROOT / "evaluation" / "evaluate.py")  # sets CPU only, chdir to the project
pr = load("pruning_prune", HERE / "prune.py")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

EVAL = ROOT / "evaluation" / "results_pruning"
OUT = HERE / "results" / "report" / "stages"
SPLIT, SEED = "validation", 42
METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced\naccuracy", "precision_macro": "Precision",
           "specificity_macro": "Specificity", "f1_macro": "Macro F1", "auc_macro": "AUC",
           "melanoma_recall": "Melanoma\nrecall"}
STAGES = {"fp32": dict(label="FP32 baseline", color="#2a78d6", hatch=""),
          "pruned": dict(label="Pruned, before fine-tuning", color="#ffffff", hatch="xx"),
          "finetuned": dict(label="Pruned + fine-tuned", color="#c25420", hatch="//")}
INK, NOTE = "#1f1e1c", "#555555"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.color": "#e5e5e5",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.8, "axes.axisbelow": True})


def shapes(model):
    return {n: tuple(p.shape) for n, p in model.state_dict().items()}


def evaluated(model_name):
    return pd.read_csv(EVAL / model_name / "performance" / f"metrics_{SPLIT}.csv").iloc[0]


def main(ratios):
    OUT.mkdir(parents=True, exist_ok=True)
    config = ev.load_config(ev.CONFIG)
    _, loader, _ = ev.build_dataloaders(config)
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    rows = []
    for ratio in ratios:
        name = f"pruned{ratio}_seed{SEED}"
        model = pr.load_fp32()
        pr.prune(model, ratio / 100, torch.zeros(1, 3, 224, 224))
        finetuned = torch.load(pr.MODELS_DIR / f"{name}_selected.pt", map_location="cpu", weights_only=False)
        if shapes(model) != shapes(finetuned):
            sys.exit(f"CHECK FAILED: rebuilt {ratio}% architecture differs from the saved fine-tuned model")
        unfused = ev.logits(model.eval(), images)
        fused = ev.logits(ev.fused_module_torchscript(model), images)
        if not torch.equal(unfused.argmax(1), fused.argmax(1)):
            sys.exit(f"CHECK FAILED: {ratio}% fused and unfused predictions differ")
        probs = fused.softmax(1).numpy()
        point, _ = ev.scores(y, probs, ev.NUM_CLASSES)
        boot = ev.bootstrap({"pruned": probs}, y, ev.NUM_CLASSES)["pruned"]
        saved = pd.read_csv(pr.RESULTS / name / "metrics.csv").set_index("stage").loc["before_finetuning"]
        diff = abs(point["accuracy"] - saved["accuracy"]) * len(y)
        if diff > 2:
            sys.exit(f"CHECK FAILED: {ratio}% before fine-tuning differs from prune.py by {diff:.0f} images")
        stage_rows = {"fp32": evaluated("fp32"), "finetuned": evaluated(name)}
        stage_rows["pruned"] = pd.Series({"model": f"{name}_before_finetuning", **{m: point[m] for m in METRICS},
                                          **{f"{m}_ci_low": ev.ci(boot[m])[0] for m in METRICS},
                                          **{f"{m}_ci_high": ev.ci(boot[m])[1] for m in METRICS}})
        for stage in STAGES:
            r = stage_rows[stage]
            rows.append({"ratio_pct": ratio, "stage": stage, **{k: r[k] for k in METRICS},
                         **{f"{k}_ci_low": r[f"{k}_ci_low"] for k in METRICS},
                         **{f"{k}_ci_high": r[f"{k}_ci_high"] for k in METRICS}})
        eff = {s: pd.read_csv(EVAL / m / "efficiency" / f"metrics_{SPLIT}.csv").iloc[0]
               for s, m in (("fp32", "fp32"), ("finetuned", name))}
        env = {s: pd.read_csv(EVAL / m / "environment" / f"metrics_{SPLIT}.csv").iloc[0]
               for s, m in (("fp32", "fp32"), ("finetuned", name))}
        plot(ratio, name, stage_rows, eff, env)
        print(f"{ratio}%: F1 FP32 {stage_rows['fp32']['f1_macro']:.4f}, pruned {point['f1_macro']:.4f} "
              f"(prune.py {saved['f1_macro']:.4f}), fine-tuned {stage_rows['finetuned']['f1_macro']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"stages_{SPLIT}.csv", index=False)


def plot(ratio, name, rows, eff, env):
    fig = plt.figure(figsize=(16, 5.6))
    ax = fig.add_axes([0.05, 0.17, 0.6, 0.66])
    x, width = np.arange(len(METRICS)), 0.27
    for k, (stage, st) in enumerate(STAGES.items()):
        r = rows[stage]
        values = np.array([r[m] * 100 for m in METRICS])
        err = np.array([[values[i] - r[f"{m}_ci_low"] * 100 for i, m in enumerate(METRICS)],
                        [r[f"{m}_ci_high"] * 100 - values[i] for i, m in enumerate(METRICS)]])
        bars = ax.bar(x + (k - 1) * width, values, width * 0.95, yerr=err, color=st["color"], hatch=st["hatch"],
                      edgecolor=INK, linewidth=0.7, label=st["label"],
                      error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5))
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=5, fontsize=7.5, rotation=90, color=INK)
    ax.set_xticks(x, METRICS.values())
    ax.set_ylim(0, 122)
    ax.set_ylabel("Score (%), macro average except accuracy")
    ax.grid(axis="x", visible=False)
    ax.set_title("Performance (error bars: 95% bootstrap CI, 1,000 resamples)", loc="left", fontsize=10.5,
                 fontweight="semibold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)

    panels = [("Size (MB)", eff["fp32"]["size_mb"], eff["finetuned"]["size_mb"], "{:.1f}"),
              ("GFLOPs / image", eff["fp32"]["gflops"], eff["finetuned"]["gflops"], "{:.2f}"),
              ("Latency, 1 thread (ms)", eff["fp32"]["latency_median_ms_1t"],
               eff["finetuned"]["latency_median_ms_1t"], "{:.1f}"),
              ("Energy / 1,000 img (Wh, est.)", env["fp32"]["energy_wh_per_1000"],
               env["finetuned"]["energy_wh_per_1000"], "{:.2f}")]
    for i, (title, base, value, fmt) in enumerate(panels):
        a = fig.add_axes([0.7 + (i % 2) * 0.155, 0.53 - (i // 2) * 0.4, 0.12, 0.27])
        bars = a.bar([0, 1], [base, value], 0.7, color=[STAGES["fp32"]["color"], STAGES["finetuned"]["color"]],
                     hatch=["", "//"], edgecolor=INK, linewidth=0.7)
        a.bar_label(bars, labels=[fmt.format(base), fmt.format(value)], padding=2, fontsize=8)
        a.set_xticks([0, 1], ["FP32", f"Pruned\n{ratio}%"], fontsize=8)
        a.set_title(f"{title}: {(value / base - 1) * 100:+.0f}%", loc="left", fontsize=9, fontweight="semibold")
        a.margins(y=0.25)
        a.grid(axis="x", visible=False)
        a.tick_params(axis="y", labelsize=7.5)
    fig.text(0.7, 0.9, "Efficiency (same architecture before and after fine-tuning)", fontsize=10.5,
             fontweight="semibold")
    fig.suptitle(f"Pruned {ratio}% (seed {SEED}) vs FP32 baseline: before and after fine-tuning (validation, 1,532 "
                 "images)", x=0.01, ha="left", fontweight="semibold", fontsize=12)
    fig.text(0.01, 0.01, "CPU, fused + TorchScript. Before fine-tuning: rebuilt deterministically with prune.py's "
             "pruning and evaluated like evaluate.py. Fine-tuning: 10 epochs, Adam, lr 1e-4 cosine, best validation "
             "macro F1 epoch. Energy: CodeCarbon estimate (no power counter).", fontsize=7.5, color=NOTE)
    fig.savefig(OUT / f"stages_{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"stages_{name}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=int, nargs="+", default=[10, 20, 30, 40])
    main(parser.parse_args().ratios)
