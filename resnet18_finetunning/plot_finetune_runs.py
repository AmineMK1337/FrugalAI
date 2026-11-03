"""Accuracy, loss and summary plots for the two fine-tuned ResNet-18 runs.

Reads the metrics saved during training; nothing is re-run. final_metrics.json is
scored on the validation split. Writes PNG (300 dpi) and PDF to results/plots/.

Usage: python plot_finetune_runs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "plots"
RUNS = {  # the colour follows the model in every figure
    "constant_lr": ("ResNet-18 fine-tuned on HAM10000 (constant LR)", "strong_aug_resnet18", "#2a78d6"),
    "cosine_lr": ("ResNet-18 fine-tuned on HAM10000 (cosine LR)", "strong_cosine25_resnet18", "#eb6834"),
}
SURFACE, MUTED, GRID = "#fcfcfb", "#52514e", "#e6e5e0"
TRAIN_STYLE = (0, (4, 3))
METRICS = {"accuracy": "Accuracy", "macro_precision": "Macro\nprecision", "macro_recall": "Macro\nrecall",
           "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "pdf.fonttype": 42,
    "legend.frameon": False, "axes.titlesize": 12, "axes.titleweight": "semibold",
})


def load():
    runs = {}
    for key, (title, folder, colour) in RUNS.items():
        history = json.loads((RESULTS / folder / "training_history.json").read_text())
        metrics = json.loads((RESULTS / folder / "final_metrics.json").read_text())
        rows = history["history"]
        runs[key] = {"title": title, "short": title.split("(")[1].rstrip(")"), "colour": colour, "rows": rows,
                     "epochs": [r["epoch"] for r in rows], "best": history["best_epoch"],
                     "final": metrics["final_metrics"]}
    return runs


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def series(run, key):
    factor = 100 if key.endswith("acc") else 1
    return [r[key] * factor for r in run["rows"]]


def finish_epoch_axis(ax, ylabel):
    ax.set_xlim(1, 25)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)


def mark_best(ax, run, key):
    ax.plot(run["best"], series(run, key)[run["epochs"].index(run["best"])], "o", ms=8, color=run["colour"],
            mec=SURFACE, mew=2, zorder=5, label=f"Best epoch ({run['best']})")


def one_model(key, run):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.2})
    for ax, metric, ylabel, name in ((axes[0], "acc", "Accuracy (%)", "Accuracy"),
                                     (axes[1], "loss", "Class-weighted cross-entropy", "Loss")):
        ax.plot(run["epochs"], series(run, f"train_{metric}"), color=run["colour"], lw=1.5, ls=TRAIN_STYLE, label="Training")
        ax.plot(run["epochs"], series(run, f"val_{metric}"), color=run["colour"], lw=2, label="Validation")
        mark_best(ax, run, f"val_{metric}")
        finish_epoch_axis(ax, ylabel)
        ax.set_title(name, loc="left")
        ax.legend()
    fig.suptitle(run["title"], fontsize=13, fontweight="semibold", x=0.125, ha="left", y=1.02)
    save(fig, f"{key}_accuracy_loss")


def comparison(runs, metric, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for run in runs.values():
        ax.plot(run["epochs"], series(run, metric), color=run["colour"], lw=2, label=run["short"])
    finish_epoch_axis(ax, ylabel)
    ax.set_title(title, loc="left")
    ax.legend()
    save(fig, name)


def learning_rate(runs):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for run in runs.values():
        ax.plot(run["epochs"], [r["lr"] * 1e4 for r in run["rows"]], color=run["colour"], lw=2, label=run["short"])
    finish_epoch_axis(ax, "Learning rate (×10⁻⁴)")
    ax.set_title("Learning rate per epoch", loc="left")
    ax.legend(loc="lower left")
    save(fig, "learning_rate")


def final_metrics(runs):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(METRICS))
    width = 0.34
    for offset, run in zip((-1, 1), runs.values()):
        ax.bar(x + offset * (width / 2 + 0.01), [run["final"][m] for m in METRICS], width, color=run["colour"],
               label=run["short"])
    ax.set_xticks(x, METRICS.values())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on the validation split")
    ax.grid(axis="x", visible=False)
    ax.set_title("Best checkpoint: validation metrics", loc="left")
    ax.legend(ncol=2, loc="upper left")
    save(fig, "final_metrics")


if __name__ == "__main__":
    data = load()
    for run_key, run_data in data.items():
        one_model(run_key, run_data)
    comparison(data, "val_acc", "Validation accuracy (%)", "Validation accuracy per epoch", "compare_val_accuracy")
    comparison(data, "val_loss", "Class-weighted cross-entropy", "Validation loss per epoch", "compare_val_loss")
    learning_rate(data)
    final_metrics(data)
    print(f"Wrote figures to {OUT}")
