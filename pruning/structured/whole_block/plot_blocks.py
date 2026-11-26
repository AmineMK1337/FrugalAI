"""Figures of whole-block pruning (stage 1), validation split only. Reads the CSV / JSON results; no model is run.

  fig01_ranking_steps        greedy ranking, subset seed 42: loss increase of every candidate at every step
  fig02_ranking_seeds        single-removal loss increase and removal position, subset seeds 42 and 1, one plot
  fig03_bn_recalibration     single-removal loss with and without BatchNorm recalibration
  fig04_taylor_vs_loss       first-order Taylor score (logged only) against the loss increase used for selection
  fig05_compression          parameters and GMACs per k, measured and analytic
  fig06_training_curves      validation macro F1, validation loss and train loss per epoch, k = 1..5, FP32 control
  fig07_stages               raw, BN-recalibrated, selected epoch, epoch 10: macro F1, accuracy, balanced accuracy,
                             melanoma recall per k
  fig08_selected_metrics     all metrics of the selected models, k = 1..5, FP32 and FP32 control
  fig09_per_class_recall     per-class recall (sensitivity) of the selected models: not saved by prune_blocks.py,
                             skipped here (evaluate.py writes per-class metrics in step 3)
  fig10_old_vs_new           selected and before-fine-tuning macro F1: this pipeline vs commit 06be969 (k = 1..4)
  fig11_tradeoff             macro F1 against GMACs: block pruning, filter pruning (seed 42), FP32, FP32 control
  fig12_start_check          fine-tuning from the recalibrated vs the raw model (only if a *_rawstart run exists)
  fig13_measured_speed       measured CPU speed-up vs the FLOPs ratio, and macro F1 vs latency (evaluate.py results)
  summary_seed42.csv         one row per k: removed blocks, parameters, GMACs, metrics per stage

Writes pruning/structured/whole_block/results/report/ (.pdf and .png).
Usage: python pruning/structured/whole_block/plot_blocks.py [--seed 42]
"""
import argparse
import io
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS, SELECTION = HERE / "results", HERE / "results" / "selection"
FILTER = ROOT / "pruning" / "structured" / "filter" / "results"
OUT = RESULTS / "report"
OLD_COMMIT = "06be969"

INK, NOTE, MUTED, GRID = "#1f1e1c", "#52514e", "#898781", "#e1e0d9"
K_COLORS = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281"}  # sequential blue: more removed
STAGES = {"raw": ("Raw (blocks removed)", "#eb6834", "xx"), "bn_recalibrated": ("BN recalibrated", "#eda100", ".."),
          "selected": ("Fine-tuned, selected epoch", "#2a78d6", ""), "epoch10": ("Fine-tuned, epoch 10", "#1baf7a", "//")}
SEED_COLORS = {42: "#2a78d6", 1: "#eb6834"}
CHOSEN, OTHER = "#2a78d6", "#c3c2b7"
METRICS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced\naccuracy", "precision_macro": "Precision",
           "specificity_macro": "Specificity", "f1_macro": "Macro F1", "auc_macro": "AUC",
           "melanoma_recall": "Melanoma\nrecall"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.edgecolor": "#c3c2b7", "axes.titleweight": "bold", "axes.titlesize": 11})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {name}")


def legend_below(ax, ncol, y=-0.14, **kw):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=9, **kw)


def runs(seed):
    out = {}
    for k in range(1, 6):
        folder = RESULTS / f"blocks{k}_seed{seed}"
        if (folder / "summary.json").exists():
            out[k] = dict(summary=json.loads((folder / "summary.json").read_text()),
                          metrics=pd.read_csv(folder / "metrics.csv"),
                          history=pd.read_csv(folder / "training_history.csv"))
    return out


def stage_row(run, stage):
    m = run["metrics"]
    return m[m["stage"].str.startswith(stage)].iloc[0]


def old_results(seed):
    """Selected and before-fine-tuning metrics of the removed implementation (commit 06be969), from git, cached."""
    cache = OUT / f"old_{OLD_COMMIT}_seed{seed}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for k in range(1, 5):
        base = f"pruning/structured/block/results/blocks{k}_seed{seed}"
        try:
            metrics = pd.read_csv(io.StringIO(subprocess.run(
                ["git", "show", f"{OLD_COMMIT}:{base}/metrics.csv"], cwd=ROOT, check=True, capture_output=True,
                text=True).stdout))
            summary = json.loads(subprocess.run(["git", "show", f"{OLD_COMMIT}:{base}/summary.json"], cwd=ROOT,
                                                check=True, capture_output=True, text=True).stdout)
        except subprocess.CalledProcessError:
            continue
        for _, r in metrics.iterrows():
            rows.append({"k": k, "stage": r["stage"], "removed_blocks": " ".join(summary.get("removed_blocks", [])),
                         "gmacs_after": summary.get("gmacs_after"), **{m: r[m] for m in METRICS}})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table


# ── Ranking ──────────────────────────────────────────────────────────────────

def fig_ranking_steps(seed):
    table = pd.read_csv(SELECTION / f"ranking_seed{seed}.csv")
    table = table[table["step"] > 0]
    steps = sorted(table["step"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(3.6 * len(steps), 3.6), sharex=True)
    fig.subplots_adjust(wspace=0.55)
    xmax = table["loss_increase"].max() * 1.9
    for ax, step in zip(axes, steps):
        t = table[table["step"] == step].sort_values("loss_increase", ascending=False)
        colors = [CHOSEN if c else OTHER for c in t["chosen"]]
        ax.barh(t["candidate"], t["loss_increase"], color=colors, height=0.6)
        for i, (_, r) in enumerate(t.iterrows()):
            ax.text(r["loss_increase"] + xmax * 0.02, i, f"+{r['loss_increase']:.2f}  F1 {r['f1_macro']:.2f}",
                    va="center", fontsize=8, color=INK if r["chosen"] else NOTE)
        removed = t["removed_before"].fillna("").iloc[0]
        ax.set_title(f"Step {step}", loc="left", pad=18)
        ax.text(0, 1.02, f"already removed: {removed.replace(' ', ', ') or 'none'}", transform=ax.transAxes,
                fontsize=7.5, color=NOTE, va="bottom")
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Loss increase vs full model")
    fig.suptitle(f"Greedy block ranking (subset seed {seed}): class-weighted CE increase after BN recalibration, "
                 "2,000 train images; blue = removed at this step", x=0.01, ha="left", fontsize=10.5, y=1.1)
    save(fig, "fig01_ranking_steps")


def fig_ranking_seeds(seeds):
    comparison = pd.read_csv(SELECTION / "ranking_comparison.csv")
    blocks = comparison["block"].tolist()
    x, width = np.arange(len(blocks)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, seed in enumerate(seeds):
        values = comparison[f"single_removal_loss_increase_seed{seed}"]
        bars = ax.bar(x + (i - 0.5) * width, values, width - 0.04, color=SEED_COLORS[seed],
                      label=f"Subset seed {seed}")
        for bar, pos in zip(bars, comparison[f"position_seed{seed}"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"#{pos}", ha="center", fontsize=8.5,
                    color=INK)
    ax.set_xticks(x, blocks)
    ax.set_ylabel("Loss increase when removed alone (step 1)")
    ax.set_title("Ranking stability across two ranking subsets (#n = greedy removal position)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, comparison[[c for c in comparison if c.startswith("single_removal_loss")]].max().max() * 1.3)
    ax.text(0.01, 0.97, "layer1.0 and layer1.1 swap places (same shape, near tie):\n"
            "same blocks per stage removed for every k, so same parameters and MACs",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=NOTE)
    legend_below(ax, len(seeds), y=-0.1)
    save(fig, "fig02_ranking_seeds")


def fig_bn_recalibration(seed):
    table = pd.read_csv(SELECTION / f"ranking_seed{seed}.csv")
    full = table[table["step"] == 0].iloc[0]
    t = table[table["step"] == 1].sort_values("candidate")
    x, width = np.arange(len(t)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, t["loss_no_bn_recal"], width - 0.04, color="#eb6834", hatch="xx", edgecolor="white",
           label="Without BN recalibration")
    ax.bar(x + width / 2, t["loss"], width - 0.04, color="#2a78d6", label="With BN recalibration (used)")
    ax.axhline(full["loss_no_bn_recal"], color="#eb6834", ls=":", lw=1.4,
               label=f"Full model, original BN ({full['loss_no_bn_recal']:.3f})")
    ax.axhline(full["loss"], color="#2a78d6", ls="--", lw=1.4,
               label=f"Full model, recalibrated BN ({full['loss']:.3f})")
    ax.set_ylim(full["loss_no_bn_recal"] * 0.6, None)
    ax.set_yscale("log")
    ax.set_xticks(x, t["candidate"])
    ax.set_ylabel("Class-weighted CE on the subset (log)")
    ax.set_title(f"Removing one block: effect of BatchNorm recalibration (subset seed {seed})", loc="left")
    ax.grid(axis="x", visible=False)
    legend_below(ax, 2, y=-0.1)
    save(fig, "fig03_bn_recalibration")


def fig_taylor(seed):
    table = pd.read_csv(SELECTION / f"ranking_seed{seed}.csv")
    table = table[table["step"] > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for step, color in zip(sorted(table["step"].unique()), K_COLORS.values()):
        t = table[table["step"] == step]
        ax.scatter(t["loss_increase"], t["taylor"], s=55, color=color, edgecolor="white", linewidth=1.5, zorder=3,
                   label=f"Step {step}")
        if step == 1:
            for j, (_, r) in enumerate(t.sort_values("loss_increase").iterrows()):
                ax.annotate(r["candidate"], (r["loss_increase"], r["taylor"]), xytext=(6, 4 if j % 2 == 0 else -12),
                            textcoords="offset points", fontsize=8, color=NOTE)
    ax.set_xlabel("Loss increase after removal + BN recalibration (used for selection)")
    ax.set_ylabel("Taylor score |dL/ds| at s = 1 (logged only)")
    ax.set_yscale("log")
    ax.set_title(f"First-order Taylor score vs actual removal cost (subset seed {seed}; labels: step 1)", loc="left")
    legend_below(ax, 5, y=-0.15)
    save(fig, "fig04_taylor_vs_loss")


# ── Pruned models ────────────────────────────────────────────────────────────

def fig_compression(r):
    ks = sorted(r)
    s = {k: r[k]["summary"] for k in ks}
    p0, m0 = s[ks[0]]["params_before"], s[ks[0]]["gmacs_before"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, base, scale, label, analytic in (
            (axes[0], "params_after", p0, 1e6, "Parameters (M)", "params_removed_analytic"),
            (axes[1], "gmacs_after", m0, 1, "GMACs", "gmacs_removed_analytic")):
        values = [s[k][key] / scale for k in ks]
        expected = [(base - s[k][analytic]) / scale for k in ks]
        ax.bar([0] + ks, [base / scale] + values, color=[MUTED] + [K_COLORS[k] for k in ks], width=0.65)
        ax.scatter(ks, expected, marker="_", s=500, color=INK, zorder=4, label="Analytic (removed blocks)")
        for k, v in zip(ks, values):
            ax.text(k, v * 1.01, f"−{(1 - v * scale / base) * 100:.1f}%", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks([0] + ks, ["FP32"] + [f"k={k}" for k in ks])
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
        legend_below(ax, 1, y=-0.1)
    axes[0].set_title("Parameters", loc="left")
    axes[1].set_title("Compute (batch 1, 224×224)", loc="left")
    order = s[ks[-1]]["removed_blocks"]
    fig.suptitle("Whole-block pruning: removal order " + " → ".join(order), x=0.01, ha="left", fontsize=10.5,
                 y=1.02)
    save(fig, "fig05_compression")


def fig_training_curves(r, seed, fp32):
    control = FILTER / f"control_seed{seed}" / "training_history.csv"
    control = pd.read_csv(control) if control.exists() else None
    panels = (("val_f1_macro", "Validation macro F1"), ("val_val_loss", "Validation loss (class-weighted CE)"),
              ("train_loss", "Train loss (augmented)"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (col, title) in zip(axes, panels):
        for k in sorted(r):
            h = r[k]["history"]
            if col == "val_val_loss":
                col_ = "val_val_loss" if "val_val_loss" in h else "val_loss"
            else:
                col_ = col
            h = h.dropna(subset=[col_])
            ax.plot(h["epoch"], h[col_], color=K_COLORS[k], lw=2, marker="o", ms=4,
                    label=f"k={k} ({' '.join(r[k]['summary']['removed_blocks'])})" if k == 1 else
                    f"k={k} (+{r[k]['summary']['removed_blocks'][-1]})")
            if col == "val_f1_macro":
                e = r[k]["summary"]["selected_epoch"]
                ax.scatter([e], [h.set_index("epoch").loc[e, col_]], s=110, facecolor="none", edgecolor=INK,
                           lw=1.4, zorder=5)
        if control is not None:
            c = control.dropna(subset=[col if col in control else "val_val_loss"])
            ccol = col if col in control else "val_val_loss"
            ax.plot(c["epoch"], c[ccol], color=MUTED, lw=1.6, ls="--", label=f"FP32 control (seed {seed})")
        if col == "val_f1_macro":
            ax.axhline(fp32["f1_macro"], color=INK, ls=":", lw=1.2, label="FP32 baseline")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch (0 = start of fine-tuning)")
        ax.set_xticks(range(0, 11))
    axes[0].text(0.99, 0.03, "○ = selected epoch", transform=axes[0].transAxes, ha="right", fontsize=8, color=NOTE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
    fig.suptitle(f"Fine-tuning after whole-block pruning (seed {seed}, starts from the BN-recalibrated model)",
                 x=0.01, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig06_training_curves")


def fig_stages(r, fp32):
    ks = sorted(r)
    panels = (("f1_macro", "Macro F1"), ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
              ("melanoma_recall", "Melanoma recall"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    width = 0.2
    for ax, (metric, title) in zip(axes.flat, panels):
        for i, (stage, (label, color, hatch)) in enumerate(STAGES.items()):
            values = [stage_row(r[k], stage)[metric] for k in ks]
            ax.bar(np.array(ks) + (i - 1.5) * width, values, width - 0.02, color=color, hatch=hatch,
                   edgecolor="white", label=label)
        ax.axhline(fp32[metric], color=INK, ls=":", lw=1.3, label="FP32 baseline")
        for k in ks:
            ax.text(k + 0.5 * width, stage_row(r[k], "selected")[metric] + 0.015,
                    f"{stage_row(r[k], 'selected')[metric]:.3f}", ha="center", fontsize=7.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_xticks(ks, [f"k={k}" for k in ks])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=9)
    fig.suptitle("Validation metrics at each stage (seed 42); labels: selected epoch", x=0.01, ha="left",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig07_stages")


def fig_selected_metrics(r, seed, fp32):
    control = FILTER / f"control_seed{seed}" / "metrics.csv"
    entries = [("FP32 baseline", MUTED, "", fp32)]
    if control.exists():
        c = pd.read_csv(control)
        entries.append((f"FP32 control (fine-tuned, seed {seed})", "#c3c2b7", "//",
                        c[c["stage"].str.startswith("selected")].iloc[0]))
    entries += [(f"k={k}", K_COLORS[k], "", stage_row(r[k], "selected")) for k in sorted(r)]
    x, width = np.arange(len(METRICS)), 0.8 / len(entries)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for i, (label, color, hatch, row) in enumerate(entries):
        ax.bar(x + (i - (len(entries) - 1) / 2) * width, [row[m] for m in METRICS], width - 0.01, color=color,
               hatch=hatch, edgecolor="white", label=label)
    ax.set_xticks(x, METRICS.values())
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Selected models (best validation macro F1 epoch), validation split", loc="left")
    ax.grid(axis="x", visible=False)
    legend_below(ax, 4, y=-0.16)
    save(fig, "fig08_selected_metrics")


def fig_old_vs_new(r, seed, fp32):
    old = old_results(seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    ks = sorted(r)
    for ax, (title, new_stages, old_stage) in zip(axes, (
            ("After fine-tuning (selected epoch)", ["selected"], "selected"),
            ("Before fine-tuning", ["raw", "bn_recalibrated"], "before_finetuning"))):
        for stage in new_stages:
            label, color, _ = STAGES[stage]
            ax.plot(ks, [stage_row(r[k], stage)["f1_macro"] for k in ks], color=color, lw=2, marker="o",
                    label=f"New: {label.lower()}")
        if len(old):
            o = old[old["stage"].str.startswith(old_stage)].sort_values("k")
            ax.plot(o["k"], o["f1_macro"], color=MUTED, lw=2, ls="--", marker="s",
                    label=f"Old ({OLD_COMMIT}, one-shot ranking, no BN recal.)")
        ax.axhline(fp32["f1_macro"], color=INK, ls=":", lw=1.2, label="FP32 baseline")
        ax.set_xticks(ks, [f"k={k}" for k in ks])
        ax.set_title(title, loc="left")
        ax.set_ylabel("Validation macro F1")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.suptitle(f"Sanity check against the removed implementation (seed {seed}); "
                 "the removed blocks may differ between the two", x=0.01, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig10_old_vs_new")


def fig_tradeoff(r, seed, fp32):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ks = sorted(r)
    g = [r[k]["summary"]["gmacs_after"] for k in ks]
    f = [stage_row(r[k], "selected")["f1_macro"] for k in ks]
    ax.plot(g, f, color="#2a78d6", lw=2, marker="o", ms=8, label="Whole-block pruning (k = 1..5)", zorder=3)
    for k, gx, fy in zip(ks, g, f):
        ax.annotate(f"k={k}", (gx, fy), xytext=(4, 9), textcoords="offset points", fontsize=8.5, color=NOTE)
    filt = []
    for ratio in (10, 20, 30, 40):
        path = FILTER / f"pruned{ratio}_seed{seed}" / "summary.json"
        if path.exists():
            s = json.loads(path.read_text())
            filt.append((ratio, s["gmacs_after"], s["selected"]["f1_macro"]))
    if filt:
        ax.plot([x[1] for x in filt], [x[2] for x in filt], color="#eb6834", lw=2, marker="D", ms=7,
                label="Filter pruning (10–40%)", zorder=3)
        for ratio, gx, fy in filt:
            ax.annotate(f"{ratio}%", (gx, fy), xytext=(4, -15), textcoords="offset points", fontsize=8.5, color=NOTE)
    g0 = r[ks[0]]["summary"]["gmacs_before"]
    ax.scatter([g0], [fp32["f1_macro"]], s=90, color=INK, marker="*", label="FP32 baseline", zorder=4)
    control = FILTER / f"control_seed{seed}" / "summary.json"
    if control.exists():
        ax.scatter([g0], [json.loads(control.read_text())["selected"]["f1_macro"]], s=70, color=MUTED, marker="^",
                   label=f"FP32 control (seed {seed})", zorder=4)
    ax.set_xlabel("GMACs per image (lower = cheaper)")
    ax.set_ylabel("Validation macro F1 (selected epoch)")
    ax.set_title(f"Accuracy–compute trade-off, seed {seed}", loc="left")
    legend_below(ax, 2, y=-0.13)
    save(fig, "fig11_tradeoff")


def fig_start_check(seed, fp32):
    pairs = []
    for k in range(1, 6):
        a, b = RESULTS / f"blocks{k}_seed{seed}", RESULTS / f"blocks{k}_seed{seed}_rawstart"
        if (b / "training_history.csv").exists():
            pairs.append((k, pd.read_csv(a / "training_history.csv"), pd.read_csv(b / "training_history.csv")))
    if not pairs:
        print("no *_rawstart run: fig12 skipped")
        return
    fig, axes = plt.subplots(1, 2 * len(pairs), figsize=(12 * len(pairs), 4.3), squeeze=False)
    for i, (k, recal, raw) in enumerate(pairs):
        for ax, col, title in ((axes[0, 2 * i], "val_f1_macro", "Validation macro F1"),
                               (axes[0, 2 * i + 1], "val_val_loss", "Validation loss")):
            ax.plot(recal["epoch"], recal[col], color=STAGES["bn_recalibrated"][1], lw=2, marker="o",
                    label="Start: BN recalibrated (used)")
            ax.plot(raw["epoch"], raw[col], color=STAGES["raw"][1], lw=2, marker="s", ls="--",
                    label="Start: raw (as filter pruning)")
            if col == "val_f1_macro":
                ax.axhline(fp32["f1_macro"], color=INK, ls=":", lw=1.2, label="FP32 baseline")
            ax.set_title(f"k={k}: {title}", loc="left")
            ax.set_xlabel("Epoch")
            ax.set_xticks(range(0, 11))
            legend_below(ax, 3, y=-0.15)
    fig.suptitle(f"Where fine-tuning starts (seed {seed})", x=0.01, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig12_start_check")


def fig_measured_speed(seed):
    """From evaluate.py (CPU, fused + TorchScript): does the MAC reduction turn into latency? And F1 vs latency."""
    folder = RESULTS / "evaluation" / "comparison"
    if not (folder / "efficiency" / "table_validation.csv").exists():
        print("no evaluate.py results: fig13 skipped")
        return
    eff = pd.read_csv(folder / "efficiency" / "table_validation.csv").set_index("model")
    perf = pd.read_csv(folder / "performance" / "table_validation.csv").set_index("model")
    names = [n for n in eff.index if n.startswith("blocks") and n.endswith(f"_seed{seed}")]
    ks = [int(n[6:].split("_")[0]) for n in names]
    ideal = [eff.loc["fp32", "gflops"] / eff.loc[n, "gflops"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(ks, ideal, color=INK, ls=":", lw=1.5, marker="_", ms=14, label="Ideal: FP32 FLOPs / model FLOPs")
    for threads, color, marker in ((1, "#2a78d6", "o"), (4, "#eb6834", "s")):
        values = [eff.loc[n, f"speedup_vs_fp32_{threads}t"] for n in names]
        ax.plot(ks, values, color=color, lw=2, marker=marker, ms=7,
                label=f"Measured, {threads} thread{'s' if threads > 1 else ''}")
        for k, v in zip(ks, values):
            ax.annotate(f"{v:.2f}×", (k, v), xytext=(0, -15 if threads == 4 else 8), textcoords="offset points",
                        ha="center", fontsize=8, color=NOTE)
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xticks(ks, [f"k={k}" for k in ks])
    ax.set_ylabel("Speed-up vs FP32 (median latency, batch 1)")
    ax.set_title("Does fewer FLOPs mean faster? (CPU)", loc="left")
    legend_below(ax, 2, y=-0.1)

    ax = axes[1]
    entries = [("fp32", "FP32 baseline", INK, "*"), (f"control_seed{seed}", f"FP32 control (seed {seed})", MUTED, "^")]
    for name, label, color, marker in entries:
        if name in eff.index:
            ax.errorbar(eff.loc[name, "latency_median_ms_1t"], perf.loc[name, "f1_macro"],
                        yerr=[[perf.loc[name, "f1_macro"] - perf.loc[name, "f1_macro_ci_low"]],
                              [perf.loc[name, "f1_macro_ci_high"] - perf.loc[name, "f1_macro"]]],
                        fmt=marker, ms=10, color=color, capsize=3, label=label, zorder=4)
    lat = [eff.loc[n, "latency_median_ms_1t"] for n in names]
    f1 = [perf.loc[n, "f1_macro"] for n in names]
    low = [perf.loc[n, "f1_macro"] - perf.loc[n, "f1_macro_ci_low"] for n in names]
    high = [perf.loc[n, "f1_macro_ci_high"] - perf.loc[n, "f1_macro"] for n in names]
    ax.errorbar(lat, f1, yerr=[low, high], fmt="-o", color="#2a78d6", lw=2, ms=7, capsize=3,
                label="Whole-block pruning", zorder=3)
    for k, x, y in zip(ks, lat, f1):
        ax.annotate(f"k={k}", (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=NOTE)
    ax.set_xlabel("Median latency, 1 thread, batch 1 (ms)")
    ax.set_ylabel("Validation macro F1 (95% bootstrap CI)")
    ax.set_title("Accuracy–latency trade-off (CPU, fused + TorchScript)", loc="left")
    legend_below(ax, 3, y=-0.13)
    cpu = json.loads((RESULTS / "evaluation" / "run_info.json").read_text()).get("cpu", "")
    fig.suptitle(f"Measured on {cpu}, seed {seed}", x=0.01, ha="left", fontsize=10.5, y=1.02)
    save(fig, "fig13_measured_speed")


def summary_table(r):
    rows = []
    for k in sorted(r):
        s = r[k]["summary"]
        row = {"k": k, "removed_blocks": " ".join(s["removed_blocks"]), "params": s["params_after"],
               "param_reduction_pct": s["param_reduction_pct"], "gmacs": s["gmacs_after"],
               "gmac_reduction_pct": s["gmac_reduction_pct"], "selected_epoch": s["selected_epoch"],
               "fused_max_abs_logit_difference": max(s["fused_vs_unfused_max_abs_logit_difference"].values())}
        for stage in STAGES:
            for metric in ("f1_macro", "accuracy", "balanced_accuracy", "specificity_macro", "melanoma_recall"):
                row[f"{stage}_{metric}"] = stage_row(r[k], stage)[metric]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary_seed42.csv", index=False)
    print(table[["k", "removed_blocks", "params", "gmacs", "selected_epoch", "raw_f1_macro",
                 "bn_recalibrated_f1_macro", "selected_f1_macro", "epoch10_f1_macro"]].round(4).to_string(index=False))


def main(args):
    seeds = [int(c.split("seed")[1]) for c in pd.read_csv(SELECTION / "ranking_comparison.csv").columns
             if c.startswith("position_seed")]
    fig_ranking_steps(seeds[0])
    fig_ranking_seeds(seeds)
    fig_bn_recalibration(seeds[0])
    fig_taylor(seeds[0])
    r = runs(args.seed)
    if not r:
        print("no fine-tuned run yet: model figures skipped")
        return
    fp32 = json.loads((RESULTS / "fp32_validation.json").read_text())
    fig_compression(r)
    fig_training_curves(r, args.seed, fp32)
    fig_stages(r, fp32)
    fig_selected_metrics(r, args.seed, fp32)
    fig_old_vs_new(r, args.seed, fp32)
    fig_tradeoff(r, args.seed, fp32)
    fig_start_check(args.seed, fp32)
    fig_measured_speed(args.seed)
    summary_table(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
