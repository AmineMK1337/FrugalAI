"""Consolidated table and plots for structured pruning. Reads only saved files (nothing is re-run):
pruning/structured/filter/results/<run>/ (prune.py) and, when present, pruning/structured/filter/results/evaluation/<run>/ (evaluate.py).

Validation split only. Performance before / after fine-tuning comes from prune.py (GPU evaluation of the same
validation images); size, parameters, GFLOPs, latency and energy come from evaluate.py (CPU, fused + TorchScript).

Writes pruning/structured/filter/results/report/:
  consolidated.csv / consolidated.md     one row per run (+ FP32): target ratio, achieved channel / parameter / GMAC
                                         reduction, balanced accuracy, macro F1, melanoma recall before and after
                                         fine-tuning (selected epoch and epoch 10), size, latency at 1 and 4 threads
  pruned_minus_control.csv               pruned - FP32 control, paired by seed (selected epoch and epoch 10)
  f1_vs_ratio, f1_vs_gmacs, f1_vs_latency, channels_per_layer, training_curves, learning_curves,
  pruning_process (.pdf + .png)

Usage: python pruning/structured/filter/report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
EVAL = ROOT / "pruning" / "structured" / "filter" / "results" / "evaluation"
OUT = RESULTS / "report"
SPLIT = "validation"
INK, FP32_COLOR, CONTROL_COLOR, NOTE = "#1f1e1c", "#2a78d6", "#7a7a7a", "#555555"
SEED_MARKERS = {42: "o", 1: "s", 2: "^"}
RATIO_STYLE = {10: dict(color="#f2b48f", linestyle=":"), 20: dict(color="#e08a5c", linestyle="-."),
               30: dict(color="#c25420", linestyle="--"), 40: dict(color="#7a2e0c", linestyle="-")}
METRICS = ("balanced_accuracy", "f1_macro", "melanoma_recall")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.color": "#e5e5e5",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def evaluation_row(model):
    """Efficiency from evaluate.py, or NaNs if this model has not been evaluated yet."""
    path = EVAL / model / "efficiency" / f"metrics_{SPLIT}.csv"
    if not path.exists():
        return {}
    e = pd.read_csv(path).iloc[0]
    return {"size_mb": e["size_mb"], "params_eval": e.get("params"), "gflops": e["gflops"],
            "latency_ms_1t": e["latency_median_ms_1t"], "latency_p95_ms_1t": e["latency_p95_ms_1t"],
            "latency_ms_4t": e["latency_median_ms_4t"], "throughput_img_s_1t": e["throughput_img_s_1t"],
            "peak_memory_mb_1t": e["peak_memory_mb_1t"]}


def load_runs():
    runs = []
    for folder in sorted(p for p in RESULTS.iterdir() if (p / "summary.json").exists()):
        s = json.loads((folder / "summary.json").read_text())
        row = {"model": s["run"], "kind": "control" if s["target_ratio"] is None else "pruned",
               "target_ratio_pct": None if s["target_ratio"] is None else round(s["target_ratio"] * 100),
               "seed": s["seed"],
               "channel_reduction_pct": s.get("channel_reduction_pct_all_convs", 0.0),
               "channel_reduction_pct_excl_stem": s.get("channel_reduction_pct_excluding_stem", 0.0),
               "param_reduction_pct": s.get("param_reduction_pct", 0.0),
               "gmac_reduction_pct": s.get("gmac_reduction_pct", 0.0),
               "params": s["params_after"], "gmacs": s["gmacs_after"], "selected_epoch": s["selected_epoch"]}
        for stage in ("before_finetuning", "selected", "epoch10"):
            for m in METRICS + ("accuracy", "identical_to_fp32_pct"):
                row[f"{m}_{stage}"] = s[stage][m]
        row.update(evaluation_row(s["run"]))
        runs.append(row)
    return pd.DataFrame(runs)


def consolidated(runs):
    fp32 = json.loads((RESULTS / "fp32_validation.json").read_text())
    first = json.loads(next(RESULTS.glob("*/summary.json")).read_text())
    base = {"model": "fp32", "kind": "fp32", "target_ratio_pct": 0, "seed": None, "channel_reduction_pct": 0.0,
            "channel_reduction_pct_excl_stem": 0.0, "param_reduction_pct": 0.0, "gmac_reduction_pct": 0.0,
            "params": first["params_before"], "gmacs": first["gmacs_before"]}
    for stage in ("before_finetuning", "selected", "epoch10"):
        for m in METRICS + ("accuracy", "identical_to_fp32_pct"):
            base[f"{m}_{stage}"] = fp32[m]
    base.update(evaluation_row("fp32"))
    table = pd.concat([pd.DataFrame([base]), runs], ignore_index=True)
    order = {"fp32": 0, "control": 1, "pruned": 2}
    table = table.sort_values(by=["kind", "target_ratio_pct", "seed"], key=lambda c: c.map(order) if c.name == "kind"
                              else c).reset_index(drop=True)
    table.to_csv(OUT / "consolidated.csv", index=False)

    def pct(v):
        return "–" if pd.isna(v) else f"{v * 100:.2f}"

    def num(v, fmt):
        return "–" if v is None or pd.isna(v) else fmt.format(v)

    lines = ["| Model | Target | Channels − | Params − | GMACs − | Bal. acc. (before → after) | Macro F1 (before → "
             "after) | Mel. recall (before → after) | Epoch | Macro F1, epoch 10 | Size (MB) | Latency 1t / 4t (ms) |",
             "|" + "---|" * 12]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['model']} | {num(r['target_ratio_pct'], '{:.0f}%')} | {r['channel_reduction_pct']:.1f}% | "
            f"{r['param_reduction_pct']:.1f}% | {r['gmac_reduction_pct']:.1f}% | "
            f"{pct(r['balanced_accuracy_before_finetuning'])} → {pct(r['balanced_accuracy_selected'])} | "
            f"{pct(r['f1_macro_before_finetuning'])} → {pct(r['f1_macro_selected'])} | "
            f"{pct(r['melanoma_recall_before_finetuning'])} → {pct(r['melanoma_recall_selected'])} | "
            f"{num(r.get('selected_epoch'), '{:.0f}')} | {pct(r['f1_macro_epoch10'])} | {num(r.get('size_mb'), '{:.1f}')} | "
            f"{num(r.get('latency_ms_1t'), '{:.2f}')} / {num(r.get('latency_ms_4t'), '{:.2f}')} |")
    (OUT / "consolidated.md").write_text("\n".join(lines) + "\n\nFP32: no fine-tuning, so before = after. "
                                         "Performance: prune.py (validation). Size and latency: evaluate.py "
                                         "(CPU, fused + TorchScript; '–' = not evaluated yet).\n")
    return table


def pruned_minus_control(runs):
    rows = []
    controls = runs[runs["kind"] == "control"].set_index("seed")
    for _, r in runs[runs["kind"] == "pruned"].iterrows():
        if r["seed"] not in controls.index:
            continue
        c = controls.loc[r["seed"]]
        row = {"model": r["model"], "target_ratio_pct": r["target_ratio_pct"], "seed": r["seed"],
               "control": c["model"]}
        for stage in ("selected", "epoch10"):
            for m in METRICS + ("accuracy",):
                row[f"{m}_{stage}_pruned_minus_control"] = r[f"{m}_{stage}"] - c[f"{m}_{stage}"]
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        means = table.groupby("target_ratio_pct").mean(numeric_only=True).reset_index()
        means["model"] = "mean over seeds"
        table = pd.concat([table, means], ignore_index=True)
    table.to_csv(OUT / "pruned_minus_control.csv", index=False)
    return table


def reference_lines(ax, table, column):
    ax.axhline(table.loc[table["kind"] == "fp32", column].iloc[0] * 100, color=FP32_COLOR, linestyle="--",
               linewidth=1.3, label="FP32 baseline")
    control = table[table["kind"] == "control"]
    if len(control):
        ax.axhline(control[column].mean() * 100, color=CONTROL_COLOR, linestyle=(0, (1, 1)), linewidth=1.5,
                   label=f"FP32 control (fine-tuned, mean of {len(control)} seed(s))")


def plot_f1_vs_ratio(table):
    pruned = table[table["kind"] == "pruned"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for stage, style, label in (("before_finetuning", dict(color="#9a9a9a", marker="o", markerfacecolor="white",
                                                           linestyle="--"), "pruned, before fine-tuning"),
                                ("selected", dict(color=INK, marker="o", linestyle="-"),
                                 "pruned, after fine-tuning (best epoch)")):
        col = f"f1_macro_{stage}"
        for seed, rows in pruned.groupby("seed"):
            ax.scatter(rows["target_ratio_pct"], rows[col] * 100, marker=SEED_MARKERS.get(seed, "o"), s=22,
                       facecolor="white" if stage == "before_finetuning" else INK, edgecolor=style["color"], zorder=3)
        mean = pruned.groupby("target_ratio_pct")[col].mean() * 100
        ax.plot(mean.index, mean.values, linewidth=1.8, label=label + " (mean of seeds)", **style)
    reference_lines(ax, table, "f1_macro_selected")
    ax.set_xticks(sorted(pruned["target_ratio_pct"].unique()))
    ax.set_xlabel("Target channel ratio pruned (%)")
    ax.set_ylabel("Validation macro F1 (%)")
    ax.set_title("Macro F1 vs pruning ratio", loc="left", fontweight="semibold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    save(fig, "f1_vs_ratio")


def plot_f1_vs(table, x, xlabel, name, title):
    data = table.dropna(subset=[x])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for _, r in data.iterrows():
        if r["kind"] == "fp32":
            st = dict(color=FP32_COLOR, marker="*", s=140)
        elif r["kind"] == "control":
            st = dict(color=CONTROL_COLOR, marker="D", s=50)
        else:
            st = dict(color=RATIO_STYLE.get(r["target_ratio_pct"], {}).get("color", INK),
                      marker=SEED_MARKERS.get(r["seed"], "o"), s=55)
        ax.scatter(r[x], r["f1_macro_selected"] * 100, edgecolor=INK, linewidth=0.7, zorder=3, **st)
        label = "FP32" if r["kind"] == "fp32" else (f"control s{r['seed']}" if r["kind"] == "control"
                                                     else f"{r['target_ratio_pct']:.0f}% s{r['seed']}")
        ax.annotate(label, (r[x], r["f1_macro_selected"] * 100), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation macro F1 (%), after fine-tuning")
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.text(0, -0.2, "Star: FP32 baseline. Diamond: FP32 control. Darker = more pruning; marker shape = seed "
            "(circle 42, square 1, triangle 2).", transform=ax.transAxes, fontsize=8, color=NOTE)
    save(fig, name)


def plot_channels(runs):
    seed42 = runs[(runs["kind"] == "pruned") & (runs["seed"] == 42)]
    if seed42.empty:
        return
    layers = {r["model"]: pd.read_csv(RESULTS / r["model"] / "layers.csv") for _, r in seed42.iterrows()}
    first = next(iter(layers.values()))
    convs = first[first["type"] == "conv"]["layer"].tolist()
    x = np.arange(len(convs))
    width = 0.8 / (len(layers) + 1)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.4 + width / 2, first.set_index("layer").loc[convs, "out_before"], width, color="white",
           edgecolor=INK, linewidth=0.7, label="FP32 (before pruning)")
    for k, (model, t) in enumerate(sorted(layers.items(), key=lambda kv: kv[0])):
        ratio = int(model[len("pruned"):].split("_")[0])
        st = RATIO_STYLE.get(ratio, {})
        ax.bar(x - 0.4 + width * (k + 1.5), t.set_index("layer").loc[convs, "out_after"], width,
               color=st.get("color", "#c25420"), edgecolor=INK, linewidth=0.5,
               hatch={10: "..", 20: "//", 30: "xx", 40: "\\\\"}.get(ratio, ""), label=f"pruned {ratio}% (target)")
    ax.set_xticks(x, [c.replace("layer", "L").replace(".downsample.0", ".ds") for c in convs], rotation=60,
                  ha="right", fontsize=8)
    ax.set_ylabel("Output channels")
    ax.set_title("Output channels per convolution, before and after pruning (seed 42; stem conv1 and the layer1 "
                 "residual stream are protected)", loc="left", fontweight="semibold", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    save(fig, "channels_per_layer")


def plot_training_curves(runs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
    for ax, (m, label) in zip(axes, (("f1_macro", "Macro F1 (%)"), ("balanced_accuracy", "Balanced accuracy (%)"),
                                     ("melanoma_recall", "Melanoma recall (%)"))):
        for _, r in runs.sort_values(["kind", "target_ratio_pct", "seed"]).iterrows():
            h = pd.read_csv(RESULTS / r["model"] / "training_history.csv")
            if r["kind"] == "control":
                st = dict(color=CONTROL_COLOR, linestyle=(0, (1, 1)))
                name = f"FP32 control s{r['seed']}"
            else:
                st = dict(RATIO_STYLE.get(r["target_ratio_pct"], {}))
                name = f"pruned {r['target_ratio_pct']:.0f}% s{r['seed']}"
            ax.plot(h["epoch"], h[f"val_{m}"] * 100, marker=SEED_MARKERS.get(r["seed"], "o"), markersize=4,
                    markerfacecolor="white", linewidth=1.6 if r["seed"] == 42 else 0.9, label=name, **st)
        fp32 = json.loads((RESULTS / "fp32_validation.json").read_text())
        ax.axhline(fp32[m] * 100, color=FP32_COLOR, linestyle="--", linewidth=1.2, label="FP32 baseline")
        ax.set_xlabel("Fine-tuning epoch (0 = right after pruning)")
        ax.set_ylabel(label)
        ax.set_xticks(range(0, 11))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=8)
    fig.suptitle("Fine-tuning curves after structured pruning (validation split)", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout()
    save(fig, "training_curves")


def run_style(r):
    if r["kind"] == "control":
        return dict(color=CONTROL_COLOR, linestyle=(0, (1, 1))), f"FP32 control s{r['seed']}"
    return dict(RATIO_STYLE.get(r["target_ratio_pct"], {})), f"pruned {r['target_ratio_pct']:.0f}% s{r['seed']}"


def plot_learning_curves(runs):
    """Training loss (mean over the epoch, augmented images) and validation loss (class-weighted, eval
    preprocessing) per epoch. Epoch 0 = right after pruning (no training yet)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for _, r in runs.sort_values(["kind", "target_ratio_pct", "seed"]).iterrows():
        h = pd.read_csv(RESULTS / r["model"] / "training_history.csv")
        st, name = run_style(r)
        kw = dict(marker=SEED_MARKERS.get(r["seed"], "o"), markersize=4, markerfacecolor="white",
                  linewidth=1.6 if r["seed"] == 42 else 0.9, label=name, **st)
        axes[0].plot(h["epoch"][1:], h["train_loss"][1:], **kw)
        axes[1].plot(h["epoch"], h["val_val_loss"], **kw)
    axes[0].set_title("Training loss (train split, augmented)", loc="left", fontsize=10, fontweight="semibold")
    axes[1].set_title("Validation loss (epoch 0 = right after pruning)", loc="left", fontsize=10,
                      fontweight="semibold")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Fine-tuning epoch")
        ax.set_ylabel("Class-weighted cross-entropy")
        ax.set_xticks(range(0, 11))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=8)
    fig.suptitle("Learning curves of the fine-tuning after structured pruning", x=0.01, ha="left",
                 fontweight="semibold")
    fig.text(0.01, -0.1, "Validation loss on a log scale: right after pruning it is far above the fine-tuned values. "
             "Adam, lr 1e-4 with cosine decay over 10 epochs, batch 32.", fontsize=8, color=NOTE)
    fig.tight_layout()
    save(fig, "learning_curves")


def plot_process(runs):
    """Top: the pipeline. Bottom: measured validation macro F1 at each stage (FP32, after pruning, each epoch)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(15, 9.5))
    top = fig.add_axes([0.02, 0.6, 0.96, 0.36])
    top.set_axis_off()
    top.set_xlim(0, 100)
    top.set_ylim(0, 40)
    steps = [
        ("1. FP32 model", "fine-tuned ResNet-18\n11.18 M params\n1.82 GMACs\n(quantizable variant)", "#ffffff"),
        ("2. Dependency groups", "Torch-Pruning graph:\nconvs joined by a\nresidual add share\nchannel indices", "#f2f2f2"),
        ("3. Rank filters", "L1 norm of each\noutput filter\n(averaged over the\nconvs of a group)", "#f2f2f2"),
        ("4. Remove filters", "same target ratio\nin every group;\nconv1, layer1 residual\nstream and fc kept", "#e3e3e3"),
        ("5. Fine-tune", "10 epochs, Adam,\nlr 1e-4 cosine,\nclass-weighted CE,\nstrong augmentation", "#d4d4d4"),
        ("6. Select & save", "best validation\nmacro F1 epoch\n(+ epoch 10);\nwhole model + TorchScript", "#c4c4c4"),
        ("7. Evaluate", "CPU, fused +\nTorchScript, 1 & 4\nthreads, same protocol\nas every model", "#ffffff"),
    ]
    width, gap = 12.2, 2.1
    for i, (title, body, face) in enumerate(steps):
        x = 0.5 + i * (width + gap)
        top.add_patch(FancyBboxPatch((x, 8), width, 24, boxstyle="round,pad=0.3,rounding_size=1.2",
                                     facecolor=face, edgecolor=INK, linewidth=1))
        top.text(x + width / 2, 28.5, title, ha="center", va="center", fontsize=9.5, fontweight="semibold")
        top.text(x + width / 2, 18.5, body, ha="center", va="center", fontsize=8.2, linespacing=1.4)
        if i < len(steps) - 1:
            top.add_patch(FancyArrowPatch((x + width + 0.2, 19), (x + width + gap - 0.2, 19), arrowstyle="-|>",
                                          mutation_scale=14, color=INK, linewidth=1.2))
    top.text(0.5, 39, "How each pruned model is made (same procedure for every ratio; the FP32 control skips steps "
             "2-4)", fontsize=11, fontweight="semibold", va="top")
    top.annotate("", xy=(0.5 + 3 * (width + gap) + width / 2, 5), xytext=(0.5 + 4 * (width + gap) + width / 2, 5),
                 arrowprops=dict(arrowstyle="<->", color=NOTE, linewidth=0.8))
    top.text(0.5 + 3.5 * (width + gap) + width / 2, 2.2, "evaluated on validation before and after fine-tuning",
             ha="center", fontsize=7.5, color=NOTE)

    ax = fig.add_axes([0.06, 0.08, 0.9, 0.42])
    fp32 = json.loads((RESULTS / "fp32_validation.json").read_text())
    for _, r in runs.sort_values(["kind", "target_ratio_pct", "seed"]).iterrows():
        if r["seed"] != 42:
            continue
        h = pd.read_csv(RESULTS / r["model"] / "training_history.csv")
        st, name = run_style(r)
        x = [-1] + h["epoch"].tolist()
        y = [fp32["f1_macro"] * 100] + (h["val_f1_macro"] * 100).tolist()
        ax.plot(x, y, marker="o", markersize=4.5, markerfacecolor="white", linewidth=1.8, label=name, **st)
        sel = r["selected_epoch"]
        ax.plot(sel, h.loc[h["epoch"] == sel, "val_f1_macro"].iloc[0] * 100, marker="*", markersize=13,
                color=st.get("color", INK), markeredgecolor=INK, linestyle="none", zorder=4)
    ax.axhline(fp32["f1_macro"] * 100, color=FP32_COLOR, linestyle="--", linewidth=1.2, label="FP32 baseline")
    ax.axvspan(-1.5, -0.5, color="#f2f2f2", zorder=0)
    ax.axvspan(-0.5, 0.5, color="#dcdcdc", zorder=0)
    for xpos, text in ((-1, "FP32"), (0, "pruned,\nno training"), (5.5, "fine-tuning (validation after each epoch)")):
        ax.text(xpos, 0.02, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color="#444444")
    ax.set_xticks(range(-1, 11), ["FP32"] + [str(e) for e in range(0, 11)])
    ax.set_xlim(-1.6, 10.5)
    ax.set_xlabel("Stage / fine-tuning epoch")
    ax.set_ylabel("Validation macro F1 (%)")
    ax.set_title("Measured macro F1 through the process (seed 42; star = selected epoch)", loc="left",
                 fontsize=10.5, fontweight="semibold")
    ax.legend(loc="center right", ncol=1, fontsize=8)
    save(fig, "pruning_process")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    table = consolidated(runs)
    pruned_minus_control(runs)
    plot_f1_vs_ratio(table)
    plot_f1_vs(table, "gmacs", "GMACs per image (own architecture)", "f1_vs_gmacs", "Macro F1 vs compute")
    plot_f1_vs(table, "latency_ms_1t", "CPU latency, batch 1, 1 thread (ms)", "f1_vs_latency",
               "Macro F1 vs latency (evaluate.py)")
    plot_channels(runs)
    plot_training_curves(runs)
    plot_learning_curves(runs)
    plot_process(runs)
    print((OUT / "consolidated.md").read_text())


if __name__ == "__main__":
    main()
