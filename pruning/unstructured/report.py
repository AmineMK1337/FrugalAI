"""Consolidated table and plots for unstructured pruning (validation only).

Sources:
  pruning/unstructured/results/<run>/          prune_unstructured.py: sparsity (global, per layer), sizes, metrics
                                               before fine-tuning / selected epoch / epoch 10 (GPU), pruned - control
  pruning/unstructured/results/evaluation/     evaluate.py (CPU, fused + TorchScript): performance with CIs and paired
                                               bootstrap vs FP32, storage, latency, memory, energy (estimate)
  pruning/structured/results/control_seed<s>/  the FP32 control (same fine-tuning, unpruned), reused
  evaluation/results_pruning/                  structured pruning, for the structured vs unstructured plots
Stage 1 (pruned, before fine-tuning) is evaluated here on the CPU exactly like evaluate.py does (fused + TorchScript,
same metrics, same 1,000 bootstrap resamples paired with FP32), from the saved <run>_before.pt. It is not timed: it has
the same architecture and the same zeros as the fine-tuned model.

Writes pruning/unstructured/results/report/: stage1_cpu_validation.csv, consolidated.csv, consolidated.md,
pruned_minus_control.csv and the plots (PDF + PNG).
Usage: python pruning/unstructured/report.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("evaluation_evaluate", ROOT / "evaluation" / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)  # CPU only; chdir to the fine-tuning project

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

RESULTS = HERE / "results"
EVAL = RESULTS / "evaluation"
OUT = RESULTS / "report"
CONTROL = ROOT / "pruning" / "structured" / "results"
STRUCTURED_EVAL = ROOT / "evaluation" / "results_pruning"
SPLIT = "validation"
INK, GRID, NOTE = "#1f1e1c", "#e5e5e5", "#555555"
FP32_COLOR, CONTROL_COLOR = "#2a78d6", "#9a9a9a"
SPARSE_COLOR, STRUCT_COLOR = "#3f6b26", "#c25420"
KEY = ["f1_macro", "balanced_accuracy", "melanoma_recall", "accuracy"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42, "axes.axisbelow": True, "axes.titleweight": "semibold",
                     "axes.titlesize": 11, "hatch.linewidth": 0.8})


def runs():
    found = []
    for path in sorted(RESULTS.glob("sparse*_seed*/summary.json")):
        s = json.loads(path.read_text())
        found.append(s)
    return sorted(found, key=lambda s: (s["seed"] != 42, s["seed"], s["target_sparsity_pct"]))


def stage1_cpu(summaries):
    """CPU evaluation of the pruned models before fine-tuning, paired with FP32 (same resamples)."""
    cache = OUT / f"stage1_cpu_{SPLIT}.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["run"])
    todo = [s["run"] for s in summaries if s["run"] not in set(done["run"])]
    if not todo:
        return done.set_index("run")
    config = ev.load_config(ev.CONFIG)
    _, loader, _ = ev.build_dataloaders(config)
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    probs = {"fp32": ev.logits(ev.fused_torchscript(ev.fp32_state()), images).softmax(1).numpy()}
    fp32_pred = probs["fp32"].argmax(1)
    for name in todo:
        module = torch.load(ROOT / "pruning/unstructured/models" / f"{name}_before.pt", map_location="cpu",
                            weights_only=False).eval()
        unfused = ev.logits(module, images)
        fused = ev.logits(ev.fused_module_torchscript(module), images)
        if not torch.equal(unfused.argmax(1), fused.argmax(1)):
            sys.exit(f"CHECK FAILED: {name} before fine-tuning: fused and unfused predictions differ")
        probs[name] = fused.softmax(1).numpy()
    boot = ev.bootstrap(probs, y, ev.NUM_CLASSES)
    rows = []
    for name in todo:
        point, _ = ev.scores(y, probs[name], ev.NUM_CLASSES)
        row = {"run": name, "identical_to_fp32_pct": (probs[name].argmax(1) == fp32_pred).mean() * 100}
        fp32_point, _ = ev.scores(y, probs["fp32"], ev.NUM_CLASSES)
        for m in ev.PERFORMANCE:
            diff = boot[name][m] - boot["fp32"][m]
            row.update({m: point[m], f"{m}_ci_low": ev.ci(boot[name][m])[0], f"{m}_ci_high": ev.ci(boot[name][m])[1],
                        f"{m}_change_vs_fp32": point[m] - fp32_point[m], f"{m}_change_ci_low": ev.ci(diff)[0],
                        f"{m}_change_ci_high": ev.ci(diff)[1],
                        f"{m}_paired_bootstrap_p": min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))})
        saved = pd.read_csv(RESULTS / name / "metrics.csv").set_index("stage").loc["before_finetuning"]
        if abs(point["accuracy"] - saved["accuracy"]) * len(y) > 2:
            sys.exit(f"CHECK FAILED: {name} before fine-tuning differs from prune_unstructured.py by > 2 images")
        rows.append(row)
    table = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache, index=False)
    return table.set_index("run")


def evaluated(table, name):
    path = EVAL / "comparison" / table / f"table_{SPLIT}.csv"
    if not path.exists():
        return None
    t = pd.read_csv(path).set_index("model")
    return t.loc[name] if name in t.index else None


def consolidated(summaries, stage1):
    control = {seed: pd.read_csv(CONTROL / f"control_seed{seed}" / "metrics.csv") for seed in
               sorted({s["seed"] for s in summaries})}
    rows = []
    fp32_perf, fp32_eff = evaluated("performance", "fp32"), evaluated("efficiency", "fp32")
    fp32_env = evaluated("environment", "fp32")
    if fp32_perf is not None:
        rows.append({"run": "fp32", "target_sparsity_pct": 0, "achieved_sparsity_pct": fp32_eff["sparsity_pct"],
                     "params_nonzero": fp32_eff["params_nonzero"], "params_total": fp32_eff["params"],
                     **{f"selected_{m}": fp32_perf[m] for m in KEY},
                     "file_mb": fp32_eff["size_mb"], "state_dict_dense_mb": fp32_eff["state_dict_dense_mb"],
                     "state_dict_gzip_mb": fp32_eff["state_dict_gzip_mb"], "gflops": fp32_eff["gflops"],
                     **{f"latency_ms_{t}t": fp32_eff[f"latency_median_ms_{t}t"] for t in (1, 4)},
                     **{f"latency_p95_ms_{t}t": fp32_eff[f"latency_p95_ms_{t}t"] for t in (1, 4)},
                     "peak_memory_mb_1t": fp32_eff["peak_memory_mb_1t"],
                     "energy_wh_per_1000": fp32_env["energy_wh_per_1000"]})
    for s in summaries:
        name, ctrl = s["run"], control[s["seed"]]
        ctrl_sel = ctrl[ctrl["stage"].str.startswith("selected")].iloc[0]
        ctrl_e10 = ctrl[ctrl["stage"] == "epoch10"].iloc[0]
        layers = pd.read_csv(RESULTS / name / "layers.csv")
        row = {"run": name, "seed": s["seed"], "target_sparsity_pct": s["target_sparsity_pct"],
               "achieved_sparsity_pct": s["pruned"]["global_sparsity_pct"],
               "layer_sparsity_min_pct": layers["sparsity_pct"].min(),
               "layer_sparsity_max_pct": layers["sparsity_pct"].max(),
               "params_nonzero": s["pruned"]["params_nonzero"], "params_total": s["pruned"]["params_total"],
               "selected_epoch": s["selected_epoch"]}
        b = stage1.loc[name]
        row.update({f"before_{m}": b[m] for m in KEY})
        row["before_f1_macro_change_vs_fp32"] = b["f1_macro_change_vs_fp32"]
        perf = evaluated("performance", name)
        sel = perf if perf is not None else s["selected"]
        row.update({f"selected_{m}": sel[m] for m in KEY})
        if perf is not None:
            row.update({"selected_f1_macro_change_vs_fp32": perf["f1_macro_change_vs_fp32"],
                        "selected_f1_change_ci_low": perf["f1_macro_change_ci_low"],
                        "selected_f1_change_ci_high": perf["f1_macro_change_ci_high"],
                        "selected_f1_paired_p": perf["f1_macro_paired_bootstrap_p"]})
        row.update({f"epoch10_{m}": s["epoch10"][m] for m in KEY})
        # pruned - control: both from the GPU validation of the training scripts (same source, same protocol)
        row.update({f"minus_control_selected_{m}": s["selected"][m] - ctrl_sel[m] for m in KEY})
        row.update({f"minus_control_epoch10_{m}": s["epoch10"][m] - ctrl_e10[m] for m in KEY})
        row["identical_to_fp32_pct_selected"] = s["selected"]["identical_to_fp32_pct"]
        eff, env = evaluated("efficiency", name), evaluated("environment", name)
        if eff is not None:
            row.update({"file_mb": eff["size_mb"], "state_dict_dense_mb": eff["state_dict_dense_mb"],
                        "state_dict_gzip_mb": eff["state_dict_gzip_mb"], "gflops": eff["gflops"],
                        **{f"latency_ms_{t}t": eff[f"latency_median_ms_{t}t"] for t in (1, 4)},
                        **{f"latency_p95_ms_{t}t": eff[f"latency_p95_ms_{t}t"] for t in (1, 4)},
                        **{f"speedup_{t}t": eff[f"speedup_vs_fp32_{t}t"] for t in (1, 4)},
                        "throughput_img_s_1t": eff["throughput_img_s_1t"],
                        "peak_memory_mb_1t": eff["peak_memory_mb_1t"],
                        "memory_change_pct_1t": eff["memory_change_vs_fp32_pct_1t"],
                        "energy_wh_per_1000": env["energy_wh_per_1000"],
                        "energy_change_pct": env["energy_wh_per_1000_change_vs_fp32_pct"]})
        else:
            row.update({"state_dict_dense_mb": s["pruned"]["state_dict_dense_mb"],
                        "state_dict_gzip_mb": s["pruned"]["state_dict_gzip_mb"]})
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(v, spec="{:.2f}", pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return spec.format(v * 100 if pct else v)


def markdown(table):
    lines = ["| Model | Sparsity (target / achieved) | Non-zero params | Macro F1: before → selected (epoch) / epoch 10 "
             "| Δ macro F1 vs FP32 (95% CI) | Pruned − control (selected / epoch 10) | Mel. recall | State dict "
             "dense / gzip (MB) | Latency 1t / 4t (ms) | Peak RAM 1t (MB) | Energy (Wh/1000, est.) |",
             "|" + "---|" * 11]
    for _, r in table.iterrows():
        if r["run"] == "fp32":
            lines.append(f"| FP32 | 0 / {fmt(r['achieved_sparsity_pct'])}% | {r['params_nonzero']:,.0f} | "
                         f"{fmt(r['selected_f1_macro'], pct=True)} | – | – | {fmt(r['selected_melanoma_recall'], pct=True)}"
                         f" | {fmt(r['state_dict_dense_mb'])} / {fmt(r['state_dict_gzip_mb'])} | "
                         f"{fmt(r['latency_ms_1t'])} / {fmt(r['latency_ms_4t'])} | {fmt(r['peak_memory_mb_1t'], '{:.0f}')}"
                         f" | {fmt(r['energy_wh_per_1000'])} |")
            continue
        ci = (f"{fmt(r.get('selected_f1_macro_change_vs_fp32'), '{:+.2f}', True)} "
              f"({fmt(r.get('selected_f1_change_ci_low'), '{:+.2f}', True)} to "
              f"{fmt(r.get('selected_f1_change_ci_high'), '{:+.2f}', True)})")
        lines.append(
            f"| {r['run']} | {r['target_sparsity_pct']:.0f} / {r['achieved_sparsity_pct']:.2f}% | "
            f"{r['params_nonzero']:,.0f} | {fmt(r['before_f1_macro'], pct=True)} → {fmt(r['selected_f1_macro'], pct=True)}"
            f" ({r['selected_epoch']:.0f}) / {fmt(r['epoch10_f1_macro'], pct=True)} | {ci} | "
            f"{fmt(r['minus_control_selected_f1_macro'], '{:+.2f}', True)} / "
            f"{fmt(r['minus_control_epoch10_f1_macro'], '{:+.2f}', True)} | "
            f"{fmt(r['before_melanoma_recall'], pct=True)} → {fmt(r['selected_melanoma_recall'], pct=True)} | "
            f"{fmt(r['state_dict_dense_mb'])} / {fmt(r['state_dict_gzip_mb'])} | "
            f"{fmt(r.get('latency_ms_1t'))} / {fmt(r.get('latency_ms_4t'))} | {fmt(r.get('peak_memory_mb_1t'), '{:.0f}')}"
            f" | {fmt(r.get('energy_wh_per_1000'))} |")
    lines.append("")
    lines.append("Before fine-tuning: CPU (this script); selected: CPU (evaluate.py) when evaluated, else GPU "
                 "(prune_unstructured.py); epoch 10 and pruned − control: GPU validation of the training scripts "
                 "(control = pruning/structured control_seed<seed>). Validation set, 1,532 images. Energy: CodeCarbon "
                 "estimate, no power counter.")
    return "\n".join(lines)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plots(table, stage1):
    seed42 = table[(table["run"] != "fp32") & (table["seed"] == 42)].sort_values("target_sparsity_pct")
    fp32 = table[table["run"] == "fp32"]
    if seed42.empty:
        return
    fp32_f1 = (fp32["selected_f1_macro"].iloc[0] if len(fp32)
               else json.loads((RESULTS / "fp32_validation.json").read_text())["f1_macro"]) * 100
    ctrl = pd.read_csv(CONTROL / "control_seed42" / "metrics.csv")
    ctrl_f1 = ctrl[ctrl["stage"].str.startswith("selected")].iloc[0]["f1_macro"] * 100
    x = seed42["target_sparsity_pct"].to_numpy()

    # 1. Macro F1 vs sparsity, before vs after fine-tuning
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    before = stage1.loc[seed42["run"]]
    for values, low, high, style in (
            (before["f1_macro"], before["f1_macro_ci_low"], before["f1_macro_ci_high"],
             dict(marker="o", mfc="white", ls="--", label="Pruned, before fine-tuning")),
            (seed42["selected_f1_macro"], None, None,
             dict(marker="s", ls="-", label="Pruned + fine-tuned (selected epoch)"))):
        v = np.asarray(values) * 100
        if low is not None:
            err = [v - np.asarray(low) * 100, np.asarray(high) * 100 - v]
        else:
            perf = [evaluated("performance", r) for r in seed42["run"]]
            err = ([v - np.array([p["f1_macro_ci_low"] * 100 for p in perf]),
                    np.array([p["f1_macro_ci_high"] * 100 for p in perf]) - v]
                   if all(p is not None for p in perf) else None)
        ax.errorbar(x, v, yerr=err, color=INK, capsize=3, lw=1.4, ms=6, **style)
    ax.plot(x, seed42["epoch10_f1_macro"] * 100, marker="^", ls=":", color=INK, mfc="white", lw=1,
            label="Pruned + fine-tuned, epoch 10")
    ax.axhline(fp32_f1, color=FP32_COLOR, lw=1.2, label=f"FP32 baseline ({fp32_f1:.2f})")
    ax.axhline(ctrl_f1, color=CONTROL_COLOR, lw=1.2, ls="-.", label=f"FP32 control, fine-tuned ({ctrl_f1:.2f})")
    ax.set_xticks(x, [f"{v:.0f}%" for v in x])
    ax.set_xlabel("Target sparsity (share of conv / linear weights set to zero)")
    ax.set_ylabel("Macro F1 (%), validation")
    ax.set_ylim(0, 85)
    ax.set_title("Unstructured pruning: macro F1 before and after fine-tuning (seed 42)", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.text(0.01, -0.03, "Error bars: 95% bootstrap CI (CPU evaluation). Epoch 10: GPU validation.", fontsize=8,
             color=NOTE)
    save(fig, "f1_vs_sparsity")

    # 2. Per-layer sparsity
    fig, ax = plt.subplots(figsize=(11, 4.6))
    markers, styles = ["o", "s", "^", "D"], ["-", "--", "-.", ":"]
    for i, (_, r) in enumerate(seed42.iterrows()):
        layers = pd.read_csv(RESULTS / r["run"] / "layers.csv")
        ax.plot(range(len(layers)), layers["sparsity_pct"], marker=markers[i % 4], ls=styles[i % 4], color=INK,
                mfc="white" if i % 2 else INK, ms=5, lw=1.1,
                label=f"target {r['target_sparsity_pct']:.0f}% (achieved {r['achieved_sparsity_pct']:.1f}%)")
    ax.set_xticks(range(len(layers)), [n.replace("layer", "L").replace(".downsample.0", ".ds") for n in
                                       layers["layer"]], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Weights set to zero (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Achieved sparsity per layer (global L1 ranking: large layers are pruned more)", loc="left")
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    save(fig, "per_layer_sparsity")

    # 3. Dense vs gzip size
    sized = pd.concat([fp32, seed42]) if len(fp32) else seed42
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = ["FP32" if r == "fp32" else f"{t:.0f}%" for r, t in zip(sized["run"], sized["target_sparsity_pct"])]
    xs, w = np.arange(len(sized)), 0.38
    b1 = ax.bar(xs - w / 2, sized["state_dict_dense_mb"], w, color="white", edgecolor=INK, hatch="//",
                label="Dense (torch.save state_dict)")
    b2 = ax.bar(xs + w / 2, sized["state_dict_gzip_mb"], w, color=SPARSE_COLOR, edgecolor=INK,
                label="gzip-compressed (level 9)")
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Size (MB)")
    ax.margins(y=0.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Storage: zeros do not shrink the dense file, only the compressed one", loc="left")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "size_dense_vs_gzip")

    # 4. Latency vs sparsity
    if "latency_ms_1t" in seed42 and seed42["latency_ms_1t"].notna().all() and len(fp32):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, t in zip(axes, (1, 4)):
            xs_ = np.concatenate([[0], x])
            ys = np.concatenate([[fp32[f"latency_ms_{t}t"].iloc[0]], seed42[f"latency_ms_{t}t"]])
            p95 = np.concatenate([[fp32[f"latency_p95_ms_{t}t"].iloc[0]], seed42[f"latency_p95_ms_{t}t"]])
            ax.plot(xs_, ys, marker="o", color=INK, label="median")
            ax.plot(xs_, p95, marker="s", mfc="white", ls="--", color=INK, label="p95")
            ax.set_xticks(xs_, ["FP32"] + [f"{v:.0f}%" for v in x])
            ax.set_ylim(0, max(p95) * 1.25)
            ax.set_title(f"{t} thread{'s' if t > 1 else ''}", loc="left")
            ax.set_ylabel("Latency, batch 1 (ms)")
            ax.set_xlabel("Target sparsity")
            ax.legend(fontsize=8.5, loc="lower right")
        fig.suptitle("Latency vs sparsity (CPU, fused + TorchScript, dense kernels)", x=0.01, ha="left",
                     fontweight="semibold")
        save(fig, "latency_vs_sparsity")

    # 5. Structured vs unstructured
    s_perf = STRUCTURED_EVAL / "comparison" / "performance" / f"table_{SPLIT}.csv"
    s_eff = STRUCTURED_EVAL / "comparison" / "efficiency" / f"table_{SPLIT}.csv"
    if s_perf.exists() and "latency_ms_1t" in seed42 and seed42["latency_ms_1t"].notna().all():
        sp_ = pd.read_csv(s_perf).set_index("model")
        se = pd.read_csv(s_eff).set_index("model")
        struct = [m for m in sp_.index if m.startswith("pruned") and m.endswith("_seed42")]
        struct.sort(key=lambda m: int(m[6:].split("_")[0]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, (xcol, xlabel) in zip(axes, (("latency", "Latency, 1 thread (ms)"),
                                             ("params", "Non-zero parameters (millions)"))):
            if xcol == "latency":
                sx, ux = se.loc[struct, "latency_median_ms_1t"], seed42["latency_ms_1t"]
                fx = fp32["latency_ms_1t"].iloc[0]
            else:
                sx, ux = se.loc[struct, "params"] / 1e6, seed42["params_nonzero"] / 1e6
                fx = fp32["params_nonzero"].iloc[0] / 1e6
            ax.plot(sx, sp_.loc[struct, "f1_macro"] * 100, marker="s", color=STRUCT_COLOR, ls="-",
                    label="Structured (channels removed)")
            ax.plot(ux, seed42["selected_f1_macro"] * 100, marker="o", mfc="white", color=INK, ls="--",
                    label="Unstructured (weights zeroed)")
            for xv, yv, lab in zip(sx, sp_.loc[struct, "f1_macro"] * 100, struct):
                ax.annotate(f"{lab[6:].split('_')[0]}%", (xv, yv), textcoords="offset points", xytext=(5, -11),
                            fontsize=7.5, color=STRUCT_COLOR)
            for xv, yv, t in zip(ux, seed42["selected_f1_macro"] * 100, seed42["target_sparsity_pct"]):
                ax.annotate(f"{t:.0f}%", (xv, yv), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
            ax.scatter([fx], [fp32_f1], marker="*", s=160, color=FP32_COLOR, zorder=5, label="FP32 baseline")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Macro F1 (%), validation")
            ax.legend(fontsize=8, loc="lower right")
        axes[0].set_title("Macro F1 vs latency", loc="left")
        axes[1].set_title("Macro F1 vs non-zero parameters", loc="left")
        fig.suptitle("Structured vs unstructured pruning (seed 42, fine-tuned, CPU evaluation; separate sessions)",
                     x=0.01, ha="left", fontweight="semibold")
        save(fig, "structured_vs_unstructured")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = runs()
    if not summaries:
        sys.exit("no runs in pruning/unstructured/results")
    stage1 = stage1_cpu(summaries)
    table = consolidated(summaries, stage1)
    table.to_csv(OUT / "consolidated.csv", index=False)
    md = markdown(table)
    (OUT / "consolidated.md").write_text(md + "\n")
    diff_cols = [c for c in table if c.startswith("minus_control")]
    table[table["run"] != "fp32"][["run", "seed", "target_sparsity_pct", "selected_epoch", *diff_cols]].to_csv(
        OUT / "pruned_minus_control.csv", index=False)
    plots(table, stage1)
    print(md)


if __name__ == "__main__":
    main()
