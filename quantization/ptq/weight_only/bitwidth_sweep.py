"""Bit-width sweep of weight-only quantization, 32 down to 2 bits: where does accuracy break?

Same scheme as quantize.py, with the number of bits b as a parameter: symmetric, scale = max|w| / qmax with
qmax = 2^(b-1) - 1, q = clamp(round(w / scale), -qmax, qmax), per output channel or per tensor. Only Conv2d and
Linear weights are quantized; biases and BatchNorm are kept in FP32. Weights are dequantized (w = scale * q) and
the model runs in FP32, so this measures the accuracy effect of the weight precision alone.
b = 8 must reproduce the saved INT8 results (validation_results.csv) exactly: checked. Integer bit widths: 32, 24,
20, 16 and every width from 15 down to 2 (dequantized in float64, then cast to float32, so integers above 2^24 are
exact). For reference, the weights are also cast to the 16-bit float formats FP16 and BF16 (and back to FP32).

Validation split only (the test set is never loaded). Nothing is trained or saved except the results below.
Writes results/bitwidth_sweep/: bitwidth_sweep.csv (one row per scheme and bit width; 95% bootstrap CIs) and
bitwidth_sweep.{png,pdf}.

Usage: python quantization/ptq/weight_only/bitwidth_sweep.py [--plot-only]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "resnet18_finetunning"
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from ham10000.data import build_dataloaders  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = Path("results/strong_cosine25_resnet18/resnet18_baseline.pth")
OUT = HERE / "results" / "bitwidth_sweep"
BITS = [32, 24, 20, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
FLOATS = {"fp16": torch.float16, "bf16": torch.bfloat16}
SCHEMES = {"per_channel": True, "per_tensor": False}
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0


def quantized_state(state, names, bits, per_channel):
    """FP32 state_dict with every Conv2d/Linear weight replaced by its b-bit dequantized version."""
    qmax = 2 ** (bits - 1) - 1
    out = dict(state)
    for name in names:
        w = state[name]
        max_abs = w.abs().flatten(1).amax(1) if per_channel else w.abs().max()
        scale = (max_abs / qmax).to(torch.float32)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        s = scale.view(-1, *[1] * (w.dim() - 1)) if scale.dim() else scale
        q = (w.double() / s.double()).round().clamp(-qmax, qmax)
        out[name] = (s.double() * q).to(torch.float32)
    return out


@torch.inference_mode()
def predict(model, batches):
    return torch.cat([model(images) for images, _ in batches])


def scores(y, pred, mel):
    recall = np.array([(pred[y == c] == c).mean() for c in range(7)])
    return {"accuracy": (pred == y).mean(), "balanced_accuracy": recall.mean(),
            "f1_macro": f1_score(y, pred, average="macro", labels=list(range(7)), zero_division=0),
            "melanoma_recall": recall[mel]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    config = load_config(CONFIG)
    _, val_loader, class_to_idx = build_dataloaders(config)
    mel = class_to_idx["mel"]
    batches = list(val_loader)
    y = torch.cat([labels for _, labels in batches]).numpy()

    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = resnet18(weights=None, num_classes=7)
    model.load_state_dict(state, strict=True)
    model.eval()
    names = [f"{n}.weight" for n, m in model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    fp32_logits = predict(model, batches)
    fp32_pred = fp32_logits.argmax(1).numpy()
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(y), size=(BOOTSTRAP, len(y)))

    def row(scheme, bits, logits, weight_error):
        pred = logits.argmax(1).numpy()
        point = scores(y, pred, mel)
        boot = [scores(y[i], pred[i], mel) for i in resamples]
        r = {"scheme": scheme, "bits": bits,
             "levels": 2 ** bits - 1 if scheme in SCHEMES else None}  # symmetric integer levels
        for k, v in point.items():
            r[k] = v
            r[f"{k}_ci_low"], r[f"{k}_ci_high"] = np.percentile([b[k] for b in boot], [2.5, 97.5])
        r["identical_to_fp32_pct"] = (pred == fp32_pred).mean() * 100
        r["softmax_mad"] = (logits.softmax(1) - fp32_logits.softmax(1)).abs().mean().item()
        r["weight_relative_rms_error_pct"] = weight_error
        print(f"{scheme:12s} {bits:2d} bits: acc {r['accuracy']:.4f}  F1 {r['f1_macro']:.4f}  mel {r['melanoma_recall']:.4f}"
              f"  identical {r['identical_to_fp32_pct']:.2f}%  weight error {weight_error:.2f}%", flush=True)
        return r

    rows = [row("fp32", 32, fp32_logits, 0.0)]
    for scheme, per_channel in SCHEMES.items():
        for bits in BITS:
            qs = quantized_state(state, names, bits, per_channel)
            num = sum(((qs[n] - state[n]) ** 2).sum() for n in names)
            den = sum((state[n] ** 2).sum() for n in names)
            model.load_state_dict(qs, strict=True)
            rows.append(row(scheme, bits, predict(model, batches), float((num / den).sqrt() * 100)))
    for scheme, dtype in FLOATS.items():
        qs = {k: (v.to(dtype).float() if k in names else v) for k, v in state.items()}
        num = sum(((qs[n] - state[n]) ** 2).sum() for n in names)
        den = sum((state[n] ** 2).sum() for n in names)
        model.load_state_dict(qs, strict=True)
        rows.append(row(scheme, 16, predict(model, batches), float((num / den).sqrt() * 100)))
    table = pd.DataFrame(rows)

    # b = 8 must equal the saved weight-only INT8 results
    saved = pd.read_csv(HERE / "results" / "validation_results.csv").set_index("model")
    for scheme in SCHEMES:
        r = table[(table["scheme"] == scheme) & (table["bits"] == 8)].iloc[0]
        for mine, theirs, tol in (("accuracy", "accuracy", 1e-9), ("f1_macro", "macro_f1", 1e-9),
                                  ("melanoma_recall", "melanoma_recall", 1e-9),
                                  ("identical_to_fp32_pct", "identical_to_fp32_pct", 1e-4)):  # saved in float32
            if abs(r[mine] - saved.loc[scheme, theirs]) > tol:
                sys.exit(f"CHECK FAILED: {scheme} 8 bits {mine} {r[mine]} != saved {saved.loc[scheme, theirs]}")
    print("8 bits reproduces the saved weight-only INT8 results", flush=True)
    table.to_csv(OUT / "bitwidth_sweep.csv", index=False)
    plot(table)


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#1f1e1c", "#5f6368", "#e6e6e6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
                         "axes.edgecolor": "#b0b0b0", "xtick.color": ink, "ytick.color": muted})
    fp32 = table[table["scheme"] == "fp32"].iloc[0]
    x_of = {b: i for i, b in enumerate(BITS)}  # equally spaced slots, 32 bits on the left
    float_x = -1.6  # FP16 / BF16 get their own slot left of the integer sweep
    # colours checked with the dataviz palette validator; markers and line style also tell the schemes apart
    style = {"per_channel": dict(color="#f47230", marker="o", linestyle="-", label="INT, per-channel scale"),
             "per_tensor": dict(color="#00a383", marker="s", linestyle="--", markerfacecolor="white",
                                label="INT, per-tensor scale")}
    float_style = {"fp16": dict(color=ink, marker="^", label="FP16 (16-bit float)"),
                   "bf16": dict(color="#6b5bb8", marker="D", markerfacecolor="white", label="BF16 (16-bit float)")}
    panels = [("f1_macro", "Macro F1 (main metric)", "%", True),
              ("melanoma_recall", "Melanoma recall", "%", True),
              ("accuracy", "Accuracy (misleading at 2-4 bits: the model predicts the majority class, nv, "
                           "for almost every image)", "%", True),
              ("identical_to_fp32_pct", "Predictions identical to FP32", "%", False)]
    # the weight rounding error is still saved in bitwidth_sweep.csv (weight_relative_rms_error_pct)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 16), sharex=True)
    fig.subplots_adjust(top=0.885, bottom=0.08, hspace=0.55, left=0.08, right=0.98)
    for ax, (col, title, unit, with_ci) in zip(axes, panels):
        scale = 100 if with_ci else 1
        ax.axvspan(x_of[8] - 0.45, x_of[8] + 0.45, color="#ececec", zorder=0)
        for scheme, st in style.items():
            t = table[table["scheme"] == scheme].copy()
            t["x"] = t["bits"].map(x_of)
            t = t.sort_values("x")
            if with_ci:
                ax.fill_between(t["x"], t[f"{col}_ci_low"] * scale, t[f"{col}_ci_high"] * scale,
                                color=st["color"], alpha=0.13, linewidth=0, zorder=1)
            ax.plot(t["x"], t[col] * scale, linewidth=2, markersize=6, zorder=3, **st)
        for k, (scheme, st) in enumerate(float_style.items()):
            r = table[table["scheme"] == scheme].iloc[0]
            ax.plot(float_x + (k - 0.5) * 0.5, r[col] * scale, linestyle="none", markersize=8, zorder=4, **st)
        ax.axvline(-0.8, color="#b0b0b0", linewidth=1, linestyle=(0, (4, 3)))
        ax.axhline(fp32[col] * scale, color="#257de0", linewidth=1.4, linestyle=(0, (1, 1.5)), zorder=2,
                   label="FP32 baseline")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=ink, pad=6)
        ax.set_ylabel(unit, color=muted)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelbottom=True)  # bit labels under every panel (each can stand alone on a slide)
        ax.set_xlabel("Bits per weight", color=muted, fontsize=9, labelpad=2)
    top = axes[0]
    top.text(x_of[8], 1.02, "INT8\n(saved models)", transform=top.get_xaxis_transform(), ha="center", va="bottom",
             fontsize=8.5, color=muted)
    axes[-1].set_xticks([float_x] + list(range(len(BITS))), ["FP16\nBF16"] + [str(b) for b in BITS])
    axes[-1].set_xlim(float_x - 0.6, len(BITS) - 0.4)
    axes[-1].set_xlabel("Bits per weight  (integer formats, equally spaced; fewer bits to the right)", color=ink)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.075, 0.955), ncol=5, fontsize=9.5)
    fig.text(0.08, 0.975, "Weight-only quantization: performance holds down to 7 bits and collapses at 4 bits",
             fontsize=14, fontweight="bold", color=ink, va="bottom")
    fig.text(0.08, 0.02, "Validation split (1,532 images). Shaded bands: 95% bootstrap CI (1,000 resamples). "
             "Grey column: 8 bits, the saved INT8 models.\nOnly the weights are quantized; activations and "
             "computation stay FP32. nv is 66% of the validation images, so predicting only nv already gives "
             "66% accuracy.", fontsize=8.5, color=muted)
    fig.savefig(OUT / "bitwidth_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "bitwidth_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw the figure from the saved CSV, without rerunning the sweep
        plot(pd.read_csv(OUT / "bitwidth_sweep.csv"))
    else:
        main()
