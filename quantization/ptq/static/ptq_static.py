"""Static INT8 post-training quantization of the fine-tuned ResNet-18 (eager mode, x86 backend).

Weights INT8 per channel, activations INT8 (reduce_range=True), calibrated on 1,000 TRAINING images
picked by class-stratified sampling. Evaluated on the VALIDATION split only; the test split is never loaded.
Nothing in resnet18_finetunning/ is modified: the checkpoint and split are hashed before and after.

Which 1,000 images are picked moves INT8 accuracy by a few tenths of a point (see benchmarking/), so the
model is calibrated with three samples (seeds 42, 1, 2) and INT8 is reported as their mean, min and max.
The seed-42 model is the one saved. FP32 latency is measured with BatchNorm fused and TorchScript, like
the INT8 file, so the speedup reflects INT8 only.

Writes: resnet18_static_int8.pt (TorchScript, seed 42) next to this script, and in results/:
calibration_images.csv, validation_results.csv (one row per model/seed), validation_summary.csv
(FP32 vs INT8 mean/min/max), confusion_matrix_<model>.csv, comparison.png/.pdf.

Usage: python quantization/ptq/static/ptq_static.py
"""
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "resnet18_finetunning"
os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project
sys.path.insert(0, str(PROJECT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.ao.quantization import convert, get_default_qconfig, prepare  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision.models.quantization import resnet18  # noqa: E402

from ham10000.data import Ham10000Dataset, build_dataloaders, eval_transform, lesion_split  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = PROJECT / "results/strong_cosine25_resnet18/resnet18_baseline.pth"
INT8_PATH = HERE / "resnet18_static_int8.pt"
OUT = HERE / "results"
N_CALIBRATION, SEEDS, SAVED_SEED = 1000, (42, 1, 2), 42
THREADS, WARMUP, RUNS = 4, 20, 100


class CheckFailed(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fp32(num_classes):
    model = resnet18(weights=None, quantize=False, num_classes=num_classes)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def calibration_subset(config, seed):
    """1,000 training images, stratified by class, eval preprocessing (no augmentation)."""
    dataset = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=eval_transform(config))
    train_idx, _, _ = lesion_split(dataset.data, config)
    train = dataset.data.loc[train_idx]
    chosen, _ = train_test_split(train.index.to_numpy(), train_size=N_CALIBRATION, stratify=train["dx"],
                                 random_state=seed)
    chosen = sorted(chosen)
    picked = dataset.data.loc[chosen, ["image_id", "lesion_id", "dx"]]
    partition = pd.read_csv(config.splits_csv).set_index("image_id")["partition"]
    if not (picked["image_id"].map(partition) == "train").all():
        raise CheckFailed("a calibration image is not in the training split")
    counts = pd.DataFrame({"calibration": picked["dx"].value_counts(), "train": train["dx"].value_counts()})
    counts["calibration_pct"] = counts["calibration"] / counts["calibration"].sum() * 100
    counts["train_pct"] = counts["train"] / counts["train"].sum() * 100
    if len(picked) != N_CALIBRATION or counts["calibration"].isna().any() or len(counts) != 7:
        raise CheckFailed(f"calibration set is not 1,000 images covering all 7 classes:\n{counts}")
    return Subset(dataset, chosen), picked, counts


def quantize(fp32, calibration_loader):
    engine = "x86" if "x86" in torch.backends.quantized.supported_engines else "onednn"
    torch.backends.quantized.engine = engine
    model = load_fp32(fp32.fc.out_features)  # a fresh copy; the FP32 model is left untouched
    model.fuse_model()
    model.qconfig = get_default_qconfig(engine)
    prepare(model, inplace=True)
    with torch.inference_mode():
        for images, _ in calibration_loader:
            model(images)
    convert(model, inplace=True)
    return model.eval(), engine


def check_quantized(model):
    """Every conv and the fc must be INT8 with per-channel weights; no float conv/linear left."""
    quantized = [(n, m) for n, m in model.named_modules() if hasattr(m, "weight") and callable(m.weight)]
    floats = [n for n, m in model.named_modules() if type(m) in (torch.nn.Conv2d, torch.nn.Linear)]
    if floats or len(quantized) != 21:
        raise CheckFailed(f"{len(quantized)} quantized layers, float layers left: {floats}")
    for name, module in quantized:
        w = module.weight()
        if w.dtype != torch.qint8 or w.qscheme() not in (torch.per_channel_affine, torch.per_channel_symmetric):
            raise CheckFailed(f"{name}: weight {w.dtype}, {w.qscheme()}")
    return len(quantized)


@torch.inference_mode()
def predict(model, batches):
    return torch.cat([model(images) for images, _ in batches])


def metrics(name, logits, y, reference_pred, classes):
    probs = logits.softmax(1).numpy()
    pred = probs.argmax(1)
    ids = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=ids, zero_division=0)
    cm = confusion_matrix(y, pred, labels=ids)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
        .to_csv(OUT / f"confusion_matrix_{name}.csv")
    fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(1) - fp
    gained = int(((reference_pred != y) & (pred == y)).sum())
    lost = int(((reference_pred == y) & (pred != y)).sum())
    return {
        "model": name, "accuracy": accuracy_score(y, pred), "precision_macro": precision.mean(),
        "recall_macro": recall.mean(), "f1_macro": f1.mean(), "specificity_macro": (tn / (tn + fp)).mean(),
        "auc_macro": np.mean([roc_auc_score(y == i, probs[:, i]) for i in ids]),
        "melanoma_recall": recall[classes.index("mel")],
        "identical_to_fp32_pct": (pred == reference_pred).mean() * 100,
        "wrong_to_right_vs_fp32": gained, "right_to_wrong_vs_fp32": lost,
        "mcnemar_p_vs_fp32": binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0,
    }


@torch.inference_mode()
def latency_ms(model, image):
    for _ in range(WARMUP):
        model(image)
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        model(image)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


SCORES = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro", "auc_macro",
          "melanoma_recall"]


def plot(summary):
    """FP32 vs INT8. Scores: INT8 bar = mean of the calibration samples, whiskers = min-max.
    Size and latency: the saved (seed-42) INT8 model."""
    surface, ink, muted, grid = "#fcfcfb", "#1f1e1c", "#52514e", "#e6e5e0"
    fp32_colour, int8_colour = "#2a78d6", "#eda100"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": muted, "axes.edgecolor": grid,
        "axes.facecolor": surface, "figure.facecolor": surface, "savefig.facecolor": surface,
        "xtick.color": muted, "ytick.color": muted, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
        "grid.color": grid, "axes.axisbelow": True, "pdf.fonttype": 42, "legend.frameon": False,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
    })
    s = summary.set_index("model")
    panels = [("File size (MB)", ["size_mb"], ["Size"], 1, "INT8_saved_seed42"),
              ("Latency (ms)", ["latency_median_ms"], ["Median, batch 1,\n4 CPU threads"], 1, "INT8_saved_seed42"),
              ("Validation scores (%)", ["accuracy", "recall_macro", "f1_macro", "auc_macro", "melanoma_recall"],
               ["Accuracy", "Macro\nrecall", "Macro F1", "Macro\nAUC", "Melanoma\nrecall"], 100, "INT8_mean")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1, 3.2]})
    for ax, (title, columns, ticks, factor, int8_row) in zip(axes, panels):
        x, width = np.arange(len(columns)), 0.38
        fp32 = s.loc["FP32", columns].to_numpy(dtype=float) * factor
        int8 = s.loc[int8_row, columns].to_numpy(dtype=float) * factor
        bars = ax.bar(x - width / 2, fp32, width * 0.92, color=fp32_colour, label="FP32 (original)")
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=ink)
        if int8_row == "INT8_mean":
            low = s.loc["INT8_min", columns].to_numpy(dtype=float) * factor
            high = s.loc["INT8_max", columns].to_numpy(dtype=float) * factor
            bars = ax.bar(x + width / 2, int8, width * 0.92, color=int8_colour,
                          yerr=[int8 - low, high - int8], error_kw={"ecolor": ink, "elinewidth": 1, "capsize": 3},
                          label=f"INT8 static PTQ (scores: mean of {len(SEEDS)} calibration samples, "
                                "whiskers min-max; size and latency: saved model)")
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in int8], padding=6, fontsize=8, color=ink)
        else:
            bars = ax.bar(x + width / 2, int8, width * 0.92, color=int8_colour)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=ink)
        ax.set_xticks(x, ticks)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[2].set_ylim(0, 100)
    handles = axes[0].get_legend_handles_labels()[0] + axes[2].get_legend_handles_labels()[0][1:]
    names = axes[0].get_legend_handles_labels()[1] + axes[2].get_legend_handles_labels()[1][1:]
    fig.legend(handles, names, loc="upper left", ncol=2, bbox_to_anchor=(0.005, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    hashes = (sha256(CHECKPOINT), sha256(config.splits_csv))
    print(f"torch {torch.__version__}; FP32 checkpoint sha256 {hashes[0]}", flush=True)
    fp32 = load_fp32(config.num_classes)

    # Validation only: build_dataloaders returns no test loader. Decode once; every model sees identical tensors.
    _, val_loader, class_to_idx = build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    batches = list(val_loader)
    y = torch.cat([labels for _, labels in batches]).numpy()
    fp32_logits = predict(fp32, batches)
    reference = fp32_logits.argmax(1).numpy()
    rows = [{"seed": None, **metrics("FP32", fp32_logits, y, reference, classes)}]

    picked_all = []
    for seed in SEEDS:
        calibration, picked, counts = calibration_subset(config, seed)
        picked_all.append(picked.assign(seed=seed))
        print(f"\n== Calibration seed {seed}: {len(picked)} training images, {picked['lesion_id'].nunique()} "
              f"lesions ==\n{counts.round(2).to_string()}", flush=True)
        loader = DataLoader(calibration, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        int8, engine = quantize(fp32, loader)
        n_layers = check_quantized(int8)
        print(f"Engine {engine}; {n_layers} conv/linear layers are INT8 with per-channel weights", flush=True)
        int8_logits = predict(int8, batches)
        if seed == SAVED_SEED:
            torch.jit.save(torch.jit.script(int8), INT8_PATH)
            int8_saved = torch.jit.load(INT8_PATH, map_location="cpu").eval()
            if not torch.equal(predict(int8_saved, batches), int8_logits):
                raise CheckFailed("the reloaded TorchScript file gives different outputs from the converted model")
        rows.append({"seed": seed, **metrics(f"INT8_seed{seed}", int8_logits, y, reference, classes)})
    pd.concat(picked_all).to_csv(OUT / "calibration_images.csv", index=False)

    # Latency: FP32 with BatchNorm fused + TorchScript (its outputs are checked against eager FP32), vs saved INT8
    fp32_fast = load_fp32(config.num_classes)
    fp32_fast.fuse_model()
    fp32_fast = torch.jit.script(fp32_fast.eval())
    if (predict(fp32_fast, batches[:2]) - predict(fp32, batches[:2])).abs().max() > 1e-3:
        raise CheckFailed("fused FP32 does not reproduce eager FP32")
    torch.set_num_threads(THREADS)
    image = batches[0][0][:1].clone()
    rows[0]["size_mb"] = CHECKPOINT.stat().st_size / 1e6
    rows[0]["latency_median_ms"], rows[0]["latency_p95_ms"] = latency_ms(fp32_fast, image)
    rows[0]["latency_eager_median_ms"] = latency_ms(fp32, image)[0]
    saved_row = next(r for r in rows if r["seed"] == SAVED_SEED)
    saved_row["size_mb"] = INT8_PATH.stat().st_size / 1e6
    saved_row["latency_median_ms"], saved_row["latency_p95_ms"] = latency_ms(int8_saved, image)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "validation_results.csv", index=False)

    int8_runs = table[table["model"] != "FP32"]
    fp32_row = table[table["model"] == "FP32"].iloc[0]
    columns = SCORES + ["identical_to_fp32_pct"]
    summary = pd.DataFrame([
        {"model": "FP32", **fp32_row[columns + ["size_mb", "latency_median_ms"]].to_dict()},
        {"model": "INT8_mean", **int8_runs[columns].mean().to_dict()},
        {"model": "INT8_min", **int8_runs[columns].min().to_dict()},
        {"model": "INT8_max", **int8_runs[columns].max().to_dict()},
        {"model": "INT8_saved_seed42", **int8_runs[int8_runs["seed"] == SAVED_SEED].iloc[0][
            columns + ["size_mb", "latency_median_ms"]].to_dict()},
    ])
    summary.to_csv(OUT / "validation_summary.csv", index=False)
    plot(summary)

    if (sha256(CHECKPOINT), sha256(config.splits_csv)) != hashes:
        raise CheckFailed("the FP32 checkpoint or the split file changed during the run")
    print(f"\nCheckpoint and split unchanged. Validation images: {len(y)}")
    pd.set_option("display.width", 200)
    print(f"\n== Per run (validation) ==\n{table.set_index('model').T.round(4).to_string()}")
    print(f"\n== Summary (validation) ==\n{summary.set_index('model').T.round(4).to_string()}")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
