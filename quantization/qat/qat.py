"""Quantization-aware training (QAT) of the fine-tuned ResNet-18: INT8 weights and activations, same scheme as
static PTQ (x86 engine, per-channel symmetric INT8 weights, per-tensor activations with reduce_range=True).

Data roles: train split = QAT fine-tuning and observer initialization; validation split = choosing the epoch.
The test split is never loaded. Nothing in resnet18_finetunning/ or quantization/ptq/ is modified: the FP32
checkpoint, the split, the PTQ model and PTQ results are hashed before and after the run.

Per seed (1, 2, 42; 42 is the saved model):
 1. torchvision.models.quantization.resnet18(quantize=False), FP32 weights (strict); train(); fuse_model(is_qat=True);
    qconfig = get_default_qat_qconfig("x86"); prepare_qat.
 2. Observer initialization on the same 1,000 stratified training images as static PTQ for that seed (eval
    preprocessing, no gradient, BatchNorm in eval mode, fake quantization off so observers see float activations).
    The x86 QAT qconfig observes activations with a moving-average min/max, while static PTQ used a
    HistogramObserver, so the literal pass alone does not reproduce PTQ: its epoch-0 INT8 result is recorded
    ("epoch0_moving_average_init"). The activation ranges of the PTQ model calibrated on those same images, and the
    per-channel weight ranges PTQ uses, are then written into the QAT observers, so QAT starts from the static PTQ
    model (checked: epoch-0 predictions identical to static PTQ and metrics equal to its saved results).
 3. 10 epochs on the GPU: class-weighted cross-entropy (training-split weights), strong augmentation, batch 32,
    Adam lr 1e-5, cosine schedule over 10 epochs (stepped per epoch, as in the FP32 run). BatchNorm statistics are
    frozen from epoch 7 on (after epoch 6), observers disabled from epoch 9 on (after epoch 8).
 4. Before training and after every epoch: copy to CPU, convert to INT8, evaluate on validation. The epoch with the
    best INT8 validation macro F1 (epochs 0-10) is kept.
Control (--control): the same 10 epochs from the FP32 model without fake quantization (BatchNorm frozen after
epoch 6 as well), FP32 validation metrics per epoch.

Writes, in quantization/qat/: resnet18_qat_int8.pt (TorchScript, seed 42, best epoch) and in results/:
settings.json, training_history.json (+ .csv), validation_results.csv, validation_summary.csv,
control_fp32_history.json (+ .csv), confusion_matrix_<model>.csv, plots/training_curves.{pdf,png}.

Usage: python quantization/qat/qat.py [--seeds 42 1 2] [--epochs 10] [--control]
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / "resnet18_finetunning"
STATIC = ROOT / "quantization" / "ptq" / "static"
sys.path.insert(0, str(PROJECT / "src"))
import os  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*(reduce_range will be deprecated|migrate to use torchao|TypedStorage).*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.ao.nn.intrinsic.qat as nniqat  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.ao.quantization import (convert, disable_fake_quant, disable_observer, enable_fake_quant,  # noqa: E402
                                   enable_observer, get_default_qat_qconfig, get_default_qconfig, prepare,
                                   prepare_qat)
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision.models.quantization import resnet18  # noqa: E402

from ham10000.data import (Ham10000Dataset, build_dataloaders, compute_class_weights, eval_transform,  # noqa: E402
                           lesion_split)
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = PROJECT / "results/strong_cosine25_resnet18/resnet18_baseline.pth"
QAT_INT8 = HERE / "resnet18_qat_int8.pt"
OUT = HERE / "results"
ENGINE = "x86"
N_CALIBRATION, SAVED_SEED = 1000, 42
LR, EPOCHS, FREEZE_BN_AFTER, DISABLE_OBSERVER_AFTER = 1e-5, 10, 6, 8
EVAL_THREADS = 16
SCORES = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro", "auc_macro",
          "melanoma_recall"]
PROTECTED = [CHECKPOINT, PROJECT / "splits/ham10000_splits.csv", PROJECT / CONFIG, STATIC / "resnet18_static_int8.pt",
             STATIC / "results/validation_results.csv", STATIC / "results/calibration_images.csv"]


class CheckFailed(RuntimeError):
    pass


def log(message):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {message}", flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fp32(num_classes):
    model = resnet18(weights=None, quantize=False, num_classes=num_classes)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    return model


# ── Calibration images: the same selection as quantization/ptq/static/ptq_static.py ─────────────

def calibration_subset(config, seed):
    dataset = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=eval_transform(config))
    train_idx, _, _ = lesion_split(dataset.data, config)
    train = dataset.data.loc[train_idx]
    chosen, _ = train_test_split(train.index.to_numpy(), train_size=N_CALIBRATION, stratify=train["dx"],
                                 random_state=seed)
    chosen = sorted(chosen)
    picked = dataset.data.loc[chosen, "image_id"].tolist()
    saved = pd.read_csv(STATIC / "results/calibration_images.csv")
    if picked != saved.loc[saved["seed"] == seed, "image_id"].tolist():
        raise CheckFailed(f"seed {seed}: calibration images differ from the static PTQ ones")
    partition = pd.read_csv(config.splits_csv).set_index("image_id")["partition"]
    if not (pd.Series(picked).map(partition) == "train").all():
        raise CheckFailed("a calibration image is not in the training split")
    return DataLoader(Subset(dataset, chosen), batch_size=config.batch_size, shuffle=False,
                      num_workers=config.num_workers)


# ── Models ───────────────────────────────────────────────────────────────────

def static_ptq_prepared(num_classes, calibration):
    """Exactly the static PTQ procedure up to convert: fuse, x86 qconfig, prepare, calibrate (CPU)."""
    model = load_fp32(num_classes).eval()
    model.fuse_model()
    model.qconfig = get_default_qconfig(ENGINE)
    prepare(model, inplace=True)
    with torch.inference_mode():
        for images, _ in calibration:
            model(images)
    return model


def qat_prepared(num_classes):
    model = load_fp32(num_classes)
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = get_default_qat_qconfig(ENGINE)
    prepare_qat(model, inplace=True)
    return model


def activation_fake_quants(model):
    """Output observers of the fused modules (not weight fake-quants, not the observer inside a FakeQuantize)."""
    return {name: m for name, m in model.named_modules() if name.endswith("activation_post_process")
            and not name.endswith("activation_post_process.activation_post_process") and "weight_fake_quant" not in name}


@torch.no_grad()
def init_observers(model, calibration, device):
    """Literal pass: the 1,000 calibration images through the QAT model, eval mode (BatchNorm running stats are not
    touched), observers on, fake quantization off (observers see float activations, as in PTQ calibration)."""
    model.eval()
    model.apply(enable_observer)
    model.apply(disable_fake_quant)
    for images, _ in calibration:
        model(images.to(device))
    model.apply(enable_fake_quant)


def cpu_qparams(fake_quant):
    """qparams as convert() computes them: on the CPU copy of the model (GPU division can differ by 1 ulp)."""
    return copy.deepcopy(fake_quant.activation_post_process).cpu().calculate_qparams()


@torch.no_grad()
def copy_ptq_activation_ranges(qat_model, ptq_model):
    """Write the activation ranges of the calibrated static PTQ model into the QAT activation observers. The range is
    the one HistogramObserver.calculate_qparams itself passes to _calculate_qparams (its _non_linear_param_search),
    so the QAT observer, which uses the same _calculate_qparams, gives exactly the PTQ scale and zero point."""
    ptq, qat = activation_fake_quants(ptq_model), activation_fake_quants(qat_model)
    if set(ptq) != set(qat) or len(ptq) == 0:
        raise CheckFailed(f"activation observers differ: {sorted(set(ptq) ^ set(qat))}")
    for name, fq in qat.items():
        scale, zp = ptq[name].calculate_qparams()
        lo, hi = ptq[name]._non_linear_param_search()
        obs = fq.activation_post_process
        obs.min_val.copy_(lo.reshape(obs.min_val.shape).to(obs.min_val.device))
        obs.max_val.copy_(hi.reshape(obs.max_val.shape).to(obs.max_val.device))
        new_scale, new_zp = cpu_qparams(fq)
        if not (torch.equal(new_scale, scale) and torch.equal(new_zp.int(), zp.int())):
            raise CheckFailed(f"{name}: qparams not reproduced ({new_scale}, {new_zp}) vs ({scale}, {zp})")
        fq.scale.copy_(new_scale.reshape(fq.scale.shape).to(fq.scale.device))
        fq.zero_point.copy_(new_zp.reshape(fq.zero_point.shape).to(fq.zero_point.device))
    return len(qat)


@torch.no_grad()
def copy_ptq_weight_ranges(qat_model, ptq_model):
    """Write the per-channel weight ranges static PTQ uses (its weight observer run on the CPU-fused weights) into the
    QAT weight observers. Without this, the QAT observers see BatchNorm folded on the GPU, whose float rounding moves
    the scales by ~1e-7 relative and flips ~0.01% of INT8 weights by one level."""
    qat = {name: m for name, m in qat_model.named_modules() if hasattr(m, "weight_fake_quant")}
    count = 0
    for name, module in ptq_model.named_modules():
        if name not in qat:
            continue
        conv = module[0] if isinstance(module, torch.nn.Sequential) else module
        observer = module.qconfig.weight()
        observer(conv.weight)
        scale, zp = observer.calculate_qparams()
        fq = qat[name].weight_fake_quant
        fq.activation_post_process.min_val.copy_(observer.min_val.reshape(fq.activation_post_process.min_val.shape))
        fq.activation_post_process.max_val.copy_(observer.max_val.reshape(fq.activation_post_process.max_val.shape))
        new_scale, new_zp = cpu_qparams(fq)
        if not (torch.equal(new_scale, scale) and torch.equal(new_zp, zp)):
            raise CheckFailed(f"{name}: weight qparams not reproduced: scale max rel diff "
                              f"{((new_scale.cpu() - scale).abs() / scale).max():.2e}, zp {new_zp.dtype}/{zp.dtype} "
                              f"equal {torch.equal(new_zp.cpu().long(), zp.long())}")
        fq.scale.resize_(new_scale.shape).copy_(new_scale.to(fq.scale.device))
        fq.zero_point.resize_(new_zp.shape).copy_(new_zp.to(fq.zero_point.device))
        count += 1
    if count != len(qat):
        raise CheckFailed(f"{count} weight observers copied, {len(qat)} in the QAT model")
    return count


def to_int8(qat_model):
    """A CPU INT8 copy of the QAT model; the QAT model itself is not changed."""
    model = copy.deepcopy(qat_model).cpu().eval()
    convert(model, inplace=True)
    return model.eval()


def check_quantized(model):
    """Every conv and the fc must be INT8 with per-channel weights; no float conv/linear left (as in PTQ)."""
    quantized = [(n, m) for n, m in model.named_modules() if hasattr(m, "weight") and callable(m.weight)]
    floats = [n for n, m in model.named_modules() if type(m) in (torch.nn.Conv2d, torch.nn.Linear)]
    if floats or len(quantized) != 21:
        raise CheckFailed(f"{len(quantized)} quantized layers, float layers left: {floats}")
    for name, module in quantized:
        w = module.weight()
        if w.dtype != torch.qint8 or w.qscheme() not in (torch.per_channel_affine, torch.per_channel_symmetric):
            raise CheckFailed(f"{name}: weight {w.dtype}, {w.qscheme()}")


# ── Metrics (same definitions as static PTQ) ────────────────────────────────

@torch.inference_mode()
def predict(model, batches, device="cpu"):
    return torch.cat([model(images.to(device)).float().cpu() for images, _ in batches])


def metrics(name, logits, y, reference_pred, classes, save_confusion=False):
    probs = logits.softmax(1).numpy()
    pred = probs.argmax(1)
    ids = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=ids, zero_division=0)
    cm = confusion_matrix(y, pred, labels=ids)
    if save_confusion:
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


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / n


def freeze_bn_fp32(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def run(seed, config, val, fp32_ref, classes, device, control=False):
    """One QAT run (or the FP32 control) for one seed. Returns (history rows, extra info, best model)."""
    y, val_batches = val
    set_seed(seed)
    train_loader, _, _ = build_dataloaders(config)
    criterion = torch.nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, config.num_classes, device))
    info = {"seed": seed}

    if control:
        model = load_fp32(config.num_classes).to(device)
        evaluate = lambda: predict(model.eval(), val_batches, device)  # noqa: E731
    else:
        calibration = calibration_subset(config, seed)
        ptq = static_ptq_prepared(config.num_classes, calibration)
        ptq_int8 = convert(copy.deepcopy(ptq), inplace=False).eval()
        ptq_logits = predict(ptq_int8, val_batches)
        model = qat_prepared(config.num_classes).to(device)
        init_observers(model, calibration, device)
        literal = metrics("literal", predict(to_int8(model), val_batches), y, fp32_ref, classes)
        info["epoch0_moving_average_init"] = {k: literal[k] for k in SCORES + ["identical_to_fp32_pct"]}
        log(f"seed {seed}: literal moving-average observer init -> INT8 val macro F1 {literal['f1_macro']:.4f}, "
            f"acc {literal['accuracy']:.4f}")
        info["activation_observers_copied"] = copy_ptq_activation_ranges(model, ptq)
        info["weight_observers_copied"] = copy_ptq_weight_ranges(model, ptq)
        int8 = to_int8(model)
        check_quantized(int8)
        epoch0_logits = predict(int8, val_batches)
        same = float((epoch0_logits.argmax(1) == ptq_logits.argmax(1)).float().mean() * 100)
        saved = pd.read_csv(STATIC / "results/validation_results.csv")
        saved = saved[saved["seed"] == seed].iloc[0]
        e0 = metrics("e0", epoch0_logits, y, fp32_ref, classes)
        info["epoch0_vs_static_ptq"] = {
            "identical_predictions_pct": same,
            "max_abs_logit_difference": float((epoch0_logits - ptq_logits).abs().max()),
            **{f"{k}_epoch0": e0[k] for k in ("accuracy", "f1_macro", "melanoma_recall")},
            **{f"{k}_static_ptq_saved": float(saved[k]) for k in ("accuracy", "f1_macro", "melanoma_recall")}}
        log(f"seed {seed}: epoch 0 vs static PTQ: {same:.2f}% identical predictions, max logit diff "
            f"{info['epoch0_vs_static_ptq']['max_abs_logit_difference']:.3g}; F1 {e0['f1_macro']:.4f} vs saved PTQ "
            f"{saved['f1_macro']:.4f}")
        if not torch.equal(epoch0_logits.argmax(1), ptq_logits.argmax(1)) or not all(
                abs(e0[k] - saved[k]) < 1e-9 for k in ("accuracy", "f1_macro", "melanoma_recall")):
            raise CheckFailed(f"seed {seed}: epoch 0 does not reproduce static PTQ: {info['epoch0_vs_static_ptq']}")
        evaluate = lambda: predict(to_int8(model), val_batches)  # noqa: E731

    set_seed(seed)  # QAT and control start training from the same RNG state (batch order, augmentation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history, best = [], None
    for epoch in range(EPOCHS + 1):
        start = time.time()
        row = {"seed": seed, "epoch": epoch, "lr": None, "train_loss": None,
               "bn_frozen": epoch > FREEZE_BN_AFTER, "observers_enabled": None if control else
               epoch <= DISABLE_OBSERVER_AFTER}
        if epoch > 0:
            row["lr"] = optimizer.param_groups[0]["lr"]
            model.train()
            if epoch > FREEZE_BN_AFTER:
                freeze_bn_fp32(model) if control else model.apply(nniqat.freeze_bn_stats)
            if not control and epoch > DISABLE_OBSERVER_AFTER:
                model.apply(disable_observer)
            row["train_loss"] = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
        m = metrics("x", evaluate(), y, fp32_ref, classes)
        row.update({f"val_{k}": m[k] for k in SCORES + ["identical_to_fp32_pct"]})
        row["epoch_seconds"] = time.time() - start
        history.append(row)
        log(f"{'control' if control else 'QAT'} seed {seed} epoch {epoch}: loss {row['train_loss']}, "
            f"val {'FP32' if control else 'INT8'} macro F1 {m['f1_macro']:.4f}, acc {m['accuracy']:.4f}, "
            f"mel recall {m['melanoma_recall']:.4f}, identical {m['identical_to_fp32_pct']:.2f}%")
        if not control and (best is None or m["f1_macro"] > best[1]):
            best = (epoch, m["f1_macro"], copy.deepcopy(model).cpu())
    if not control:
        info["best_epoch"], info["best_val_f1_macro"] = best[0], best[1]
        return history, info, best[2]
    return history, info, None


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_curves(history, control, ptq_seed_f1, fp32_f1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.color": "#e5e5e5",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "pdf.fonttype": 42})
    styles = {42: dict(color="#1f1e1c", marker="o", linestyle="-", linewidth=1.8),
              1: dict(color="#555555", marker="s", linestyle="--", linewidth=1.2),
              2: dict(color="#888888", marker="^", linestyle=":", linewidth=1.4)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for seed, rows in history.groupby("seed"):
        rows = rows.sort_values("epoch")
        trained = rows[rows["epoch"] > 0]
        label = f"QAT seed {seed}" + (" (saved)" if seed == SAVED_SEED else "")
        axes[0].plot(trained["epoch"], trained["train_loss"], label=label, markersize=5, **styles[seed])
        axes[1].plot(rows["epoch"], rows["val_f1_macro"] * 100, label=label, markersize=5, **styles[seed])
        axes[2].plot(rows["epoch"], rows["val_accuracy"] * 100, label=label, markersize=5, **styles[seed])
    if control is not None:
        c = control.groupby("epoch")[["val_f1_macro", "val_accuracy", "train_loss"]].mean().reset_index()
        kw = dict(color="#2a78d6", marker="D", markerfacecolor="white", linestyle="-.", linewidth=1.2, markersize=5,
                  label=f"FP32 control, no fake quant (mean of {control['seed'].nunique()} seeds; FP32 eval)")
        axes[0].plot(c["epoch"][1:], c["train_loss"][1:], **kw)
        axes[1].plot(c["epoch"], c["val_f1_macro"] * 100, **kw)
        axes[2].plot(c["epoch"], c["val_accuracy"] * 100, **kw)
    axes[1].axhline(fp32_f1 * 100, color="#2a78d6", linewidth=0.9, linestyle=(0, (1, 1)), label="FP32 (start)")
    axes[1].axhline(np.mean(ptq_seed_f1) * 100, color="#b0b0b0", linewidth=0.9, linestyle=(0, (5, 2)),
                    label="Static PTQ (mean of 3 seeds)")
    for ax, title, ylabel in zip(axes, ["Training loss (QAT, fake-quant)", "Validation macro F1 (INT8)",
                                        "Validation accuracy (INT8)"],
                                 ["Class-weighted cross-entropy", "Macro F1 (%)", "Accuracy (%)"]):
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Epoch (0 = before training, = static PTQ)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, EPOCHS + 1))
        for x, text in ((FREEZE_BN_AFTER + 0.5, "BN frozen"), (DISABLE_OBSERVER_AFTER + 0.5, "observers off")):
            ax.axvline(x, color="#b0b0b0", linewidth=0.8, linestyle="--")
            ax.text(x + 0.1, 1.0, text, transform=ax.get_xaxis_transform(), fontsize=7.5, color="#555555", va="top")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9)
    fig.text(0.01, -0.16, "QAT curves: fake-quantized model converted to INT8 (CPU) after each epoch. FP32 control: "
             "same fine-tuning without fake quantization, evaluated in FP32. Validation split only.",
             fontsize=8, color="#555555")
    fig.tight_layout()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "plots" / "training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "plots" / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_from_saved():
    """Training curves from the saved CSVs (nothing is re-run)."""
    history = pd.read_csv(OUT / "training_history.csv")
    control_path = OUT / "control_fp32_history.csv"
    ptq = pd.read_csv(STATIC / "results/validation_results.csv")
    fp32_f1 = ptq.loc[ptq["model"] == "FP32", "f1_macro"].iloc[0]
    plot_curves(history, pd.read_csv(control_path) if control_path.exists() else None,
                ptq[ptq["model"] != "FP32"]["f1_macro"].to_numpy(), fp32_f1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.quantized.engine = ENGINE
    torch.set_num_threads(EVAL_THREADS)
    if not torch.cuda.is_available():
        raise CheckFailed("no GPU")
    device = torch.device("cuda")
    hashes = {str(p): sha256(p) for p in PROTECTED}
    config = load_config(CONFIG)

    # Validation tensors, decoded once (no test loader is ever built)
    _, val_loader, class_to_idx = build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    val_batches = list(val_loader)
    y = torch.cat([labels for _, labels in val_batches]).numpy()
    fp32_logits = predict(load_fp32(config.num_classes).eval(), val_batches)
    fp32_ref = fp32_logits.argmax(1).numpy()
    fp32_row = {"seed": None, "epoch": None, **metrics("FP32", fp32_logits, y, fp32_ref, classes)}
    log(f"validation: {len(y)} images; FP32 macro F1 {fp32_row['f1_macro']:.4f}")

    history, infos, rows = [], [], [fp32_row]
    if not args.control_only:
        for seed in args.seeds:
            h, info, best_qat = run(seed, config, (y, val_batches), fp32_ref, classes, device)
            history += h
            infos.append(info)
            int8 = to_int8(best_qat)
            check_quantized(int8)
            logits = predict(int8, val_batches)
            rows.append({"seed": seed, "epoch": info["best_epoch"],
                         **metrics(f"QAT_seed{seed}", logits, y, fp32_ref, classes, save_confusion=True)})
            if seed == SAVED_SEED:
                torch.jit.save(torch.jit.script(int8), QAT_INT8)
                reloaded = torch.jit.load(str(QAT_INT8), map_location="cpu").eval()
                if not torch.equal(predict(reloaded, val_batches), logits):
                    raise CheckFailed("the reloaded TorchScript file gives different outputs from the converted model")
                rows[-1]["size_mb"] = QAT_INT8.stat().st_size / 1e6
                log(f"saved {QAT_INT8} (epoch {info['best_epoch']}, {rows[-1]['size_mb']:.2f} MB)")
            pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)  # progress, overwritten at end

        table = pd.DataFrame(rows)
        table.to_csv(OUT / "validation_results.csv", index=False)
        qat = table[table["model"] != "FP32"]
        cols = SCORES + ["identical_to_fp32_pct"]
        summary = pd.DataFrame([
            {"model": "FP32", **table[table["model"] == "FP32"].iloc[0][cols].to_dict()},
            {"model": "QAT_mean", **qat[cols].mean().to_dict()},
            {"model": "QAT_min", **qat[cols].min().to_dict()},
            {"model": "QAT_max", **qat[cols].max().to_dict()},
            *([{"model": f"QAT_saved_seed{SAVED_SEED}", **qat[qat["seed"] == SAVED_SEED].iloc[0][cols].to_dict()}]
              if SAVED_SEED in args.seeds else [])])
        summary.to_csv(OUT / "validation_summary.csv", index=False)
        pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
        (OUT / "training_history.json").write_text(json.dumps(
            {"selection": "epoch (0-10) with the best INT8 validation macro F1", "runs": infos, "history": history},
            indent=2, default=float))

    if args.control or args.control_only:
        control = []
        for seed in args.seeds:
            h, _, _ = run(seed, config, (y, val_batches), fp32_ref, classes, device, control=True)
            control += h
        pd.DataFrame(control).to_csv(OUT / "control_fp32_history.csv", index=False)
        (OUT / "control_fp32_history.json").write_text(json.dumps(
            {"description": "FP32 fine-tuned for the same 10 epochs with the same settings, no fake quantization; "
                            "FP32 validation metrics per epoch (epoch 0 = the FP32 model)", "history": control},
            indent=2, default=float))

    plot_from_saved()

    (OUT / "settings.json").write_text(json.dumps({
        "fp32_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "config": CONFIG,
        "model": "torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), strict load",
        "fusion": "model.train(); fuse_model(is_qat=True)", "qconfig": f"get_default_qat_qconfig('{ENGINE}')",
        "qconfig_repr": str(get_default_qat_qconfig(ENGINE)), "engine": ENGINE,
        "weights": "INT8 per-channel symmetric (qint8, -128..127)",
        "activations": "INT8 per-tensor affine (quint8, reduce_range=True -> 0..127)",
        "observer_init": "same 1,000 stratified training images as static PTQ for the seed (eval preprocessing, "
                         "no gradient, BN eval, fake quant off), then activation ranges copied from the static PTQ "
                         "HistogramObserver calibration on those images (and PTQ per-channel weight ranges), so epoch 0 = static PTQ",
        "seeds": args.seeds, "saved_seed": SAVED_SEED, "epochs": EPOCHS, "batch_size": config.batch_size,
        "optimizer": "Adam", "lr": LR, "lr_schedule": f"CosineAnnealingLR(T_max={EPOCHS}), stepped per epoch",
        "loss": "CrossEntropyLoss, class weights 'balanced' from the training split",
        "augmentation": config.augmentation, "freeze_bn_stats_after_epoch": FREEZE_BN_AFTER,
        "disable_observers_after_epoch": DISABLE_OBSERVER_AFTER,
        "selection": "epoch 0-10 with the best INT8 validation macro F1 (validation split; test never used)",
        "control": "same fine-tuning from FP32 without fake quantization, BN frozen after epoch 6, FP32 evaluation",
        "device_training": torch.cuda.get_device_name(0), "device_int8_eval": "cpu",
        "torch": torch.__version__, "num_workers": config.num_workers,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sha256_protected_files": hashes,
    }, indent=2))

    if {str(p): sha256(p) for p in PROTECTED} != hashes:
        raise CheckFailed("a protected file (FP32 checkpoint, split, config or PTQ output) changed during the run")
    log("protected files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--control", action="store_true", help="also run the FP32 control")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="for smoke tests only")
    parser.add_argument("--plot-only", action="store_true", help="redraw the training curves from the saved CSVs")
    arguments = parser.parse_args()
    EPOCHS = arguments.epochs
    if arguments.plot_only:
        plot_from_saved()
        sys.exit(0)
    try:
        main(arguments)
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
