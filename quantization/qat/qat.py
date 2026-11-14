"""Quantization-aware training (QAT) of the fine-tuned ResNet-18: INT8 weights and activations (x86 engine,
per-channel symmetric INT8 weights, per-tensor activations with reduce_range=True).

Data roles: train split = QAT fine-tuning and observer initialization; validation split = monitoring each epoch.
The test split is never loaded. The FP32 checkpoint, the split and the static PTQ files used for the observer
initialization are hashed before and after the run and must not change.

 1. torchvision.models.quantization.resnet18(quantize=False), FP32 weights (strict); train(); fuse_model(is_qat=True);
    qconfig = get_default_qat_qconfig("x86"); prepare_qat.
 2. Observer initialization on 1,000 stratified training images (seed 42; the same images as the static PTQ
    calibration). The activation and per-channel weight ranges of a static PTQ calibration on those images are copied
    into the QAT observers, so before training the INT8 model equals static PTQ (checked).
 3. 10 epochs on the GPU, seed 42, weights trained in every phase (class-weighted cross-entropy with training-split
    weights, strong augmentation, batch 32, Adam):
      epochs 1-2   adaptation:    fake quant on, observers updating, BN updating, lr 1e-5 constant
      epochs 3-7   fine-tuning:   cosine lr decay starting from 1e-5 (8-epoch cosine over epochs 3-10)
      epochs 8-10  stabilization: BN statistics frozen (after epoch 7), observers disabled (scales and zero points
                                  fixed), weights keep training, lr continues the cosine decay toward 0
 4. After every epoch: copy to CPU, convert to INT8, evaluate on validation (monitoring only).
    The saved model is the final one (epoch 10): no epoch is chosen on validation.

Writes quantization/qat/resnet18_qat_int8.pt (TorchScript) and, in results/: settings.json, training_history.{json,csv},
validation_results.csv (FP32 and QAT), confusion_matrix_QAT.csv, plots/training_curves.{pdf,png}.

Usage: python quantization/qat/qat.py [--plot-only]
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support,  # noqa: E402
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
N_CALIBRATION, SEED = 1000, 42
LR, EPOCHS = 1e-5, 10
ADAPTATION_END, FINETUNE_END = 2, 7  # epochs 1-2 adaptation, 3-7 fine-tuning, 8-10 stabilization
EVAL_THREADS = 16
SCORES = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro", "auc_macro",
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
        "model": name, "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(), "f1_macro": f1.mean(), "specificity_macro": (tn / (tn + fp)).mean(),
        "auc_macro": np.mean([roc_auc_score(y == i, probs[:, i]) for i in ids]),
        "melanoma_recall": recall[classes.index("mel")],
        "identical_to_fp32_pct": (pred == reference_pred).mean() * 100,
        "wrong_to_right_vs_fp32": gained, "right_to_wrong_vs_fp32": lost,
        "mcnemar_p_vs_fp32": binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0,
    }


# ── Training ─────────────────────────────────────────────────────────────────

def phase(epoch):
    if epoch == 0:
        return "start"
    return "adaptation" if epoch <= ADAPTATION_END else "fine-tuning" if epoch <= FINETUNE_END else "stabilization"


def lr_at(epoch):
    """1e-5 constant in epochs 1-2, then one cosine from 1e-5 (epoch 3) toward 0 over epochs 3-10."""
    if epoch <= ADAPTATION_END:
        return LR
    k, n = epoch - ADAPTATION_END - 1, EPOCHS - ADAPTATION_END
    return LR * 0.5 * (1 + math.cos(math.pi * k / n))


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


def start_from_ptq(config, model, val_batches, y, fp32_ref, classes, device, info):
    """Observer initialization (literal pass, then PTQ ranges) and the check that epoch 0 = static PTQ."""
    calibration = calibration_subset(config, SEED)
    ptq = static_ptq_prepared(config.num_classes, calibration)
    ptq_logits = predict(convert(copy.deepcopy(ptq), inplace=False).eval(), val_batches)
    init_observers(model, calibration, device)
    info["activation_observers_copied"] = copy_ptq_activation_ranges(model, ptq)
    info["weight_observers_copied"] = copy_ptq_weight_ranges(model, ptq)
    int8 = to_int8(model)
    check_quantized(int8)
    epoch0_logits = predict(int8, val_batches)
    saved = pd.read_csv(STATIC / "results/validation_results.csv")
    saved = saved[saved["seed"] == SEED].iloc[0]
    e0 = metrics("e0", epoch0_logits, y, fp32_ref, classes)
    info["epoch0_vs_static_ptq"] = {
        "identical_predictions_pct": float((epoch0_logits.argmax(1) == ptq_logits.argmax(1)).float().mean() * 100),
        "max_abs_logit_difference": float((epoch0_logits - ptq_logits).abs().max())}
    log(f"epoch 0 vs static PTQ: {info['epoch0_vs_static_ptq']}")
    if not torch.equal(epoch0_logits.argmax(1), ptq_logits.argmax(1)) or not all(
            abs(e0[k] - saved[k]) < 1e-9 for k in ("accuracy", "f1_macro", "melanoma_recall")):
        raise CheckFailed(f"epoch 0 does not reproduce static PTQ: {info['epoch0_vs_static_ptq']}")


def train(config, val, fp32_ref, classes, device):
    """QAT, seed 42. Returns (history rows, info, final QAT model on the CPU)."""
    y, val_batches = val
    set_seed(SEED)
    train_loader, _, _ = build_dataloaders(config)
    criterion = torch.nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, config.num_classes, device))
    info = {"seed": SEED}
    model = qat_prepared(config.num_classes).to(device)
    start_from_ptq(config, model, val_batches, y, fp32_ref, classes, device, info)

    set_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    for epoch in range(EPOCHS + 1):
        start = time.time()
        bn_frozen, observers_on = epoch > FINETUNE_END, epoch <= FINETUNE_END
        row = {"epoch": epoch, "phase": phase(epoch), "lr": None, "bn": "frozen" if bn_frozen else "updating",
               "observers": "updating" if observers_on else "disabled", "train_loss": None}
        if epoch > 0:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(epoch)
            row["lr"] = lr_at(epoch)
            model.train()
            if bn_frozen:
                model.apply(nniqat.freeze_bn_stats)
            if not observers_on:
                model.apply(disable_observer)
            row["train_loss"] = train_one_epoch(model, train_loader, criterion, optimizer, device)
        m = metrics("x", predict(to_int8(model), val_batches), y, fp32_ref, classes)
        row.update({f"val_{k}": m[k] for k in SCORES + ["identical_to_fp32_pct"]})
        row["epoch_seconds"] = time.time() - start
        history.append(row)
        log(f"epoch {epoch} [{row['phase']}, lr {row['lr']}, BN {row['bn']}, observers {row['observers']}]: "
            f"loss {row['train_loss']}, val INT8 F1 {m['f1_macro']:.4f}, acc {m['accuracy']:.4f}, bal acc "
            f"{m['balanced_accuracy']:.4f}, mel recall {m['melanoma_recall']:.4f}, identical "
            f"{m['identical_to_fp32_pct']:.2f}%")
    return history, info, copy.deepcopy(model).cpu()


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_curves(history, fp32_row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.color": "#e5e5e5",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "pdf.fonttype": 42})
    panels = [("val_f1_macro", "Macro F1 (%)", "f1_macro"), ("val_accuracy", "Accuracy (%)", "accuracy"),
              ("val_melanoma_recall", "Melanoma recall (%)", "melanoma_recall")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (column, ylabel, ref) in zip(axes, panels):
        ax.plot(history["epoch"], history[column] * 100, color="#8b5cc7", linewidth=2.2, marker="o", markersize=5,
                label="INT8 QAT (converted after each epoch)")
        ax.axhline(fp32_row[ref] * 100, color="#2a78d6", linewidth=1.3, linestyle="--", label="FP32 baseline")
        for lo, hi, shade, name in ((0.5, ADAPTATION_END + 0.5, "#f2f2f2", "adaptation"),
                                    (ADAPTATION_END + 0.5, FINETUNE_END + 0.5, "#ffffff", "fine-tuning"),
                                    (FINETUNE_END + 0.5, EPOCHS + 0.5, "#dcdcdc", "stabilization")):
            ax.axvspan(lo, hi, color=shade, zorder=0)
            ax.axvline(lo, color="#9a9a9a", linewidth=0.7, linestyle="--", zorder=0.5)
            ax.text((lo + hi) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.5, color="#444444")
        ax.set_xticks(range(0, EPOCHS + 1))
        ax.set_xlim(-0.4, EPOCHS + 0.5)
        ax.set_xlabel("Epoch (0 = before training)")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.suptitle("QAT training curves, validation split (seed 42; saved model = epoch 10)", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "plots" / "training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "plots" / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_from_saved():
    """Training curves from the saved CSVs (nothing is re-run)."""
    table = pd.read_csv(OUT / "validation_results.csv")
    plot_curves(pd.read_csv(OUT / "training_history.csv"), table[table["model"] == "FP32"].iloc[0])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
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
    rows = [{"epoch": None, **metrics("FP32", fp32_logits, y, fp32_ref, classes)}]
    log(f"validation: {len(y)} images; FP32 macro F1 {rows[0]['f1_macro']:.4f}")

    history, info, model = train(config, (y, val_batches), fp32_ref, classes, device)
    int8 = to_int8(model)
    check_quantized(int8)
    logits = predict(int8, val_batches)
    rows.append({"epoch": EPOCHS, **metrics("QAT", logits, y, fp32_ref, classes, save_confusion=True)})
    if abs(rows[-1]["f1_macro"] - history[-1]["val_f1_macro"]) > 1e-12:
        raise CheckFailed("final metrics differ from the training-time evaluation")
    torch.jit.save(torch.jit.script(int8), QAT_INT8)
    reloaded = torch.jit.load(str(QAT_INT8), map_location="cpu").eval()
    if not torch.equal(predict(reloaded, val_batches), logits):
        raise CheckFailed("the reloaded TorchScript file gives different outputs from the converted model")
    rows[-1]["size_mb"] = QAT_INT8.stat().st_size / 1e6
    log(f"saved {QAT_INT8} (epoch {EPOCHS}, {rows[-1]['size_mb']:.2f} MB)")

    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    (OUT / "training_history.json").write_text(json.dumps({"run": info, "history": history}, indent=2, default=float))
    pd.DataFrame(rows).to_csv(OUT / "validation_results.csv", index=False)
    plot_from_saved()

    (OUT / "settings.json").write_text(json.dumps({
        "fp32_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "config": CONFIG,
        "model": "torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), strict load",
        "fusion": "model.train(); fuse_model(is_qat=True)", "qconfig": f"get_default_qat_qconfig('{ENGINE}')",
        "qconfig_repr": str(get_default_qat_qconfig(ENGINE)), "engine": ENGINE,
        "weights": "INT8 per-channel symmetric (qint8, -128..127)",
        "activations": "INT8 per-tensor affine (quint8, reduce_range=True -> 0..127)",
        "observer_init": "1,000 stratified training images (seed 42, same as the static PTQ calibration); the "
                         "activation and per-channel weight ranges of a static PTQ calibration on those images are "
                         "copied into the QAT observers, so epoch 0 = static PTQ (checked)",
        "seed": SEED, "epochs": EPOCHS, "batch_size": config.batch_size,
        "optimizer": "Adam", "loss": "CrossEntropyLoss, class weights 'balanced' from the training split",
        "augmentation": config.augmentation,
        "schedule": {
            "adaptation": f"epochs 1-{ADAPTATION_END}: fake quant on, observers updating, BN updating, lr {LR} constant",
            "fine-tuning": f"epochs {ADAPTATION_END + 1}-{FINETUNE_END}: cosine lr decay from {LR}, observers and BN "
                           "updating",
            "stabilization": f"epochs {FINETUNE_END + 1}-{EPOCHS}: BN statistics frozen (after epoch {FINETUNE_END}), "
                             "observers disabled (scales/zero points fixed), weights keep training, lr continues "
                             "the cosine toward 0",
            "lr_per_epoch": {e: lr_at(e) for e in range(1, EPOCHS + 1)}},
        "saved_model": f"final model (epoch {EPOCHS}); validation only monitored, no epoch selection; test never used",
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
    parser.add_argument("--plot-only", action="store_true", help="redraw the training curves from the saved CSVs")
    if parser.parse_args().plot_only:
        plot_from_saved()
        sys.exit(0)
    try:
        main()
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
