"""Structured (channel) pruning of the fine-tuned ResNet-18, then fine-tuning. Structured only, FP32, no quantization.

Every run starts from the same FP32 checkpoint, loaded (strict=True) into
torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), so the pruned models can be
fused (fuse_model) and quantized later.

Pruning: Torch-Pruning DependencyGraph (dependency-aware: the convolutions joined by a residual addition are pruned
together, on the same channel indices), local pruning with the same TARGET ratio for every prunable channel group.
Importance: L1 norm of each output filter (Li et al., 2017, "Pruning Filters for Efficient ConvNets"); when several
convolutions share a channel group (residual stream: conv2 of each block and the downsample conv), the filter L1 norms
of those convolutions are averaged per channel. Excluded (ignored_layers): the stem conv1 (so its whole channel group,
the layer1 residual stream, is kept) and fc (its 7 outputs; its inputs follow layer4). The ACHIEVED ratio is recorded
per layer (rounding, groups).

Fine-tuning, same setup as the FP32 training: class-weighted cross-entropy (weights from the train split), strong
augmentation, eval preprocessing for validation, Adam, batch 32, lr 1e-4, cosine annealing over 10 epochs (stepped per
epoch), seed given. Validation after every epoch; the selected model is the best validation macro F1 epoch (first
maximum), epoch 10 is also reported. FP32 control (--control): the unpruned model fine-tuned with the same settings.

Data roles: train = fine-tuning; validation = choosing ratios and epochs; the test set is never loaded.
The FP32 checkpoint, the split and the config are hashed before and after the run.

Writes, for a run named pruned<ratio>_seed<seed> or control_seed<seed>:
  pruning/structured/models/<run>_{selected,epoch10}.pt   whole model (torch.save; load with weights_only=False)
  pruning/structured/models/<run>_{selected,epoch10}.ts   TorchScript (traced, unfused, eval mode)
  pruning/structured/results/<run>/  layers.csv (per-layer channels before/after, achieved ratio), summary.json
      (overall channel / parameter / GMAC reduction, metrics before fine-tuning, selected epoch, epoch 10, and a
      copy of the run's settings: pruning, fine-tuning, versions, hashes),
      training_history.csv, metrics.csv (one row per stage), run.log

Usage: python pruning/structured/prune.py --ratios 0.2 [0.1 ...] --seeds 42 [1 2] [--control]
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / "resnet18_finetunning"
sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project
warnings.filterwarnings("ignore", message=".*(TypedStorage|torch.ao.quantization).*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch_pruning as tp  # noqa: E402
from importlib.metadata import version  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torchvision.models.quantization import resnet18  # noqa: E402

from ham10000.data import build_dataloaders, compute_class_weights  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = PROJECT / "results/strong_cosine25_resnet18/resnet18_baseline.pth"
PROTECTED = [CHECKPOINT, PROJECT / "splits/ham10000_splits.csv", PROJECT / CONFIG]
MODELS_DIR, RESULTS = HERE / "models", HERE / "results"
EPOCHS, LR = 10, 1e-4
NUM_CLASSES, MELANOMA = 7, "mel"
SCORES = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro",
          "auc_macro", "melanoma_recall"]


class CheckFailed(RuntimeError):
    pass


def log(message, file=None):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    if file:
        with open(file, "a") as f:
            f.write(line + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fp32():
    model = resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


# ── Pruning ──────────────────────────────────────────────────────────────────

class FilterL1Importance(tp.importance.Importance):
    """L1 norm of each output filter (Li et al., 2017). For a channel group shared by several convolutions (residual
    stream), the per-channel filter L1 norms of the convolutions whose OUTPUT channels are in the group are averaged.
    BatchNorm weights and the input-channel weights of the next layers are not used."""

    def __call__(self, group):
        scores = [dep.target.module.weight.detach().abs().flatten(1).sum(1)[idxs]
                  for dep, idxs in group
                  if isinstance(dep.target.module, nn.Conv2d) and dep.handler == tp.prune_conv_out_channels]
        return torch.stack(scores).mean(0) if scores else None


def layer_shapes(model):
    """(in, out) of every Conv2d, BatchNorm2d and Linear, plus parameter count, by module name."""
    rows = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            rows[name] = ("conv", m.in_channels, m.out_channels, sum(p.numel() for p in m.parameters()))
        elif isinstance(m, nn.BatchNorm2d):
            rows[name] = ("bn", m.num_features, m.num_features, sum(p.numel() for p in m.parameters()))
        elif isinstance(m, nn.Linear):
            rows[name] = ("linear", m.in_features, m.out_features, sum(p.numel() for p in m.parameters()))
    return rows


def prune(model, ratio, example):
    """Prune in place (local, same target ratio for every group); returns the dependency groups that were prunable."""
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    names = {m: n for n, m in model.named_modules()}
    groups = []
    for g in dg.get_all_groups(ignored_layers=[model.conv1, model.fc], root_module_types=[nn.Conv2d, nn.Linear]):
        outs = sorted({names[d.target.module] for d, _ in g if d.handler == tp.prune_conv_out_channels})
        groups.append(outs)
    pruner = tp.pruner.BasePruner(model, example, importance=FilterL1Importance(), pruning_ratio=ratio,
                                  global_pruning=False, ignored_layers=[model.conv1, model.fc])
    pruner.step()
    return groups


def pruning_report(before, after, macs_params):
    rows = []
    for name, (kind, cin, cout, params) in before.items():
        _, cin2, cout2, params2 = after[name]
        rows.append({"layer": name, "type": kind, "in_before": cin, "in_after": cin2, "out_before": cout,
                     "out_after": cout2, "out_removed_pct": (1 - cout2 / cout) * 100,
                     "in_removed_pct": (1 - cin2 / cin) * 100, "params_before": params, "params_after": params2,
                     "params_removed_pct": (1 - params2 / params) * 100})
    layers = pd.DataFrame(rows)
    convs = layers[layers["type"] == "conv"]
    prunable = convs[convs["layer"] != "conv1"]
    (macs0, p0), (macs1, p1) = macs_params
    overall = {
        "channel_reduction_pct_all_convs": (1 - convs["out_after"].sum() / convs["out_before"].sum()) * 100,
        "channel_reduction_pct_excluding_stem": (1 - prunable["out_after"].sum() / prunable["out_before"].sum()) * 100,
        "params_before": p0, "params_after": p1, "param_reduction_pct": (1 - p1 / p0) * 100,
        "gmacs_before": macs0 / 1e9, "gmacs_after": macs1 / 1e9, "gmac_reduction_pct": (1 - macs1 / macs0) * 100,
        "achieved_out_ratio_min_pct": float(prunable.loc[prunable["out_removed_pct"] > 0, "out_removed_pct"].min())
        if (prunable["out_removed_pct"] > 0).any() else 0.0,
        "achieved_out_ratio_max_pct": float(prunable["out_removed_pct"].max()),
        "convs_with_unpruned_outputs": convs.loc[convs["out_removed_pct"] == 0, "layer"].tolist(),
    }
    return layers, overall


# ── Metrics ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    return torch.cat([model(x.to(device, non_blocking=True)).float().cpu() for x, _ in batches])


def metrics(logits, y, reference_pred, mel, criterion=None):
    probs = logits.softmax(1).numpy()
    pred = probs.argmax(1)
    cm = np.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp_, predicted, actual = np.diag(cm).astype(float), cm.sum(0), cm.sum(1)
    precision = np.where(predicted > 0, tp_ / np.maximum(predicted, 1), 0.0)
    recall = tp_ / actual
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    specificity = (cm.sum() - actual - predicted + tp_) / (cm.sum() - actual)
    out = {"accuracy": tp_.sum() / len(y), "balanced_accuracy": recall.mean(), "precision_macro": precision.mean(),
           "recall_macro": recall.mean(), "specificity_macro": specificity.mean(), "f1_macro": f1.mean(),
           "auc_macro": np.mean([roc_auc_score(y == i, probs[:, i]) for i in range(NUM_CLASSES)]),
           "melanoma_recall": recall[mel], "identical_to_fp32_pct": (pred == reference_pred).mean() * 100}
    if criterion is not None:
        out["val_loss"] = float(criterion(logits, torch.as_tensor(y)).item())
    return out


# ── Run ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
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


def save_model(model, stem):
    model = copy.deepcopy(model).cpu().eval()
    torch.save(model, MODELS_DIR / f"{stem}.pt")
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.zeros(1, 3, 224, 224))
    traced.save(str(MODELS_DIR / f"{stem}.ts"))
    reloaded = torch.load(MODELS_DIR / f"{stem}.pt", map_location="cpu", weights_only=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        if not torch.equal(reloaded(x), model(x)):
            raise CheckFailed(f"{stem}: reloaded whole model differs")
        if (torch.jit.load(str(MODELS_DIR / f"{stem}.ts"))(x) - model(x)).abs().max() > 1e-4:
            raise CheckFailed(f"{stem}: TorchScript differs")


def run(ratio, seed, config, val, device, settings):
    name = f"control_seed{seed}" if ratio is None else f"pruned{round(ratio * 100)}_seed{seed}"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "run.log"
    logfile.write_text("")
    y, batches, fp32_pred, mel = val
    model = load_fp32()
    example = torch.zeros(1, 3, 224, 224)
    before = layer_shapes(model)
    macs0, p0 = tp.utils.count_ops_and_params(model, example)
    summary = {"run": name, "target_ratio": ratio, "seed": seed, "settings": settings}

    if ratio is not None:
        state0 = {k: v.clone() for k, v in model.state_dict().items()}
        groups = prune(model, ratio, example)
        after = layer_shapes(model)
        macs1, p1 = tp.utils.count_ops_and_params(model, example)
        layers, overall = pruning_report(before, after, ((macs0, p0), (macs1, p1)))
        # checks: forward pass, physically smaller weights, protected layers untouched
        with torch.no_grad():
            if tuple(model(example).shape) != (1, NUM_CLASSES):
                raise CheckFailed("forward pass after pruning")
        for n_, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and tuple(m.weight.shape) != (m.out_channels, m.in_channels,
                                                                      *m.kernel_size):
                raise CheckFailed(f"{n_}: weight shape {tuple(m.weight.shape)} does not match its channels")
        if not torch.equal(model.conv1.weight, state0["conv1.weight"]) or model.fc.out_features != NUM_CLASSES:
            raise CheckFailed("an excluded layer changed")
        if p1 >= p0:
            raise CheckFailed("no parameter was removed")
        layers.to_csv(out / "layers.csv", index=False)
        summary.update({"dependency_groups_prunable": groups, **overall})
        log(f"{name}: params {p0:,} -> {p1:,} (-{overall['param_reduction_pct']:.1f}%), GMACs {macs0 / 1e9:.3f} -> "
            f"{macs1 / 1e9:.3f} (-{overall['gmac_reduction_pct']:.1f}%), channels -"
            f"{overall['channel_reduction_pct_all_convs']:.1f}% (all convs); achieved per-layer output ratio "
            f"{overall['achieved_out_ratio_min_pct']:.1f}-{overall['achieved_out_ratio_max_pct']:.1f}%", logfile)
    else:
        summary.update({"params_before": p0, "params_after": p0, "gmacs_before": macs0 / 1e9,
                        "gmacs_after": macs0 / 1e9})

    model.to(device)
    set_seed(seed)
    train_loader, _, _ = build_dataloaders(config)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, NUM_CLASSES, device))
    cpu_criterion = nn.CrossEntropyLoss(weight=criterion.weight.cpu())
    start_metrics = metrics(predict(model, batches, device), y, fp32_pred, mel, cpu_criterion)
    summary["before_finetuning"] = start_metrics
    log(f"{name} before fine-tuning: val F1 {start_metrics['f1_macro']:.4f}, acc {start_metrics['accuracy']:.4f}, "
        f"bal acc {start_metrics['balanced_accuracy']:.4f}, mel recall {start_metrics['melanoma_recall']:.4f}",
        logfile)

    set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = [{"epoch": 0, "lr": None, "train_loss": None, **{f"val_{k}": v for k, v in start_metrics.items()}}]
    best = None
    for epoch in range(1, EPOCHS + 1):
        t0, lr = time.time(), optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        m = metrics(predict(model, batches, device), y, fp32_pred, mel, cpu_criterion)
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss, **{f"val_{k}": v for k, v in m.items()},
                        "seconds": time.time() - t0})
        log(f"{name} epoch {epoch} lr {lr:.3g}: loss {loss:.4f}, val loss {m['val_loss']:.4f}, F1 {m['f1_macro']:.4f}, "
            f"acc {m['accuracy']:.4f}, bal acc {m['balanced_accuracy']:.4f}, mel {m['melanoma_recall']:.4f}", logfile)
        if best is None or m["f1_macro"] > best[1]["f1_macro"]:
            best = (epoch, m, copy.deepcopy(model).cpu())
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    summary["selected_epoch"], summary["selected"] = best[0], best[1]
    summary["epoch10"] = {k[4:]: v for k, v in history[-1].items() if k.startswith("val_")}
    save_model(best[2], f"{name}_selected")
    save_model(model, f"{name}_epoch10")
    rows = [{"run": name, "stage": stage, **values} for stage, values in
            (("before_finetuning", start_metrics), (f"selected_epoch{best[0]}", best[1]),
             ("epoch10", summary["epoch10"]))]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    summary["files"] = {s: str((MODELS_DIR / f"{name}_{s}.pt").relative_to(ROOT)) for s in ("selected", "epoch10")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"{name}: selected epoch {best[0]} (val F1 {best[1]['f1_macro']:.4f}); epoch 10 F1 "
        f"{summary['epoch10']['f1_macro']:.4f}; models saved", logfile)


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise CheckFailed("no GPU")
    device = torch.device("cuda")
    hashes = {str(p): sha256(p) for p in PROTECTED}
    config = load_config(CONFIG)
    _, val_loader, class_to_idx = build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    batches = list(val_loader)  # decoded once; the test set is never loaded
    y = torch.cat([labels for _, labels in batches]).numpy()
    fp32 = load_fp32()
    fp32_logits = predict(fp32, batches, "cpu")
    mel = classes.index(MELANOMA)
    fp32_metrics = metrics(fp32_logits, y, fp32_logits.argmax(1).numpy(), mel)
    log(f"validation {len(y)} images; FP32 val F1 {fp32_metrics['f1_macro']:.4f}, acc {fp32_metrics['accuracy']:.4f}")
    (RESULTS / "fp32_validation.json").write_text(json.dumps(fp32_metrics, indent=2, default=float))
    settings = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "config": CONFIG,
        "model": "torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), strict load",
        "pruning": "Torch-Pruning DependencyGraph + BasePruner, local (global_pruning=False), same target ratio for "
                   "every prunable channel group, ignored_layers=[conv1, fc]",
        "importance": "L1 norm of each output filter; averaged over the convs whose outputs share a channel group",
        "fine_tuning": {"epochs": EPOCHS, "optimizer": "Adam", "lr": LR, "schedule": "CosineAnnealingLR(T_max=10), "
                        "stepped per epoch", "batch_size": config.batch_size, "augmentation": config.augmentation,
                        "loss": "CrossEntropyLoss, balanced class weights from the train split"},
        "selection": "best validation macro F1 epoch (first maximum); epoch 10 also reported",
        "torch": torch.__version__, "torch_pruning": version("torch-pruning"),
        "torch_pruning_module_version": tp.__version__, "gpu": torch.cuda.get_device_name(0),
        "sha256_protected": hashes}
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2))  # last invocation; each run keeps a copy
    for seed in args.seeds:
        if args.control:
            run(None, seed, config, (y, batches, fp32_logits.argmax(1).numpy(), mel), device, settings)
        for ratio in args.ratios:
            run(ratio, seed, config, (y, batches, fp32_logits.argmax(1).numpy(), mel), device, settings)
    if {str(p): sha256(p) for p in PROTECTED} != hashes:
        raise CheckFailed("the FP32 checkpoint, the split or the config changed during the run")
    log("protected files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=float, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--control", action="store_true", help="also run the unpruned FP32 control")
    try:
        main(parser.parse_args())
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
