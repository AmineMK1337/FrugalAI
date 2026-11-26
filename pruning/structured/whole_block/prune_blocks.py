"""Whole-block pruning of the fine-tuned ResNet-18 (stage 1), then fine-tuning. FP32, no quantization.

Removed blocks: the first k blocks of the greedy order in results/selection/ranking_seed42.csv (rank_blocks.py,
identity-shortcut blocks only). Each is replaced by nn.Identity (standard torch modules only: the saved whole model
loads in evaluate.py without any import). Every run starts from the same FP32 checkpoint, loaded (strict=True) into
torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), which is never modified.

Stages saved per run:
  raw       blocks removed, nothing else changed
  recal     raw + BatchNorm recalibration on the ranking subset (results/selection/subset_seed42.csv, 2,000 train images,
            evaluation preprocessing; block_scoring.recalibrate_bn: every BatchNorm reset, cumulative average)
  selected  best validation macro F1 epoch of the fine-tuning (first maximum)
  epoch10   last epoch
raw and recal are rebuilt deterministically from the checkpoint, the ranking CSV and the subset CSV (cuDNN
deterministic while they are built; checked by building recal twice), so they are saved as .pt only and not in git.

Fine-tuning: the recipe of pruning/structured/filter/prune.py, unchanged (class-weighted cross-entropy with train-split
weights, strong augmentation, Adam, batch 32, lr 1e-4, CosineAnnealingLR(T_max=10) stepped per epoch, seed set before
the loaders and again before the optimizer, best validation macro F1 epoch kept). One difference from filter pruning,
by design: fine-tuning starts from the BN-recalibrated model (--start recal, default). --start raw starts from the raw
model instead (filter pruning's choice), as a check; that run is named blocks<k>_seed<s>_rawstart.

Checks (stop the run on failure): the unpruned model reproduces the saved FP32 validation result
(pruning/structured/filter/results/fp32_validation.json, up to 2 images: GPU vs CPU); output shape (1, 7); removed
parameters exactly and MACs within 1% of the analytic values of the removed blocks; recal rebuilt twice is identical;
fused (fuse_model) + TorchScript logits equal the unfused ones (max difference recorded, stop above 1e-3, the
evaluate.py tolerance); reloaded .pt and .ts equal the saved model; protected files unchanged.

Data roles: train = ranking subset, BN recalibration, fine-tuning; validation = choosing k and the epoch; the test set
is never loaded.

Writes, for a run named blocks<k>_seed<s>[_rawstart]:
  pruning/structured/whole_block/models/<run>_{raw,recal}.pt              whole model (not in git)
  pruning/structured/whole_block/models/<run>_{selected,epoch10}.{pt,ts}  whole model and traced TorchScript (unfused)
  pruning/structured/whole_block/results/<run>/  summary.json, metrics.csv (one row per stage), training_history.csv,
      blocks.csv (removed blocks, analytic and measured parameters / MACs), run.log

Usage: python pruning/structured/whole_block/prune_blocks.py --k 1 2 3 4 5 --seeds 42 [1 2] [--start raw]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import block_scoring as bs  # noqa: E402  (chdir to the fine-tuning project)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch_pruning as tp  # noqa: E402
from importlib.metadata import version  # noqa: E402

from ham10000.data import build_dataloaders, compute_class_weights  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

sp = bs.sp
MODELS_DIR, RESULTS = HERE / "models", HERE / "results"
SELECTION = RESULTS / "selection"
RANKING_SEED, MODE = 42, "identity"
EXAMPLE = torch.zeros(1, 3, 224, 224)
STAGE_CHANNELS = {"layer1": (64, 56), "layer2": (128, 28), "layer3": (256, 14), "layer4": (512, 7)}
FP32_REFERENCE = HERE.parent / "filter" / "results" / "fp32_validation.json"
CheckFailed, log = sp.CheckFailed, sp.log


def removal_order():
    table = pd.read_csv(SELECTION / f"ranking_seed{RANKING_SEED}.csv")
    order = table[table["chosen"] == True].sort_values("step")["candidate"].tolist()  # noqa: E712
    saved = json.loads((SELECTION / "checks.json").read_text())["order_used_for_pruning"]
    if order != saved:
        raise CheckFailed(f"ranking CSV order {order} differs from checks.json {saved}")
    return order


def analytic(name):
    """Parameters (2 convs 3x3 C->C without bias + 2 BatchNorms) and conv MACs of an identity block."""
    c, hw = STAGE_CHANNELS[name.split(".")[0]]
    return 2 * 9 * c * c + 2 * 2 * c, 2 * 9 * c * c * hw * hw


def macs_params(model):
    macs, params = tp.utils.count_ops_and_params(model, EXAMPLE)
    return int(macs), int(params)


def save_pt(model, stem):
    model = copy.deepcopy(model).cpu().eval()
    torch.save(model, MODELS_DIR / f"{stem}.pt")
    reloaded = torch.load(MODELS_DIR / f"{stem}.pt", map_location="cpu", weights_only=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        if not torch.equal(reloaded(x), model(x)):
            raise CheckFailed(f"{stem}: reloaded whole model differs")


def save_model(model, stem):
    """As filter/prune.py save_model, into this method's folder: whole model .pt + traced TorchScript .ts."""
    save_pt(model, stem)
    model = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, EXAMPLE)
    traced.save(str(MODELS_DIR / f"{stem}.ts"))
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        if (torch.jit.load(str(MODELS_DIR / f"{stem}.ts"))(x) - model(x)).abs().max() > 1e-4:
            raise CheckFailed(f"{stem}: TorchScript differs")


@torch.no_grad()
def fused_difference(model, images):
    """Max |logits| difference between the unfused model and fuse_model() + torch.jit.trace (CPU, real images)."""
    model = copy.deepcopy(model).cpu().eval()
    reference = model(images)
    model.fuse_model()
    traced = torch.jit.trace(model, EXAMPLE)
    return float((traced(images) - reference).abs().max())


def build_recal(raw, batches):
    """BN recalibration with cuDNN deterministic (restored afterwards): rebuilding gives the same weights."""
    previous = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        recal = bs.recalibrate_bn(copy.deepcopy(raw), batches)
        again = bs.recalibrate_bn(copy.deepcopy(raw), batches)
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous
    a, b = recal.state_dict(), again.state_dict()
    if any(not torch.equal(a[key], b[key]) for key in a):
        raise CheckFailed("BN recalibration is not reproducible")
    return recal


def run(k, seed, start, order, config, val, subset_batches, device, settings):
    name = f"blocks{k}_seed{seed}" + ("_rawstart" if start == "raw" else "")
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "run.log"
    logfile.write_text("")
    y, batches, fp32_pred, mel, fp32_images = val
    removed = order[:k]
    model = sp.load_fp32()
    macs0, p0 = macs_params(model)

    # ── raw: blocks removed ──
    raw = bs.pruned_copy(model, removed, MODE).eval()
    macs1, p1 = macs_params(raw)
    with torch.no_grad():
        if tuple(raw(EXAMPLE).shape) != (1, sp.NUM_CLASSES):
            raise CheckFailed("output shape after removal")
    rows = [{"block": n, "params_analytic": analytic(n)[0], "macs_analytic": analytic(n)[1]} for n in removed]
    blocks = pd.DataFrame(rows)
    blocks.to_csv(out / "blocks.csv", index=False)
    params_removed, macs_removed = p0 - p1, macs0 - macs1
    if params_removed != blocks["params_analytic"].sum():
        raise CheckFailed(f"removed {params_removed} parameters, analytic {blocks['params_analytic'].sum()}")
    if abs(macs_removed / blocks["macs_analytic"].sum() - 1) > 0.01:
        raise CheckFailed(f"removed {macs_removed} MACs, analytic {blocks['macs_analytic'].sum()}")
    summary = {"run": name, "k": k, "seed": seed, "start": start, "removed_blocks": removed,
               "params_before": p0, "params_after": p1, "param_reduction_pct": params_removed / p0 * 100,
               "params_removed_analytic": int(blocks["params_analytic"].sum()),
               "gmacs_before": macs0 / 1e9, "gmacs_after": macs1 / 1e9, "gmac_reduction_pct": macs_removed / macs0 * 100,
               "gmacs_removed_analytic": float(blocks["macs_analytic"].sum() / 1e9), "settings": settings}
    log(f"{name}: removed {removed}; params {p0:,} -> {p1:,} (-{summary['param_reduction_pct']:.1f}%, analytic "
        f"{summary['params_removed_analytic']:,} removed), GMACs {macs0 / 1e9:.3f} -> {macs1 / 1e9:.3f} "
        f"(-{summary['gmac_reduction_pct']:.1f}%, analytic {summary['gmacs_removed_analytic']:.3f} removed)", logfile)

    # ── recal: BatchNorm recalibrated on the ranking subset ──
    recal = build_recal(raw.to(device), subset_batches)
    summary["fused_vs_unfused_max_abs_logit_difference"] = {s: fused_difference(m, fp32_images)
                                                            for s, m in (("raw", raw), ("recal", recal))}
    save_pt(raw, f"{name}_raw")
    save_pt(recal, f"{name}_recal")

    sp.set_seed(seed)  # as prune.py: seed, then the loaders
    train_loader, _, _ = build_dataloaders(config)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, sp.NUM_CLASSES, device))
    cpu_criterion = nn.CrossEntropyLoss(weight=criterion.weight.cpu())
    stage_metrics = {s: sp.metrics(sp.predict(m.to(device), batches, device), y, fp32_pred, mel, cpu_criterion)
                     for s, m in (("raw", raw), ("recal", recal))}
    for s, m in stage_metrics.items():
        log(f"{name} {s}: val F1 {m['f1_macro']:.4f}, acc {m['accuracy']:.4f}, bal acc {m['balanced_accuracy']:.4f}, "
            f"mel recall {m['melanoma_recall']:.4f}", logfile)
    summary["raw"], summary["recal"] = stage_metrics["raw"], stage_metrics["recal"]

    # ── fine-tuning: the filter/prune.py loop, unchanged ──
    model = (recal if start == "recal" else raw).to(device)
    start_metrics = stage_metrics[start]
    sp.set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=sp.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sp.EPOCHS)
    history = [{"epoch": 0, "lr": None, "train_loss": None, **{f"val_{k_}": v for k_, v in start_metrics.items()}}]
    best = None
    for epoch in range(1, sp.EPOCHS + 1):
        t0, lr = time.time(), optimizer.param_groups[0]["lr"]
        loss = sp.train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        m = sp.metrics(sp.predict(model, batches, device), y, fp32_pred, mel, cpu_criterion)
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss, **{f"val_{k_}": v for k_, v in m.items()},
                        "seconds": time.time() - t0})
        log(f"{name} epoch {epoch} lr {lr:.3g}: loss {loss:.4f}, val loss {m['val_loss']:.4f}, F1 {m['f1_macro']:.4f}, "
            f"acc {m['accuracy']:.4f}, bal acc {m['balanced_accuracy']:.4f}, mel {m['melanoma_recall']:.4f}", logfile)
        if best is None or m["f1_macro"] > best[1]["f1_macro"]:
            best = (epoch, m, copy.deepcopy(model).cpu())
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    summary["selected_epoch"], summary["selected"] = best[0], best[1]
    summary["epoch10"] = {k_[4:]: v for k_, v in history[-1].items() if k_.startswith("val_")}
    summary["fused_vs_unfused_max_abs_logit_difference"].update(
        selected=fused_difference(best[2], fp32_images), epoch10=fused_difference(model, fp32_images))
    worst = max(summary["fused_vs_unfused_max_abs_logit_difference"].values())
    if worst > 1e-3:
        raise CheckFailed(f"fused + TorchScript differs from the unfused model by {worst:.2e}")
    save_model(best[2], f"{name}_selected")
    save_model(model, f"{name}_epoch10")
    rows = [{"run": name, "stage": stage, **values} for stage, values in
            (("raw", stage_metrics["raw"]), ("bn_recalibrated", stage_metrics["recal"]),
             (f"selected_epoch{best[0]}", best[1]), ("epoch10", summary["epoch10"]))]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    summary["files"] = {s: str((MODELS_DIR / f"{name}_{s}.pt").relative_to(sp.ROOT))
                        for s in ("raw", "recal", "selected", "epoch10")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"{name}: selected epoch {best[0]} (val F1 {best[1]['f1_macro']:.4f}); epoch 10 F1 "
        f"{summary['epoch10']['f1_macro']:.4f}; fused vs unfused max logit difference {worst:.2e}; models saved",
        logfile)


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise CheckFailed("no GPU")
    device = torch.device("cuda")
    hashes = {str(p): sp.sha256(p) for p in sp.PROTECTED}
    config = load_config(sp.CONFIG)
    order = removal_order()

    # validation (choosing k and epochs only); the test set is never loaded
    _, val_loader, class_to_idx = build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    batches = list(val_loader)
    y = torch.cat([labels for _, labels in batches]).numpy()
    fp32 = sp.load_fp32()
    fp32_logits = sp.predict(fp32.to(device), batches, device)
    mel = classes.index(sp.MELANOMA)
    fp32_metrics = sp.metrics(fp32_logits, y, fp32_logits.argmax(1).numpy(), mel)
    saved = json.loads(FP32_REFERENCE.read_text())
    diff = abs(fp32_metrics["accuracy"] - saved["accuracy"]) * len(y)
    log(f"validation {len(y)} images; unpruned FP32 (strict load) val F1 {fp32_metrics['f1_macro']:.4f}, acc "
        f"{fp32_metrics['accuracy']:.4f}; saved CPU result F1 {saved['f1_macro']:.4f}, acc {saved['accuracy']:.4f} "
        f"({diff:.0f} image(s) apart)")
    if diff > 2:
        raise CheckFailed("the unpruned model does not reproduce its saved validation result")
    (RESULTS / "fp32_validation.json").write_text(json.dumps(
        {**fp32_metrics, "saved_cpu_result": str(FP32_REFERENCE.relative_to(sp.ROOT)), "accuracy_images_apart": diff},
        indent=2, default=float))

    # ranking subset (train only) for BN recalibration, and real images for the fused check
    subset = pd.read_csv(SELECTION / f"subset_seed{RANKING_SEED}.csv")
    subset_batches, leakage = bs.load_subset(config, subset, device)
    fp32_images = torch.cat([x for x, _ in subset_batches[:2]]).cpu()
    settings = {
        "checkpoint": str(sp.CHECKPOINT.relative_to(sp.ROOT)), "config": sp.CONFIG,
        "model": "torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), strict load",
        "pruning": "whole identity-shortcut BasicBlocks replaced by nn.Identity, first k of the greedy order",
        "removal_order": order, "ranking": str((SELECTION / f"ranking_seed{RANKING_SEED}.csv").relative_to(sp.ROOT)),
        "bn_recalibration": {"subset": str((SELECTION / f"subset_seed{RANKING_SEED}.csv").relative_to(sp.ROOT)),
                             "images": len(subset), "batch_size": bs.BATCH, "leakage_check": leakage,
                             "method": "reset every BatchNorm, momentum=None (cumulative average), train-mode "
                                       "forward passes without gradients, momentum restored"},
        "fine_tuning": {"epochs": sp.EPOCHS, "optimizer": "Adam", "lr": sp.LR, "schedule": "CosineAnnealingLR(T_max=10),"
                        " stepped per epoch", "batch_size": config.batch_size, "augmentation": config.augmentation,
                        "loss": "CrossEntropyLoss, balanced class weights from the train split",
                        "start": "BN-recalibrated model (filter pruning starts from the raw pruned model)" if args.start == "recal"
                        else "raw pruned model (as filter pruning)"},
        "selection": "best validation macro F1 epoch (first maximum); epoch 10 also reported",
        "torch": torch.__version__, "torch_pruning": version("torch-pruning"), "gpu": torch.cuda.get_device_name(0),
        "sha256_protected": hashes}
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2))  # last invocation; each run keeps a copy
    for seed in args.seeds:
        for k in args.k:
            run(k, seed, args.start, order, config, (y, batches, fp32_logits.argmax(1).numpy(), mel, fp32_images),
                subset_batches, device, settings)
    if {str(p): sp.sha256(p) for p in sp.PROTECTED} != hashes:
        raise CheckFailed("the FP32 checkpoint, the split or the config changed during the run")
    log("protected files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--start", choices=["recal", "raw"], default="recal")
    try:
        main(parser.parse_args())
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
