"""Block (depth) pruning of the fine-tuned ResNet-18, then fine-tuning. Whole residual blocks are removed; the
remaining layers keep their full width. FP32, no quantization, no filter or weight pruning.

Every run starts from the same FP32 checkpoint, loaded (strict=True) into
torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7).

Removable blocks: only BasicBlocks whose shortcut is the identity (same shape in and out): layer1.0, layer1.1,
layer2.1, layer3.1, layer4.1. layer2.0, layer3.0 and layer4.0 change the resolution and the channel count (downsample
shortcut) and are kept. A removed block is replaced by nn.Identity: its input goes straight to the next block. A
BasicBlock computes relu(x + F(x)) and x is already non-negative (output of a ReLU), so the identity is exactly "skip
the residual branch F".

Block importance (computed once, on the full model, before any removal: one-shot ranking): the increase of the
class-weighted cross-entropy on a fixed subset of 1,000 TRAIN images (seed 42, evaluation preprocessing: resize +
normalize, no augmentation) when that block alone is replaced by the identity. The k blocks with the smallest
increase are removed (k = 1, 2, 3, 4). The validation split is not used for the ranking.

Fine-tuning, identical to pruning/structured/prune.py (whose functions are reused): class-weighted cross-entropy
(train-split weights), strong augmentation, Adam, batch 32, lr 1e-4, cosine annealing over 10 epochs (stepped per
epoch), seed given. Selected model = best validation macro F1 epoch (first maximum); epoch 10 is also reported.
The FP32 control is reused from pruning/structured (control_seed<seed>), not retrained.

Data roles: train = block ranking and fine-tuning; validation = choosing k and epochs; the test set is never loaded.
The FP32 checkpoint, the split and the config are hashed before and after.

Writes:
  pruning/block/results/block_importance.csv   loss / accuracy / macro F1 on the ranking subset with each block removed
  pruning/block/models/blocks<k>_seed<s>_{before,selected,epoch10}.{pt,ts}  whole model (torch.save; load with
      weights_only=False) and TorchScript. before = right after removal, not fine-tuned. Not in git.
  pruning/block/results/blocks<k>_seed<s>/  summary.json, training_history.csv, metrics.csv (one row per stage), run.log

Usage: python pruning/block/prune_blocks.py --blocks 1 2 3 4 --seeds 42 [1 2]
"""
import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("structured_prune", ROOT / "pruning" / "structured" / "prune.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)  # helpers, data and the fine-tuning loop of the structured runs (also chdir + sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch_pruning as tp  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from ham10000.data import Ham10000Dataset, eval_transform, lesion_split  # noqa: E402

MODELS_DIR, RESULTS = HERE / "models", HERE / "results"
CONTROL_RESULTS = ROOT / "pruning" / "structured" / "results"
REMOVABLE = ["layer1.0", "layer1.1", "layer2.1", "layer3.1", "layer4.1"]
RANKING_IMAGES, RANKING_SEED = 1000, 42
EPOCHS, LR = sp.EPOCHS, sp.LR
CheckFailed, log = sp.CheckFailed, sp.log
EXAMPLE = torch.zeros(1, 3, 224, 224)


def remove_blocks(model, blocks):
    """Replace each named block (e.g. "layer3.1") by nn.Identity, in place."""
    for name in blocks:
        layer, index = name.split(".")
        block = getattr(model, layer)[int(index)]
        if block.downsample is not None:
            raise CheckFailed(f"{name} has a downsample shortcut and cannot be removed")
        getattr(model, layer)[int(index)] = nn.Identity()
    return model


def macs_params(model):
    macs, params = tp.utils.count_ops_and_params(model, EXAMPLE)
    return macs / 1e9, int(params)


def ranking_loader(config):
    """1,000 fixed train images (seed 42) with the evaluation preprocessing."""
    dataset = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=eval_transform(config))
    train_idx, _, _ = lesion_split(dataset.data, config)
    rng = np.random.default_rng(RANKING_SEED)
    chosen = sorted(rng.choice(np.asarray(train_idx), size=RANKING_IMAGES, replace=False).tolist())
    return DataLoader(Subset(dataset, chosen), batch_size=64, shuffle=False, num_workers=config.num_workers)


def rank_blocks(config, criterion, device):
    batches = list(ranking_loader(config))
    y = torch.cat([labels for _, labels in batches]).numpy()

    def score(model):
        logits = sp.predict(model.to(device), batches, device)
        m = sp.metrics(logits, y, logits.argmax(1).numpy(), 0, criterion)
        return {"loss": m["val_loss"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}

    full = score(sp.load_fp32())
    rows = [{"block": "none (full model)", **full, "loss_increase": 0.0}]
    for block in REMOVABLE:
        s = score(remove_blocks(sp.load_fp32(), [block]))
        rows.append({"block": block, **s, "loss_increase": s["loss"] - full["loss"]})
    table = pd.DataFrame(rows)
    order = table.iloc[1:].sort_values("loss_increase")["block"].tolist()
    table["removal_order"] = table["block"].map({b: i + 1 for i, b in enumerate(order)})
    return table, order


def control_metrics(seed):
    path = CONTROL_RESULTS / f"control_seed{seed}" / "metrics.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return {"selected": table[table["stage"].str.startswith("selected")].iloc[0].to_dict(),
            "epoch10": table[table["stage"] == "epoch10"].iloc[0].to_dict()}


def run(k, seed, order, config, val, device, settings):
    name = f"blocks{k}_seed{seed}"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "run.log"
    logfile.write_text("")
    y, batches, fp32_pred, mel = val
    model = sp.load_fp32()
    gmacs0, params0 = macs_params(model)
    removed = order[:k]
    remove_blocks(model, removed)
    gmacs1, params1 = macs_params(model)
    with torch.no_grad():
        if tuple(model(EXAMPLE).shape) != (1, sp.NUM_CLASSES):
            raise CheckFailed("forward pass after block removal")
    kept = [f"{l}.{i}" for l in ("layer1", "layer2", "layer3", "layer4") for i in (0, 1)
            if not isinstance(getattr(model, l)[i], nn.Identity)]
    if len(kept) != 8 - k:
        raise CheckFailed(f"{len(kept)} blocks kept, expected {8 - k}")
    log(f"{name}: removed {', '.join(removed)}; blocks kept {len(kept)} of 8; params {params0:,} -> {params1:,} "
        f"(-{(1 - params1 / params0) * 100:.1f}%), GMACs {gmacs0:.3f} -> {gmacs1:.3f} "
        f"(-{(1 - gmacs1 / gmacs0) * 100:.1f}%)", logfile)
    sp.MODELS_DIR = MODELS_DIR
    sp.save_model(model, f"{name}_before")

    model.to(device)
    sp.set_seed(seed)
    train_loader, _, _ = sp.build_dataloaders(config)
    criterion = nn.CrossEntropyLoss(weight=sp.compute_class_weights(train_loader, sp.NUM_CLASSES, device))
    cpu_criterion = nn.CrossEntropyLoss(weight=criterion.weight.cpu())
    start = sp.metrics(sp.predict(model, batches, device), y, fp32_pred, mel, cpu_criterion)
    log(f"{name} before fine-tuning: val F1 {start['f1_macro']:.4f}, acc {start['accuracy']:.4f}, bal acc "
        f"{start['balanced_accuracy']:.4f}, mel recall {start['melanoma_recall']:.4f}", logfile)

    sp.set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = [{"epoch": 0, "lr": None, "train_loss": None, **{f"val_{k_}": v for k_, v in start.items()}}]
    best = None
    for epoch in range(1, EPOCHS + 1):
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
    sp.save_model(best[2], f"{name}_selected")
    sp.save_model(model, f"{name}_epoch10")

    epoch10 = {k_[4:]: v for k_, v in history[-1].items() if k_.startswith("val_")}
    control = control_metrics(seed)
    summary = {"run": name, "blocks_removed": removed, "blocks_kept": kept, "seed": seed,
               "params_before": params0, "params_after": params1, "gmacs_before": gmacs0, "gmacs_after": gmacs1,
               "param_reduction_pct": (1 - params1 / params0) * 100, "gmac_reduction_pct": (1 - gmacs1 / gmacs0) * 100,
               "before_finetuning": start, "selected_epoch": best[0], "selected": best[1], "epoch10": epoch10,
               "settings": settings}
    if control:
        summary["control"] = f"control_seed{seed} (pruning/structured/results)"
        summary["pruned_minus_control"] = {
            stage: {k_: summary[stage][k_] - control[stage][k_] for k_ in sp.SCORES} for stage in ("selected", "epoch10")}
    rows = [{"run": name, "stage": stage, **values} for stage, values in
            (("before_finetuning", start), (f"selected_epoch{best[0]}", best[1]), ("epoch10", epoch10))]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    summary["files"] = {s: str((MODELS_DIR / f"{name}_{s}.pt").relative_to(ROOT)) for s in ("before", "selected",
                                                                                           "epoch10")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"{name}: selected epoch {best[0]} (val F1 {best[1]['f1_macro']:.4f}); epoch 10 F1 {epoch10['f1_macro']:.4f}"
        + (f"; minus control_seed{seed}: {summary['pruned_minus_control']['selected']['f1_macro'] * 100:+.2f} "
           f"(selected), {summary['pruned_minus_control']['epoch10']['f1_macro'] * 100:+.2f} (epoch 10)"
           if control else "") + "; models saved", logfile)


def main(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise CheckFailed("no GPU")
    device = torch.device("cuda")
    hashes = {str(p): sp.sha256(p) for p in sp.PROTECTED}
    config = sp.load_config(sp.CONFIG)
    train_loader, val_loader, class_to_idx = sp.build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    weights = sp.compute_class_weights(train_loader, sp.NUM_CLASSES, "cpu")

    importance, order = rank_blocks(config, nn.CrossEntropyLoss(weight=weights), device)
    importance.to_csv(RESULTS / "block_importance.csv", index=False)
    log("block importance (class-weighted loss increase on 1,000 train images when the block alone is removed): "
        + ", ".join(f"{r.block} {r.loss_increase:+.4f}" for r in importance.iloc[1:].itertuples())
        + f"; removal order: {', '.join(order)}")

    batches = list(val_loader)  # decoded once; the test set is never loaded
    y = torch.cat([labels for _, labels in batches]).numpy()
    fp32_logits = sp.predict(sp.load_fp32(), batches, "cpu")
    mel = classes.index(sp.MELANOMA)
    settings = {
        "checkpoint": str(sp.CHECKPOINT.relative_to(ROOT)), "config": sp.CONFIG,
        "model": "torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), strict load",
        "pruning": "block (depth) pruning: the k least important identity-shortcut BasicBlocks replaced by nn.Identity",
        "removable_blocks": REMOVABLE,
        "importance": f"class-weighted cross-entropy increase on {RANKING_IMAGES} train images (seed {RANKING_SEED}, "
                      "eval preprocessing) when the block alone is removed; ranked once on the full model",
        "removal_order": order,
        "fine_tuning": {"epochs": EPOCHS, "optimizer": "Adam", "lr": LR, "schedule": "CosineAnnealingLR(T_max=10), "
                        "stepped per epoch", "batch_size": config.batch_size, "augmentation": config.augmentation,
                        "loss": "CrossEntropyLoss, balanced class weights from the train split"},
        "selection": "best validation macro F1 epoch (first maximum); epoch 10 also reported",
        "control": "pruning/structured/results/control_seed<seed> (same fine-tuning, unpruned), reused",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "sha256_protected": hashes}
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2))
    for seed in args.seeds:
        for k in args.blocks:
            run(k, seed, order, config, (y, batches, fp32_logits.argmax(1).numpy(), mel), device, settings)
    if {str(p): sp.sha256(p) for p in sp.PROTECTED} != hashes:
        raise CheckFailed("the FP32 checkpoint, the split or the config changed during the run")
    log("protected files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", required=True, help="number of blocks removed (1 to 5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    try:
        main(parser.parse_args())
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
