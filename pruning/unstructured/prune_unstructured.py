"""Unstructured (weight) pruning of the fine-tuned ResNet-18, then fine-tuning. Unstructured only, FP32: no structured
pruning, no quantization.

Every run starts from the same FP32 checkpoint, loaded (strict=True) into
torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7).

Pruning: torch.nn.utils.prune.global_unstructured with L1Unstructured over the weights of every Conv2d and Linear
(conv1, all residual-block and downsample convs, fc): the given fraction of those weights with the smallest |w|,
ranked together across layers (so layers end up with different sparsity). Biases and BatchNorm are not touched. The
achieved sparsity is recorded globally and per layer.

Stage 1, right after pruning (no training): full validation metrics.
Fine-tuning, identical to the structured-pruning runs (pruning/structured/filter/prune.py, whose functions are reused):
class-weighted cross-entropy (train-split weights), strong augmentation, Adam, batch 32, lr 1e-4, cosine annealing over
10 epochs (stepped per epoch), seed given. The pruning masks stay active (weight = weight_orig x mask), so pruned weights
stay exactly zero; prune.remove then makes the zeros permanent. Selected model = best validation macro F1 epoch (first
maximum); epoch 10 is also reported. The FP32 control is reused from pruning/structured/filter (control_seed<seed>), not
retrained.

Checks (the run stops if one fails): achieved sparsity = target (rounding), pruned positions still zero and the
zero count unchanged after fine-tuning, after fuse_model() (zeros x BatchNorm scale stay zero) and in the TorchScript
export; the FP32 checkpoint, the split and the config are hashed before and after.

Data roles: train = fine-tuning; validation = all choices; the test set is never loaded.

Writes, for a run named sparse<pct>_seed<seed>:
  pruning/unstructured/models/<run>_{before,selected,epoch10}.{pt,ts}  whole model (torch.save, load with
      weights_only=False) and TorchScript (traced, unfused). before = stage 1 (pruned, not fine-tuned). Not in git.
  pruning/unstructured/results/<run>/  layers.csv (per-layer sparsity), summary.json (sparsity, sizes, metrics per
      stage, pruned - control, settings), training_history.csv, metrics.csv (one row per stage), run.log

Usage: python pruning/unstructured/prune_unstructured.py --sparsities 50 [30 70 90] --seeds 42 [1 2]
"""
import argparse
import copy
import gzip
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("structured_prune", ROOT / "pruning" / "structured" / "filter" / "prune.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)  # helpers, data and the fine-tuning loop of the structured runs (also chdir + sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.utils.prune as tprune  # noqa: E402

MODELS_DIR, RESULTS = HERE / "models", HERE / "results"
CONTROL_RESULTS = ROOT / "pruning" / "structured" / "filter" / "results"
EPOCHS, LR = sp.EPOCHS, sp.LR
CheckFailed, log = sp.CheckFailed, sp.log


def prunable(model):
    """(name, module) of every Conv2d and Linear, in model order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def weight(module):
    """The effective weight: weight_orig x mask while pruning is attached, the plain weight after prune.remove."""
    return module.weight.detach()


def sparsity_table(model):
    rows = []
    for name, m in prunable(model):
        w = weight(m)
        rows.append({"layer": name, "type": type(m).__name__, "shape": "x".join(map(str, w.shape)),
                     "weights": w.numel(), "zeros": int((w == 0).sum()), "nonzero": int((w != 0).sum())})
    table = pd.DataFrame(rows)
    table["sparsity_pct"] = table["zeros"] / table["weights"] * 100
    return table


def counts(model):
    """Zeros in the prunable weights; total and non-zero parameters (all parameters, biases and BatchNorm included)."""
    layers = prunable(model)
    zeros = sum(int((weight(m) == 0).sum()) for _, m in layers)
    weights = sum(weight(m).numel() for _, m in layers)
    others = [p for m in model.modules() if not isinstance(m, (nn.Conv2d, nn.Linear))
              for p in m.parameters(recurse=False)]  # BatchNorm weights and biases
    others += [m.bias for _, m in layers if m.bias is not None]
    total = weights + sum(p.numel() for p in others)
    nonzero = weights - zeros + sum(int((p != 0).sum()) for p in others)
    return {"prunable_weights": weights, "zeros": zeros, "global_sparsity_pct": zeros / weights * 100,
            "params_total": total, "params_nonzero": nonzero}


def sizes(model):
    """Bytes of the state_dict as saved by torch.save (dense FP32), and gzip-compressed (level 9)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    raw = buffer.getvalue()
    return {"state_dict_dense_mb": len(raw) / 1e6, "state_dict_gzip_mb": len(gzip.compress(raw, 9)) / 1e6}


def prune_global(model, sparsity):
    tprune.global_unstructured([(m, "weight") for _, m in prunable(model)], pruning_method=tprune.L1Unstructured,
                               amount=sparsity)


def made_permanent(model):
    """A copy with the masks applied for good, as prune.remove does (weight = weight_orig x mask): plain weights with
    exact zeros, no hooks. Built from the state_dict, because a module with masks attached cannot be deep-copied."""
    state = model.state_dict()
    plain = {}
    for key, value in state.items():
        if key.endswith("_orig"):
            plain[key[:-len("_orig")]] = (value * state[key[:-len("_orig")] + "_mask"]).detach().cpu().clone()
        elif not key.endswith("_mask"):
            plain[key] = value.detach().cpu().clone()
    copy_ = sp.resnet18(weights=None, quantize=False, num_classes=sp.NUM_CLASSES)
    copy_.load_state_dict(plain, strict=True)
    return copy_.eval()


def remove_masks(model):
    """prune.remove on the training model itself (zeros made permanent in place)."""
    for _, m in prunable(model):
        if tprune.is_pruned(m):
            tprune.remove(m, "weight")
    return model


def check_zeros(model, expected_mask, where):
    """Pruned positions are exactly zero and no other weight is (fine-tuning cannot create or remove zeros)."""
    for name, m in prunable(model):
        w = weight(m).cpu()
        if not torch.equal(w == 0, expected_mask[name]):
            raise CheckFailed(f"{where}: zeros of {name} differ from the pruning mask "
                              f"({int((w == 0).sum())} zeros, mask {int(expected_mask[name].sum())})")


def zeros_fused_and_torchscript(model, stem):
    """Zero count in the prunable weights after fuse_model() and in the saved TorchScript file."""
    fused = copy.deepcopy(model).eval()
    fused.fuse_model()
    fused_zeros = sum(int((m.weight == 0).sum()) for _, m in prunable(fused))
    ts = torch.jit.load(str(MODELS_DIR / f"{stem}.ts"))
    names = {f"{n}.weight" for n, _ in prunable(model)}
    ts_zeros = sum(int((p == 0).sum()) for n, p in ts.named_parameters() if n in names)
    return fused_zeros, ts_zeros


def save(model, stem):
    sp.MODELS_DIR = MODELS_DIR  # sp.save_model writes to its MODELS_DIR and checks the reload and TorchScript
    sp.save_model(model, stem)


def control_metrics(seed):
    path = CONTROL_RESULTS / f"control_seed{seed}" / "metrics.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return {"selected": table[table["stage"].str.startswith("selected")].iloc[0].to_dict(),
            "epoch10": table[table["stage"] == "epoch10"].iloc[0].to_dict()}


def run(pct, seed, config, val, device, settings):
    name = f"sparse{pct}_seed{seed}"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "run.log"
    logfile.write_text("")
    y, batches, fp32_pred, mel = val
    model = sp.load_fp32()
    dense = {**counts(model), **sizes(model)}

    # Stage 0 -> 1: prune
    prune_global(model, pct / 100)
    mask = {n: (weight(m) == 0).cpu().clone() for n, m in prunable(model)}
    layers = sparsity_table(model)
    stage1 = made_permanent(model)
    pruned = {**counts(stage1), **sizes(stage1)}
    expected = round(pct / 100 * pruned["prunable_weights"])
    if pruned["zeros"] != expected:
        raise CheckFailed(f"{pruned['zeros']} zeros, expected {expected} ({pct}%)")
    with torch.no_grad():
        if tuple(stage1(torch.zeros(1, 3, 224, 224)).shape) != (1, sp.NUM_CLASSES):
            raise CheckFailed("forward pass after pruning")
    log(f"{name}: {pruned['zeros']:,} of {pruned['prunable_weights']:,} weights zeroed "
        f"({pruned['global_sparsity_pct']:.2f}%), per layer {layers['sparsity_pct'].min():.1f}% "
        f"({layers.loc[layers['sparsity_pct'].idxmin(), 'layer']}) to {layers['sparsity_pct'].max():.1f}% "
        f"({layers.loc[layers['sparsity_pct'].idxmax(), 'layer']}); non-zero params {pruned['params_nonzero']:,} of "
        f"{pruned['params_total']:,}; state_dict {pruned['state_dict_dense_mb']:.2f} MB dense, "
        f"{pruned['state_dict_gzip_mb']:.2f} MB gzip (FP32 {dense['state_dict_gzip_mb']:.2f})", logfile)
    save(stage1, f"{name}_before")

    model.to(device)
    sp.set_seed(seed)
    train_loader, _, _ = sp.build_dataloaders(config)
    criterion = nn.CrossEntropyLoss(weight=sp.compute_class_weights(train_loader, sp.NUM_CLASSES, device))
    cpu_criterion = nn.CrossEntropyLoss(weight=criterion.weight.cpu())
    start = sp.metrics(sp.predict(model, batches, device), y, fp32_pred, mel, cpu_criterion)
    log(f"{name} before fine-tuning: val F1 {start['f1_macro']:.4f}, acc {start['accuracy']:.4f}, bal acc "
        f"{start['balanced_accuracy']:.4f}, mel recall {start['melanoma_recall']:.4f}", logfile)

    # Fine-tuning with the masks attached (pruned weights get no update: weight = weight_orig x mask)
    sp.set_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = [{"epoch": 0, "lr": None, "train_loss": None, **{f"val_{k}": v for k, v in start.items()}}]
    best = None
    for epoch in range(1, EPOCHS + 1):
        t0, lr = time.time(), optimizer.param_groups[0]["lr"]
        loss = sp.train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        m = sp.metrics(sp.predict(model, batches, device), y, fp32_pred, mel, cpu_criterion)
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss, **{f"val_{k}": v for k, v in m.items()},
                        "seconds": time.time() - t0})
        log(f"{name} epoch {epoch} lr {lr:.3g}: loss {loss:.4f}, val loss {m['val_loss']:.4f}, F1 {m['f1_macro']:.4f}, "
            f"acc {m['accuracy']:.4f}, bal acc {m['balanced_accuracy']:.4f}, mel {m['melanoma_recall']:.4f}", logfile)
        if best is None or m["f1_macro"] > best[1]["f1_macro"]:
            best = (epoch, m, made_permanent(model))
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    # Sparsity checks: after fine-tuning, after fuse_model(), in the TorchScript export
    checks = {}
    for stage, final in (("before", stage1), ("selected", best[2]), ("epoch10", made_permanent(model))):
        stem = f"{name}_{stage}"
        check_zeros(final, mask, f"{stem} after fine-tuning" if stage != "before" else stem)
        if stage != "before":
            save(final, stem)
        fused_zeros, ts_zeros = zeros_fused_and_torchscript(final, stem)
        checks[stage] = {"zeros": counts(final)["zeros"], "zeros_after_fuse_model": fused_zeros,
                         "zeros_in_torchscript": ts_zeros}
        if not fused_zeros == ts_zeros == pruned["zeros"]:
            raise CheckFailed(f"{stem}: zeros {pruned['zeros']}, after fuse {fused_zeros}, TorchScript {ts_zeros}")
    log(f"{name}: sparsity unchanged after fine-tuning, fuse_model() and TorchScript ({pruned['zeros']:,} zeros)",
        logfile)

    epoch10 = {k[4:]: v for k, v in history[-1].items() if k.startswith("val_")}
    layers["sparsity_after_finetuning_pct"] = sparsity_table(best[2])["sparsity_pct"].values
    layers.to_csv(out / "layers.csv", index=False)
    control = control_metrics(seed)
    summary = {"run": name, "target_sparsity_pct": pct, "seed": seed, "dense_fp32": dense, "pruned": pruned,
               "sparsity_checks": checks, "before_finetuning": start, "selected_epoch": best[0],
               "selected": best[1], "epoch10": epoch10, "settings": settings}
    if control:
        summary["control"] = f"control_seed{seed} (pruning/structured/filter/results)"
        summary["pruned_minus_control"] = {
            stage: {k: summary[stage][k] - control[stage][k] for k in sp.SCORES} for stage in ("selected", "epoch10")}
    rows = [{"run": name, "stage": stage, **values} for stage, values in
            (("before_finetuning", start), (f"selected_epoch{best[0]}", best[1]), ("epoch10", epoch10))]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    summary["files"] = {s: str((MODELS_DIR / f"{name}_{s}.pt").relative_to(ROOT))
                        for s in ("before", "selected", "epoch10")}
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
    _, val_loader, class_to_idx = sp.build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    batches = list(val_loader)  # decoded once; the test set is never loaded
    y = torch.cat([labels for _, labels in batches]).numpy()
    fp32_logits = sp.predict(sp.load_fp32(), batches, "cpu")
    mel = classes.index(sp.MELANOMA)
    fp32_metrics = sp.metrics(fp32_logits, y, fp32_logits.argmax(1).numpy(), mel)
    log(f"validation {len(y)} images; FP32 val F1 {fp32_metrics['f1_macro']:.4f}, acc {fp32_metrics['accuracy']:.4f}")
    (RESULTS / "fp32_validation.json").write_text(json.dumps(fp32_metrics, indent=2, default=float))
    settings = {
        "checkpoint": str(sp.CHECKPOINT.relative_to(ROOT)), "config": sp.CONFIG,
        "model": "torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=7), strict load",
        "pruning": "torch.nn.utils.prune.global_unstructured, L1Unstructured, over the weights of every Conv2d and "
                   "Linear (biases and BatchNorm untouched); masks kept during fine-tuning, then prune.remove",
        "fine_tuning": {"epochs": EPOCHS, "optimizer": "Adam", "lr": LR, "schedule": "CosineAnnealingLR(T_max=10), "
                        "stepped per epoch", "batch_size": config.batch_size, "augmentation": config.augmentation,
                        "loss": "CrossEntropyLoss, balanced class weights from the train split"},
        "selection": "best validation macro F1 epoch (first maximum); epoch 10 also reported",
        "control": "pruning/structured/filter/results/control_seed<seed> (same fine-tuning, unpruned), reused",
        "gzip": "gzip level 9 of the torch.save(state_dict) bytes",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "sha256_protected": hashes}
    (RESULTS / "settings.json").write_text(json.dumps(settings, indent=2))
    for seed in args.seeds:
        for pct in args.sparsities:
            run(pct, seed, config, (y, batches, fp32_logits.argmax(1).numpy(), mel), device, settings)
    if {str(p): sp.sha256(p) for p in sp.PROTECTED} != hashes:
        raise CheckFailed("the FP32 checkpoint, the split or the config changed during the run")
    log("protected files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsities", type=int, nargs="+", required=True, help="target sparsity in percent")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    try:
        main(parser.parse_args())
    except CheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
