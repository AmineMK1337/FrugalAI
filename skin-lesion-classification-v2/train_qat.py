"""
Quantization-aware training (QAT) of the fine-tuned ResNet-18
==============================================================
Starts from the FP32 weights, fuses conv/bn/relu for QAT, inserts fake-quant
modules (onednn engine, x86 qconfig with reduce_range=True, see
src/ham10000/quantize.py) and fine-tunes on GPU with
the same loss and augmentation as the FP32 model:
  class-weighted cross-entropy (weights from the train split), strong augmentation,
  seed 42, Adam lr 1e-5 with a cosine schedule, 5 epochs.
BatchNorm statistics are frozen after epoch 3 and observers after epoch 4.
After every epoch a CPU copy is converted to INT8 and scored on the VALIDATION
split; the epoch with the best INT8 validation macro F1 is kept and saved as
TorchScript. The test split is never used.

--control runs the same 5 epochs (same loss, augmentation, seed, schedule and
BatchNorm freeze) on the plain FP32 model with no quantization, to separate the
effect of quantization from that of extra training. Its model is not tested.

Usage:
  python train_qat.py             # QAT  -> results/qat_int8/
  python train_qat.py --control   # FP32 -> results/fp32_control_5ep/
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import torch
import torch.ao.quantization as tq
import torch.nn as nn

from ham10000.data import build_dataloaders, compute_class_weights
from ham10000.models import build_model
from ham10000.quantize import (QCONFIG_BACKEND, check_int8, cpu_instruction_support, engine_info, eval_split_loader,
                               load_quantizable_fp32, predict, qconfig_info, set_engine, val_scores)
from ham10000.train import set_seed, train_one_epoch
from ham10000.utils import load_config

FP32_PATH = "results/strong_cosine25_resnet18/resnet18_baseline.pth"
CONFIG_PATH = "configs/strong_cosine25.yaml"
EPOCHS = 5
LR = 1e-5
SEED = 42
FREEZE_BN_AFTER = 3        # BatchNorm statistics frozen from epoch 4 on
FREEZE_OBSERVERS_AFTER = 4  # observers frozen from epoch 5 on (QAT only)


def main():
    parser = argparse.ArgumentParser(description="QAT of ResNet-18 on HAM10000 (or the FP32 control)")
    parser.add_argument("--control", action="store_true", help="FP32 fine-tuning without quantization")
    args = parser.parse_args()
    qat = not args.control
    out_dir = Path("results/qat_int8" if qat else "results/fp32_control_5ep")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_engine()
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda")

    if qat:
        model = load_quantizable_fp32(FP32_PATH, config.num_classes).train()
        model.fuse_model(is_qat=True)
        model.qconfig = tq.get_default_qat_qconfig(QCONFIG_BACKEND)  # same x86 qconfig family as PTQ (reduce_range=True)
        tq.prepare_qat(model, inplace=True)
    else:
        model = build_model(config.num_classes, config.model_name)
        model.load_state_dict(torch.load(FP32_PATH, map_location="cpu", weights_only=True))
        bn_frozen = {"on": False}
        # train_one_epoch calls model.train(); this hook puts BatchNorm back in eval mode once frozen
        def keep_bn_frozen(module, _inputs):
            if bn_frozen["on"]:
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        model.register_forward_pre_hook(keep_bn_frozen)
    model.to(device)

    train_loader, _, class_to_idx = build_dataloaders(config)  # strong augmentation from the config
    val_loader = eval_split_loader(config, "val")
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, config.num_classes, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    set_seed(SEED)  # just before training, so QAT and control see the same batch order
    history, best = [], {"macro_f1": -1.0}
    model_path = out_dir / ("resnet18_qat_int8.pt" if qat else "resnet18_control.pth")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        if epoch > FREEZE_BN_AFTER:
            if qat:
                model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
            else:
                bn_frozen["on"] = True
        if qat and epoch > FREEZE_OBSERVERS_AFTER:
            model.apply(tq.disable_observer)

        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if qat:
            eval_model = tq.convert(copy.deepcopy(model).cpu().eval(), inplace=False)
            y, pred = predict(eval_model, val_loader, "cpu")
        else:
            eval_model = model
            y, pred = predict(model, val_loader, device)
        scores = val_scores(y, pred, class_to_idx)
        record = {"epoch": epoch, "lr": lr, "train_loss": train_loss,
                  **{f"val_{k}": v for k, v in scores.items()},
                  "bn_frozen": epoch > FREEZE_BN_AFTER,
                  "observers_frozen": qat and epoch > FREEZE_OBSERVERS_AFTER,
                  "seconds": time.time() - t0}
        history.append(record)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"{'INT8' if qat else 'FP32'} val acc={scores['accuracy']:.4f} macro_f1={scores['macro_f1']:.4f} "
              f"mel_recall={scores['mel_recall']:.4f} | lr={lr:.2e} | {record['seconds']:.0f}s", flush=True)

        if scores["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **scores}
            if qat:
                torch.jit.save(torch.jit.script(eval_model), str(model_path))
                best_int8 = eval_model
            else:
                torch.save(model.state_dict(), model_path)
            print(f"  -> best so far, saved {model_path}", flush=True)

    summary = {"mode": "qat" if qat else "fp32_control", "best_epoch": best["epoch"], "best_val": best,
               "total_seconds": time.time() - t_start, "history": history}
    json.dump(summary, open(out_dir / "training_history.json", "w"), indent=2)

    settings = {
        "fp32_state_dict": FP32_PATH,
        "config": CONFIG_PATH,
        "mode": summary["mode"],
        "epochs": EPOCHS, "optimizer": "Adam", "lr": LR, "lr_schedule": "cosine (T_max=5)", "seed": SEED,
        "loss": "CrossEntropyLoss with balanced class weights from the train split",
        "augmentation": config.augmentation, "batch_size": config.batch_size,
        "freeze_bn_stats_after_epoch": FREEZE_BN_AFTER,
        "selection": "epoch with best validation macro F1" + (" of the converted INT8 model" if qat else ""),
        "best_epoch": best["epoch"],
        "torch_version": torch.__version__,
    }
    if qat:
        settings.update({
            **engine_info(),
            "qconfig_backend": QCONFIG_BACKEND,
            **qconfig_info(tq.get_default_qat_qconfig(QCONFIG_BACKEND)),
            "cpu_instruction_support": cpu_instruction_support(),
            "api": "eager mode, torchvision.models.quantization.resnet18(weights=None, quantize=False), fuse_model(is_qat=True), prepare_qat, convert",
            "freeze_observers_after_epoch": FREEZE_OBSERVERS_AFTER,
            "checks": check_int8(best_int8, str(model_path), config.num_classes, config.image_size),
        })
    json.dump(settings, open(out_dir / "settings.json", "w"), indent=2)
    print(f"\nBest epoch {best['epoch']}: {best}")
    if qat:
        print("checks:", settings["checks"])


if __name__ == "__main__":
    main()
