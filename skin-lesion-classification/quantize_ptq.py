"""
Post-training static INT8 quantization (PTQ) of the fine-tuned ResNet-18
========================================================================
Eager-mode quantization of torchvision's quantizable ResNet-18:
onednn engine, x86 qconfig (reduce_range=True); see src/ham10000/quantize.py.

1. Loads the FP32 state_dict, fuses conv/bn/relu, inserts observers.
2. Calibrates on images from the TRAIN split (deterministic eval transform).
3. Ablation picked on VALIDATION macro F1: calibration size x weight
   granularity (per-channel / per-tensor). The test split is never used.

Usage:
  python quantize_ptq.py          # ablation only -> results/ptq_int8/ptq_ablation.csv
  python quantize_ptq.py --save   # rebuild the selected setting -> results/ptq_int8/resnet18_ptq_int8.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import torch.ao.quantization as tq

from ham10000.data import Ham10000Dataset, lesion_split
from ham10000.quantize import (QCONFIG_BACKEND, check_int8, cpu_instruction_support, engine_info, eval_split_loader,
                               load_quantizable_fp32, predict, qconfig_info, set_engine, val_scores)
from ham10000.utils import load_config

FP32_PATH = "results/strong_cosine25_resnet18/resnet18_baseline.pth"
CONFIG_PATH = "configs/strong_cosine25.yaml"
OUT_DIR = Path("results/ptq_int8")
CALIB_SIZES = [128, 512, 1024, 2048]
DEFAULT_CALIB_SIZE = 1024
CALIB_SEED = 42


def qconfig_for(weights):
    default = tq.get_default_qconfig(QCONFIG_BACKEND)  # HistogramObserver(reduce_range=True), per-channel symmetric weights
    if weights == "per_channel":
        return default
    return tq.QConfig(activation=default.activation, weight=tq.default_weight_observer)  # per-tensor symmetric qint8


def quantize(config, calib_loader, weights):
    model = load_quantizable_fp32(FP32_PATH, config.num_classes).eval()
    model.fuse_model(is_qat=False)
    model.qconfig = qconfig_for(weights)
    tq.prepare(model, inplace=True)
    with torch.inference_mode():
        for x, _ in calib_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model


def main():
    parser = argparse.ArgumentParser(description="PTQ ablation / save of the selected INT8 model")
    parser.add_argument("--save", action="store_true", help="rebuild the setting selected in ptq_ablation.csv and save it")
    args = parser.parse_args()

    set_engine()
    torch.manual_seed(CALIB_SEED)
    config = load_config(CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = Ham10000Dataset(config.metadata_csv, config.data_dir)
    train_idx, _, _ = lesion_split(ds.data, config)
    class_to_idx = ds.class_to_idx
    # One fixed shuffle of the train split; each calibration set is a prefix of it, so smaller sets are nested in larger ones
    order = np.random.default_rng(CALIB_SEED).permutation(len(train_idx))
    ds.data.loc[[train_idx[i] for i in order[:max(CALIB_SIZES)]], ["image_id", "lesion_id", "dx"]].to_csv(
        OUT_DIR / "calibration_images.csv", index=False)

    val_loader = eval_split_loader(config, "val")
    fp32 = load_quantizable_fp32(FP32_PATH, config.num_classes)
    y_val, fp32_pred = predict(fp32, val_loader)
    fp32_scores = val_scores(y_val, fp32_pred, class_to_idx)
    print(f"FP32 on val ({len(y_val)} images): {fp32_scores}", flush=True)

    if not args.save:
        rows = []
        for weights in ["per_channel", "per_tensor"]:
            for n in CALIB_SIZES:
                t0 = time.time()
                calib_loader = eval_split_loader(config, "train", indices=order[:n].tolist(), batch_size=32)
                _, pred = predict(quantize(config, calib_loader, weights), val_loader)
                scores = val_scores(y_val, pred, class_to_idx)
                rows.append({"weights": weights, "calib_images": n, **{f"val_{k}": v for k, v in scores.items()},
                             "val_agreement_with_fp32": float((pred == fp32_pred).mean()), "seconds": time.time() - t0})
                print(rows[-1], flush=True)
        ablation = pd.DataFrame(rows)
        # Best validation macro F1; ties go to per-channel weights, then to the default calibration size
        rank = list(zip(-ablation["val_macro_f1"], ablation["weights"] != "per_channel",
                        ablation["calib_images"] != DEFAULT_CALIB_SIZE, ablation["calib_images"]))
        best = min(range(len(rank)), key=lambda i: rank[i])
        ablation["selected"] = [i == best for i in range(len(ablation))]
        ablation["qconfig"] = f"{QCONFIG_BACKEND} defaults (reduce_range=True)"
        ablation.to_csv(OUT_DIR / "ptq_ablation.csv", index=False)
        print(ablation.drop(columns=["seconds", "qconfig"]).round(4).to_string(index=False))
        return

    ablation = pd.read_csv(OUT_DIR / "ptq_ablation.csv")
    best = ablation[ablation["selected"]].iloc[0]
    weights, n = best["weights"], int(best["calib_images"])
    model = quantize(config, eval_split_loader(config, "train", indices=order[:n].tolist(), batch_size=32), weights)
    _, pred = predict(model, val_loader)
    scores = val_scores(y_val, pred, class_to_idx)
    # Calibration is deterministic, so the rebuilt model must reproduce the ablation's validation score
    assert abs(scores["macro_f1"] - best["val_macro_f1"]) < 1e-9, (scores, best.to_dict())

    out_path = OUT_DIR / "resnet18_ptq_int8.pt"
    torch.jit.save(torch.jit.script(model), str(out_path))
    checks = check_int8(model, str(out_path), config.num_classes, config.image_size)

    settings = {
        "fp32_state_dict": FP32_PATH,
        "config": CONFIG_PATH,
        **engine_info(),
        "qconfig_backend": QCONFIG_BACKEND,
        **qconfig_info(qconfig_for(weights)),
        "cpu_instruction_support": cpu_instruction_support(),
        "api": "eager mode, torchvision.models.quantization.resnet18(weights=None, quantize=False), fuse_model(is_qat=False), prepare, convert",
        "selected": {"weights": weights, "calib_images": n},
        "selection_rule": "max validation macro F1; ties -> per_channel, then calibration size 1024",
        "val_scores": scores,
        "fp32_val_scores": fp32_scores,
        "calibration": {"split": "train", "transform": "eval (resize + normalize, no augmentation)",
                        "seed": CALIB_SEED, "nested_prefixes_of": "results/ptq_int8/calibration_images.csv"},
        "ablation_files": {"x86_qconfig": "ptq_ablation.csv",
                           "onednn_qconfig_reduce_range_false": "ptq_ablation_onednn_qconfig.csv"},
        "checks": checks,
        "torch_version": torch.__version__,
    }
    json.dump(settings, open(OUT_DIR / "settings.json", "w"), indent=2)
    print("Saved", out_path, "| selected:", settings["selected"], "| val:", scores, "| checks:", checks)


if __name__ == "__main__":
    main()
