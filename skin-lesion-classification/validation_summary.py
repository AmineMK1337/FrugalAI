"""
Validation-split summary of FP32, INT8-PTQ, INT8-QAT and the FP32 control
=========================================================================
Run before any test evaluation. All four models are scored on CPU on the
validation split (the split used for every PTQ / QAT choice).

Usage:
  python validation_summary.py   # -> results/validation_summary.csv
"""
import sys

sys.path.insert(0, "src")

import pandas as pd
import torch

from ham10000.data import Ham10000Dataset
from ham10000.models import build_model
from ham10000.quantize import eval_split_loader, predict, set_engine, val_scores
from ham10000.utils import load_config

CONFIG_PATH = "configs/strong_cosine25.yaml"
MODELS = {
    "FP32": "results/strong_cosine25_resnet18/resnet18_baseline.pth",
    "INT8-PTQ": "results/ptq_int8/resnet18_ptq_int8.pt",
    "INT8-QAT": "results/qat_int8/resnet18_qat_int8.pt",
    "FP32 control (+5 epochs, no quantization)": "results/fp32_control_5ep/resnet18_control.pth",
}


def load(path, config):
    if path.endswith(".pt"):
        return torch.jit.load(path, map_location="cpu")
    model = build_model(config.num_classes, config.model_name)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def main():
    set_engine()
    config = load_config(CONFIG_PATH)
    class_to_idx = Ham10000Dataset(config.metadata_csv, config.data_dir).class_to_idx
    loader = eval_split_loader(config, "val")

    preds, rows = {}, []
    for name, path in MODELS.items():
        y, preds[name] = predict(load(path, config), loader)
        rows.append({"model": name, **val_scores(y, preds[name], class_to_idx),
                     "identical_to_fp32_pct": float((preds[name] == preds["FP32"]).mean() * 100)})
        print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.to_csv("results/validation_summary.csv", index=False)
    print(f"\nValidation split: {len(y)} images")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
