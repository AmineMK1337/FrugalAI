"""Weight-only INT8 compression of the fine-tuned ResNet-18, evaluated on the VALIDATION split.

No dynamic or static quantization, no training, no calibration, no test set. Inference
stays FP32 after dequantizing, so no speedup is expected: only the file gets smaller.

Writes to results/ (next to this script): resnet18_w8_perchannel.pt, resnet18_w8_pertensor.pt,
weight_errors.csv, validation_results.csv, file_sizes.csv.

Usage (from anywhere):
  python quantization/ptq/weight_only/ptq_weight_only.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "resnet18_finetunning"
os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project
sys.path.insert(0, str(PROJECT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

import quantize as q  # noqa: E402
from ham10000.data import build_dataloaders  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = Path("results/strong_cosine25_resnet18/resnet18_baseline.pth")
OUT = HERE / "results"
VARIANTS = {"per_channel": OUT / "resnet18_w8_perchannel.pt", "per_tensor": OUT / "resnet18_w8_pertensor.pt"}


@torch.inference_mode()
def predict(model, batches):
    logits, labels = zip(*[(model(images), y) for images, y in batches])
    return torch.cat(logits), torch.cat(labels)


def metrics_row(name, logits, labels, reference_logits, class_to_idx):
    preds, y, p = logits.argmax(1), labels.numpy(), logits.argmax(1).numpy()
    per_class = recall_score(y, p, labels=list(class_to_idx.values()), average=None)
    return {
        "model": name,
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "melanoma_recall": per_class[class_to_idx["mel"]],
        **{f"recall_{c}": per_class[i] for c, i in class_to_idx.items()},
        "identical_to_fp32_pct": (preds == reference_logits.argmax(1)).float().mean().item() * 100,
        # mean over images and classes of |softmax INT8-weights - softmax FP32|
        "softmax_mad": (logits.softmax(1) - reference_logits.softmax(1)).abs().mean().item(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    checkpoint_sha = q.sha256_file(CHECKPOINT)
    split_sha = q.sha256_file(config.splits_csv)
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, FP32 sha256 {checkpoint_sha}", flush=True)

    fp32 = resnet18(weights=None, num_classes=config.num_classes)
    fp32.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    fp32.eval()
    original_state = {k: v.clone() for k, v in fp32.state_dict().items()}
    print(f"{len(q.weight_names(fp32))} Conv2d/Linear weights quantized; "
          f"{len(original_state) - len(q.weight_names(fp32))} other entries kept", flush=True)

    # Compress, save, check
    error_rows, size_rows = [], [{"file": CHECKPOINT.name, "variant": "FP32", "bytes": CHECKPOINT.stat().st_size}]
    for variant, path in VARIANTS.items():
        compressed = q.quantize_weights(fp32, per_channel=variant == "per_channel")
        q.check_format(compressed, original_state)
        error_rows += q.weight_error_rows(compressed, original_state)
        q.save_compressed(compressed, path)
        q.check_saved_file(path, compressed)
        size_rows.append({"file": path.name, "variant": variant, "bytes": path.stat().st_size})
        print(f"{variant}: format, scale/2 and reload checks PASS", flush=True)

    errors = pd.DataFrame(error_rows)
    errors.to_csv(OUT / "weight_errors.csv", index=False)
    sizes = pd.DataFrame(size_rows)
    sizes["MB"] = sizes["bytes"] / 1e6
    sizes["pct_of_fp32"] = sizes["bytes"] / sizes["bytes"].iloc[0] * 100
    sizes.to_csv(OUT / "file_sizes.csv", index=False)

    # Validation only: build_dataloaders returns no test loader. Decode once so all models get identical tensors.
    _, val_loader, class_to_idx = build_dataloaders(config)
    batches = [(images, labels) for images, labels in val_loader]
    if list(class_to_idx) != sorted(class_to_idx):
        raise q.WeightCheckFailed(f"unexpected class order {class_to_idx}")

    reference, labels = predict(fp32, batches)
    rows = [metrics_row("FP32", reference, labels, reference, class_to_idx)]
    for variant, path in VARIANTS.items():
        model = q.load_dequantized_resnet18(path, config.num_classes)
        logits, _ = predict(model, batches)
        rows.append(metrics_row(variant, logits, labels, reference, class_to_idx))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "validation_results.csv", index=False)

    if q.sha256_file(CHECKPOINT) != checkpoint_sha or q.sha256_file(config.splits_csv) != split_sha:
        raise q.WeightCheckFailed("the FP32 checkpoint or the split file changed during the run")
    print(f"FP32 checkpoint SHA-256 unchanged: {checkpoint_sha}\nSplit file unchanged. Validation images: {len(labels)}")

    pd.set_option("display.width", 250, "display.max_columns", None)
    print("\n== File sizes ==\n", sizes.round(2).to_string(index=False))
    print("\n== Validation ==\n", table.set_index("model").T.round(4).to_string())
    for variant in VARIANTS:
        part = errors[errors["variant"] == variant]
        print(f"\n== {variant}: largest relative errors ==\n", part.sort_values("relative_error", ascending=False).head(5)
              .drop(columns="variant").to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print(f"\n{variant}: overall worst max_abs_error {part['max_abs_error'].max():.4g} "
              f"({part.loc[part['max_abs_error'].idxmax(), 'layer']}); "
              f"max error / (scale/2) {part['max_error_over_half_scale'].max():.7f}; "
              f"all-zero channels {part['n_zero_channels'].sum()}")


if __name__ == "__main__":
    try:
        main()
    except q.WeightCheckFailed as error:
        sys.exit(f"CHECK FAILED, stopping: {error}")
