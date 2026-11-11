"""Static Quantization-Aware Training (QAT) of fine-tuned ResNet-18, evaluated on VALIDATION split.

Mirrors the workflow and output conventions of quantization/ptq/weight_only/:
- Evaluates FP32 baseline on the validation split.
- Fuses Conv-BN-ReLU modules using torchvision's quantizable ResNet-18.
- Inserts fake-quantization observers for both weights (INT8) and activations (INT8).
- Fine-tunes/calibrates on the HAM10000 training set.
- Converts to true INT8 quantized model.
- Writes to results/:
    - resnet18_qat_int8.pt
    - file_sizes.csv
    - validation_results.csv
    - weight_errors.csv

Usage:
    python quantization/static_quantization_aware_training/static_qat.py [--epochs 2] [--lr 1e-4]
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "resnet18_finetunning"
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU execution

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, balanced_accuracy_score,
    f1_score, recall_score,
)
from torchvision.models import resnet18  # noqa: E402

import quantize as q  # noqa: E402
from ham10000.data import build_dataloaders, compute_class_weights  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = Path("results/strong_cosine25_resnet18/resnet18_baseline.pth")
OUT = HERE / "results"


@torch.inference_mode()
def predict(model, batches):
    logits, labels = zip(*[(model(images), y) for images, y in batches])
    return torch.cat(logits), torch.cat(labels)


def metrics_row(name, logits, labels, reference_logits, class_to_idx):
    preds = logits.argmax(1)
    y = labels.numpy()
    p = preds.numpy()
    per_class = recall_score(y, p, labels=list(class_to_idx.values()), average=None)
    return {
        "model": name,
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "melanoma_recall": per_class[class_to_idx["mel"]],
        **{f"recall_{c}": per_class[i] for c, i in class_to_idx.items()},
        "identical_to_fp32_pct": (preds == reference_logits.argmax(1)).float().mean().item() * 100,
        "softmax_mad": (logits.softmax(1) - reference_logits.softmax(1)).abs().mean().item(),
    }


def train_qat_loop(qat_model, train_loader, epochs, lr, class_weights=None):
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    total_batches = len(train_loader)
    print(f"Starting QAT fine-tuning for {epochs} epoch(s)...", flush=True)

    for epoch in range(epochs):
        qat_model.train()
        running_loss = 0.0
        start_t = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            outputs = qat_model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 50 == 0 or step == total_batches:
                elapsed = time.time() - start_t
                avg_l = running_loss / step
                print(f"  [Epoch {epoch+1}/{epochs}] Step {step}/{total_batches} - loss: {avg_l:.4f} ({elapsed:.1f}s)", flush=True)

        # In later epochs, freeze observers and batch norm statistics
        if epoch >= 1:
            qat_model.apply(torch.ao.quantization.disable_observer)
        if epoch >= 2:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    return qat_model


def compute_weight_errors(fp32_state, qat_model_int8):
    """Compare dequantized INT8 weights to original FP32 weights."""
    rows = []
    q_weights = q.get_quantized_weights_dict(qat_model_int8)

    for name, item in q_weights.items():
        if name in fp32_state:
            w = fp32_state[name].double()
            w_hat = item["dequantized"].double()
            error = (w - w_hat).abs()
            scale = item["scale"]
            scale_min = scale.min().item() if isinstance(scale, torch.Tensor) else float(scale)
            scale_max = scale.max().item() if isinstance(scale, torch.Tensor) else float(scale)

            rows.append({
                "variant": "static_qat_int8",
                "layer": name,
                "n_weights": w.numel(),
                "scale_min": scale_min,
                "scale_max": scale_max,
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
                "relative_error": ((w - w_hat).norm() / (w.norm() + 1e-12)).item(),
                "pct_q_zero": (item["qweight"].int_repr() == 0).double().mean().item() * 100,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Static QAT for ResNet-18")
    parser.add_argument("--epochs", type=int, default=2, help="Number of QAT fine-tuning epochs (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    checkpoint_sha = q.sha256_file(CHECKPOINT)
    split_sha = q.sha256_file(config.splits_csv)

    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, FP32 sha256 {checkpoint_sha}", flush=True)

    # 1. Load FP32 baseline
    fp32 = resnet18(weights=None, num_classes=config.num_classes)
    original_state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    fp32.load_state_dict(original_state, strict=True)
    fp32.eval()

    # 2. Prepare QAT model
    print("\n== Setting up QAT (fusing modules, inserting observers) ==", flush=True)
    qat_model = q.create_qat_resnet18(num_classes=config.num_classes, pretrained_state_dict=original_state)

    # 3. DataLoaders
    train_loader, val_loader, class_to_idx = build_dataloaders(config)
    class_weights = compute_class_weights(train_loader, config.num_classes, device="cpu") if config.use_class_weights else None

    # 4. Train QAT
    print(f"\n== Fine-tuning QAT ({len(train_loader)} batches/epoch) ==", flush=True)
    qat_model = train_qat_loop(qat_model, train_loader, epochs=args.epochs, lr=args.lr, class_weights=class_weights)

    # 5. Convert to INT8
    print("\n== Converting QAT model to INT8 ==", flush=True)
    qat_int8 = q.convert_qat_model(qat_model)

    qat_model_path = OUT / "resnet18_qat_int8.pt"
    # Save the whole quantized model wrapper using torch.jit or torch.save
    torch.save(qat_int8, qat_model_path)
    print(f"Saved INT8 model -> {qat_model_path}", flush=True)

    # 6. File size comparison
    size_rows = [
        {"file": CHECKPOINT.name, "variant": "FP32", "bytes": CHECKPOINT.stat().st_size},
        {"file": qat_model_path.name, "variant": "QAT_INT8", "bytes": qat_model_path.stat().st_size},
    ]
    sizes = pd.DataFrame(size_rows)
    sizes["MB"] = sizes["bytes"] / 1e6
    sizes["pct_of_fp32"] = sizes["bytes"] / sizes["bytes"].iloc[0] * 100
    sizes.to_csv(OUT / "file_sizes.csv", index=False)

    # 7. Validation evaluation
    print("\n== Running Validation ==", flush=True)
    batches = [(images, labels) for images, labels in val_loader]
    reference, labels = predict(fp32, batches)
    qat_logits, _ = predict(qat_int8, batches)

    val_rows = [
        metrics_row("FP32", reference, labels, reference, class_to_idx),
        metrics_row("QAT_INT8", qat_logits, labels, reference, class_to_idx),
    ]
    val_table = pd.DataFrame(val_rows)
    val_table.to_csv(OUT / "validation_results.csv", index=False)

    # 8. Weight errors
    weight_error_rows = compute_weight_errors(original_state, qat_int8)
    errors_df = pd.DataFrame(weight_error_rows)
    errors_df.to_csv(OUT / "weight_errors.csv", index=False)

    # Print summary
    pd.set_option("display.width", 250, "display.max_columns", None)
    print("\n== File sizes ==\n", sizes.round(2).to_string(index=False))
    print("\n== Validation Results ==\n", val_table.set_index("model").T.round(4).to_string())
    if not errors_df.empty:
        print("\n== Weight Error Summary (top 5 relative errors) ==\n",
              errors_df.sort_values("relative_error", ascending=False).head(5).to_string(index=False))

    print(f"\nAll results saved to {OUT}")


if __name__ == "__main__":
    main()

