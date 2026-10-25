"""
EXP-001: ResNet18 HAM10000 Baseline — Complete Runner
=====================================================
This script orchestrates training, evaluation, and Grad-CAM
using the original repository code, adding only recording/saving
of per-epoch metrics and experiment outputs.

NO changes to: model architecture, augmentation, loss, optimizer,
learning rate, batch size, split, or number of epochs.
"""
import sys
import os
import json
import time
import random
from pathlib import Path
from datetime import datetime

# Ensure src is importable
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ham10000.utils import load_config
from ham10000.data import build_dataloaders, compute_class_weights
from ham10000.models import build_model
from ham10000.train import set_seed, train_one_epoch, validate

# ── Configuration ────────────────────────────────────────────
CONFIG_PATH = "configs/baseline.yaml"
RESULTS_DIR = Path("results/baseline_resnet18")
GRADCAM_DIR = RESULTS_DIR / "gradcam"

def main():
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)

    # ── 1. Record environment ────────────────────────────────
    import torchvision
    import sklearn
    import pandas
    import matplotlib

    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 0) if torch.cuda.is_available() else None,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pandas.__version__,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "platform": sys.platform,
    }
    print("=" * 60)
    print("EXP-001 ENVIRONMENT")
    print("=" * 60)
    for k, v in env_info.items():
        print(f"  {k}: {v}")

    with open(RESULTS_DIR / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # ── 2. Build dataloaders ─────────────────────────────────
    print("\n" + "=" * 60)
    print("BUILDING DATALOADERS")
    print("=" * 60)
    train_loader, val_loader, class_to_idx = build_dataloaders(config)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"  Train samples: {n_train}")
    print(f"  Val samples:   {n_val}")
    print(f"  Classes:       {class_to_idx}")

    # ── 3. Setup model, loss, optimizer (original code) ──────
    print("\n" + "=" * 60)
    print("MODEL SETUP")
    print("=" * 60)
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = build_model(config.num_classes).to(device)
    print(f"  Model: {config.model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.use_class_weights:
        class_weights = compute_class_weights(train_loader, config.num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Class weights: {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(f"  Optimizer: Adam, lr={config.learning_rate}")
    print(f"  Loss: CrossEntropyLoss (weighted={config.use_class_weights})")

    # ── 4. Training loop with history recording ──────────────
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    checkpoint_path = RESULTS_DIR / "best_model.pth"
    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    total_train_start = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # Use the original train_one_epoch and validate functions
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Also compute training accuracy for the record
        _, train_acc = validate(model, train_loader, criterion, device)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "epoch_duration_sec": round(epoch_duration, 1),
        }
        history.append(epoch_record)

        print(f'Epoch {epoch+1}/{config.num_epochs} | '
              f'train_loss={train_loss:.4f} | '
              f'val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | '
              f'train_acc={train_acc:.4f} | '
              f'time={epoch_duration:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(checkpoint_path))
            # Also save to repo root for evaluate.py/gradcam.py compatibility
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  -> Best model saved (val_loss={val_loss:.4f})')

    total_train_time = time.time() - total_train_start

    print(f"\nTraining complete in {total_train_time:.1f}s")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    # Save training history
    training_summary = {
        "experiment_id": "EXP-001",
        "total_training_time_sec": round(total_train_time, 1),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "history": history,
    }
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    # ── 5. Evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION (using best checkpoint)")
    print("=" * 60)

    # Load best checkpoint
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    model.eval()

    from ham10000.evaluate import get_predictions, save_confusion_matrix
    from sklearn.metrics import classification_report

    all_preds, all_labels = get_predictions(model, val_loader, device)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Classification report
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_text)

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Save classification report
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"EXP-001 Classification Report\n")
        f.write(f"Checkpoint: best_model.pth (epoch {best_epoch})\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(report_text)

    # Save report as JSON too
    with open(RESULTS_DIR / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # Confusion matrices
    save_confusion_matrix(all_labels, all_preds, class_names,
                          str(RESULTS_DIR / "confusion_matrix.png"))
    save_confusion_matrix(all_labels, all_preds, class_names,
                          str(RESULTS_DIR / "confusion_matrix_normalized.png"),
                          normalize="true")

    # ── 6. Grad-CAM ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRAD-CAM")
    print("=" * 60)

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from ham10000.gradcam import prepare_image, save_gradcam, collect_samples

        val_subset = val_loader.dataset
        correct, incorrect = collect_samples(model, val_subset, device, n_each=5)
        print(f"Found {len(correct)} correct and {len(incorrect)} incorrect samples")

        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        for i, sample in enumerate(correct, start=1):
            rgb_img, input_tensor = prepare_image(sample['path'])
            input_tensor = input_tensor.to(device)
            targets = [ClassifierOutputTarget(sample['pred_idx'])]
            grayscale_cam = cam(input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            true_cls = idx_to_class[sample['true_idx']]
            pred_cls = idx_to_class[sample['pred_idx']]
            out_path = str(GRADCAM_DIR / f"gradcam_correct_{i}.png")
            save_gradcam(rgb_img, visualization, true_cls, pred_cls, out_path)

        for i, sample in enumerate(incorrect, start=1):
            rgb_img, input_tensor = prepare_image(sample['path'])
            input_tensor = input_tensor.to(device)
            targets = [ClassifierOutputTarget(sample['pred_idx'])]
            grayscale_cam = cam(input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            true_cls = idx_to_class[sample['true_idx']]
            pred_cls = idx_to_class[sample['pred_idx']]
            out_path = str(GRADCAM_DIR / f"gradcam_incorrect_{i}.png")
            save_gradcam(rgb_img, visualization, true_cls, pred_cls, out_path)

        print("Grad-CAM complete.")
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()

    # ── 7. Final summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXP-001 COMPLETE")
    print("=" * 60)

    accuracy = report_dict.get("accuracy", 0)
    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})

    final_metrics = {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro.get("precision", 0), 4),
        "macro_recall": round(macro.get("recall", 0), 4),
        "macro_f1": round(macro.get("f1-score", 0), 4),
        "weighted_precision": round(weighted.get("precision", 0), 4),
        "weighted_recall": round(weighted.get("recall", 0), 4),
        "weighted_f1": round(weighted.get("f1-score", 0), 4),
    }

    print(f"\n  Accuracy:         {final_metrics['accuracy']}")
    print(f"  Macro Precision:  {final_metrics['macro_precision']}")
    print(f"  Macro Recall:     {final_metrics['macro_recall']}")
    print(f"  Macro F1:         {final_metrics['macro_f1']}")
    print(f"  Weighted F1:      {final_metrics['weighted_f1']}")
    print(f"  Best Epoch:       {best_epoch}")
    print(f"  Training Time:    {total_train_time:.1f}s")
    print(f"  Checkpoint:       {checkpoint_path}")

    # Per-class
    print(f"\n  Per-class metrics:")
    for cls in class_names:
        c = report_dict.get(cls, {})
        print(f"    {cls:>6s}:  P={c.get('precision',0):.2f}  R={c.get('recall',0):.2f}  F1={c.get('f1-score',0):.2f}  support={c.get('support',0)}")

    # Save final metrics
    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump({
            "experiment_id": "EXP-001",
            "final_metrics": final_metrics,
            "per_class": {cls: report_dict[cls] for cls in class_names},
            "best_epoch": best_epoch,
            "total_training_time_sec": round(total_train_time, 1),
            "checkpoint_path": str(checkpoint_path),
        }, f, indent=2)

    # Save experiment config
    with open(RESULTS_DIR / "experiment_config.json", "w") as f:
        json.dump({
            "seed": config.seed,
            "model_name": config.model_name,
            "num_classes": config.num_classes,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "use_class_weights": config.use_class_weights,
            "use_augmentation": config.use_augmentation,
            "train_test_split": config.train_test_split,
            "metadata_csv": config.metadata_csv,
            "data_dir": config.data_dir,
        }, f, indent=2)

    print(f"\nAll outputs saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
