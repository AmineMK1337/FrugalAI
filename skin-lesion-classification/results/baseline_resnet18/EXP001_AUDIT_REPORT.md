# EXP-001 Integrity Audit

## 1. Overall Status

PASS

The EXP-001 baseline experiment was executed fully according to the frozen protocol without modifications. The outputs, metrics, and checkpoints were generated successfully and properly evaluated on the non-overlapping validation set.

## 2. Dataset Integrity

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Total Images | 10,015 | 10,015 | PASS |
| Metadata Rows | 10,015 | 10,015 | PASS |
| Missing Images | 0 | 0 | PASS |
| Classes Detected | 7 | 7 | PASS |
| Train Samples | 7,974 | 7,974 | PASS |
| Validation Samples | 2,041 | 2,041 | PASS |
| Zero Lesion Overlap | 0 overlap | 0 overlap | PASS |
| Class Mapping Correct | 0=akiec...6=vasc | 0=akiec...6=vasc | PASS |

## 3. Split Integrity

The dataset split uses lesion-level separation implemented in `src/ham10000/data.py` (line 94-104) via `sklearn.model_selection.train_test_split` on unique lesion IDs with `random_state=42`. Validation set samples (1,494 unique lesions, 2,041 images) are completely disjoint from the training set (5,976 unique lesions, 7,974 images) at the lesion level. No test data was exposed during validation.

## 4. Class Weight Integrity

Class weights are computed correctly in `src/ham10000/data.py` via `compute_class_weights`, which explicitly extracts only the training split subset labels (`subset.indices`) to prevent information leakage from the validation set.

## 5. Model Integrity

The ResNet18 model is instantiated in `src/ham10000/models.py` with `weights="IMAGENET1K_V1"`. The classifier layer is explicitly replaced with `nn.Linear(num_features, 7)` for full fine-tuning on the 7 target classes. No quantum layers, attention modules, or other hidden components were added.

## 6. Training Protocol Integrity

The protocol accurately followed the frozen rules:
* Optimizer: Adam
* Learning Rate: 0.0001
* Batch Size: 32
* Epochs: 10
* Seed: 42
* Loss: Weighted CrossEntropyLoss
* W&B: Disabled
No unexpected schedulers, TTA, PCA, MixUp, or ensemble methods were used.

## 7. Validation Step Investigation

The earlier anomaly showing a 250-step validation progress bar is fully explained by the `run_exp001.py` implementation. During each epoch, the script computes the training accuracy directly after completing the train loop:
`_, train_acc = validate(model, train_loader, criterion, device)`
Because the `validate` function in `src/ham10000/train.py` hardcodes `desc="validation"` in its tqdm progress bar, evaluating the training loader (7,974 images / 32 batch size = ~250 batches) produces a progress bar that says "validation: 250/250". The actual validation loader evaluates 2,041 images / 32 batch size = ~64 batches, which correctly corresponds to the true validation step. The validation set evaluation metric was handled correctly and cleanly.

## 8. Checkpoint Integrity

The checkpoint `results/baseline_resnet18/best_model.pth` exists, is loadable, corresponds to epoch 7 (which achieved the lowest validation loss of 0.5936), and maintains the standard ResNet18 architecture format.

## 9. Metric Integrity

Accuracy, Macro F1, Weighted F1, Precision, and Recall are correctly extracted via `sklearn.metrics.classification_report`. Confusion matrices are generated perfectly. 
AUC was not included in the baseline report because it was not reliably established from the completed experiment.

## 10. Grad-CAM Integrity

Grad-CAM was successfully generated from the best checkpoint using `model.layer4[-1]` as the target layer. 5 correct and 5 incorrect validation images were appropriately extracted and saved.

## 11. Problems Found

NONE

## 12. Final Recommendation

EXP-001 can be frozen as the official classical baseline. No further adjustments to this baseline are needed.
