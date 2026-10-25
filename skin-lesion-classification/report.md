# HAM10000 Skin Lesion Classification: Technical Audit & Evaluation Report

**Date:** October 13, 2026  
**Model:** ResNet-18 (Transfer Learning / Full Fine-Tuning)  
**Dataset:** HAM10000 / ISIC Dermoscopic Lesion Dataset (7 Classes)  
**Configuration File:** `configs/baseline.yaml`  
**Checkpoint:** `best_model.pth`  

---

## 1. Executive Summary

This report documents the end-to-end pipeline execution, architectural verification, and diagnostic performance analysis for the ResNet-18 baseline model trained on the HAM10000 dataset.

* **Task:** Automated 7-class dermoscopic skin lesion classification.
* **Checkpoint Status:** Genuine transfer learning fine-tuning starting from ImageNet weights, trained on 9,375 images and evaluated on 2,345 validation images.
* **Key Results:**
  * **Overall Accuracy:** 75.0%
  * **Macro Average F1-Score:** 0.67
  * **Macro Average Recall (Sensitivity):** 76.0%
  * **Melanoma (`mel`) Recall:** 78.0% (Precision: 40.0%)
  * **Basal Cell Carcinoma (`bcc`) Recall:** 81.0% (Precision: 60.0%)
* **Verdict:** The 75% accuracy is **not a malfunction or failure**. It represents a deliberate, standard machine-learning trade-off resulting from **class-weighted training** (`use_class_weights: true`), which prioritizes catching malignant lesions (high sensitivity/recall) over maximizing raw overall accuracy.

---

## 2. Pipeline Overview: What Was Done

### 2.1 Environment & Dependency Resolution
* Resolved environment conflicts in `requirements.txt` by removing an inaccessible private SSH dependency (`ham10000-classifier.git`) and switching to a local editable build (`-e .`).
* Verified installation of all core libraries: PyTorch, Torchvision, Scikit-learn, Pandas, Pillow, and Matplotlib.

### 2.2 Dataset Path Adaptation & Schema Mapping
* Adapted file paths in configuration files (`configs/baseline.yaml`, `configs/efficientnet_b0.yaml`, `configs/vit_tiny.yaml`) from legacy paths to the active repository directory:
  * `metadata_csv`: `data/ham10000_metadata_2026-10-02.csv`
  * `data_dir`: `data`
* Patched the dataset loader in `src/ham10000/data.py`:
  * Added discovery for flat directory storage under `data/ISIC-images/`.
  * Normalized metadata schemas: mapped `isic_id -> image_id` and translated ISIC textual diagnoses (`diagnosis_3`, `diagnosis_2`) to standard 7-class HAM10000 codes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

### 2.3 Data Partitioning & Leakage Prevention
* Built lesion-aware data loaders using `train_test_split(..., random_state=42)`.
* **Lesion-Level Split:** Grouping by `lesion_id` ensures that multiple dermoscopic images of the same patient lesion never cross between training and validation sets, eliminating data leakage.
* Resulting split:
  * **Training Set:** 9,375 images (augmented with RandomResizedCrop, RandomHorizontalFlip, RandomRotation, and ColorJitter).
  * **Validation Set:** 2,345 images (deterministic resize to $224 \times 224$ and ImageNet normalization).

### 2.4 Training & Evaluation
* Trained ResNet-18 with Adam optimizer ($\text{lr} = 10^{-4}$), batch size 32, and weighted Cross-Entropy Loss to counter heavy class imbalance.
* Saved the lowest validation loss weights to `best_model.pth`.
* Evaluated `best_model.pth` on the 2,345 validation images via `python -m ham10000.evaluate`.

---

## 3. Architecture Verification: Is This Truly Fine-Tuned ResNet-18?

**Yes.** The model is a fully fine-tuned ResNet-18. Specifically:

1. **Pretrained Backbone:**
   In `src/ham10000/models.py`, the network is initialized with Torchvision's official ImageNet weights:
   ```python
   model = models.resnet18(weights="IMAGENET1K_V1")
   ```
   This loads pre-learned visual filters (edges, textures, shapes) learned from 1.28 million ImageNet images.

2. **Custom Classifier Head:**
   The original 1000-class linear projection is replaced with a 7-class diagnostic head matching HAM10000:
   ```python
   num_features = model.fc.in_features  # 512
   model.fc = nn.Linear(512, 7)
   ```
   State dict inspection of `best_model.pth` confirms 122 weight tensors, with `fc.weight` shaped as `torch.Size([7, 512])` and `fc.bias` shaped as `torch.Size([7])`.

3. **Full-Network Gradient Updates:**
   In `src/ham10000/train.py`:
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
   ```
   All layers (the conv layers across all 4 residual blocks + the classification head) were updated with backpropagation on HAM10000 images, which defines **full fine-tuning** (as opposed to linear probing / feature extraction where backbones are frozen).

### What Is Missing / Differences from Advanced Fine-Tuning Setups
While the pipeline is functional and valid, some advanced training techniques were not implemented:
* **Staged Unfreezing (Warm-Up):** The entire backbone was trained from step 1 at $\text{lr}=10^{-4}$, rather than freezing the backbone for 2–3 epochs to stabilize the randomly initialized FC layer first.
* **Learning Rate Scheduler:** No decay schedule (e.g., CosineAnnealing or ReduceLROnPlateau) was applied; a static learning rate was maintained throughout.
* **Dedicated Test Set:** The current workflow utilizes an 80/20 train/val split without a third separate test set held completely untouched until final paper reporting.

---

## 4. Evaluation Analysis: Is 75% Accuracy Good, or Is There a Problem?

### 4.1 Detailed Metrics Breakdown

```
              precision    recall  f1-score   support
       akiec       0.46      0.77      0.58        62
         bcc       0.60      0.81      0.69       108
         bkl       0.59      0.63      0.61       231
          df       0.59      0.70      0.64        23
         mel       0.40      0.78      0.53       265
          nv       0.96      0.75      0.84      1616
        vasc       0.72      0.90      0.80        40

    accuracy                           0.75      2345
   macro avg       0.62      0.76      0.67      2345
weighted avg       0.83      0.75      0.77      2345
```

### 4.2 Why 75% Accuracy Is Good in This Context

#### 1. The Class Imbalance "Trap"
HAM10000 is heavily skewed toward benign melanocytic nevi (`nv`):
* `nv` accounts for **1,616 out of 2,345 validation samples (68.9%)**.
* A trivial, uninformative model that always predicts `nv` without inspecting the image would achieve **~69% accuracy**, but with **0% sensitivity on Melanoma and all other cancers**.
* Standard models trained **without class weights** easily achieve **82–84% accuracy**, but they do so by sacrificing minority classes (often dropping melanoma recall below 40–50%).

#### 2. High Clinical Sensitivity (Recall)
Because `use_class_weights: true` was enabled:
* **Melanoma (`mel`):** **78% Recall.** The model caught 207 out of 265 melanomas in validation.
* **Basal Cell Carcinoma (`bcc`):** **81% Recall.**
* **Vascular Lesions (`vasc`):** **90% Recall.**
* **Actinic Keratosis (`akiec`):** **77% Recall.**
* **Macro Recall is 76%**, proving the network generalized across rare and common classes alike.

### 4.3 Where the Real Problem / Trade-off Lies

1. **Melanoma Precision is Low (40%):**
   * While 78% of melanomas are detected, only 40% of lesions predicted as melanoma are truly malignant. 60% are false alarms (mostly benign moles `nv` or keratoses `bkl`).
   * This false positive rate is the direct penalty paid for assigning heavy inverse weights to rare classes.

2. **Resolution Downsampling ($224 \times 224$):**
   * HAM10000 images originate at $600 \times 450$ resolution. Downsampling to $224 \times 224$ removes subtle dermoscopic details (pigment network atypicality, irregular streaks, regression structures).

3. **Model Capacity:**
   * ResNet-18 has 11.2 million parameters. Modern dermatology benchmarks typically employ EfficientNet-B0/B3 or Vision Transformers (ViT) to improve separation between visually similar atypical nevi and early-stage melanomas.

---

## 5. Summary & Recommendations

| Question | Assessment |
| :--- | :--- |
| **Is ResNet-18 fine-tuned on HAM10000?** | **Yes.** Initialized with ImageNet weights, adapted with a 7-class head, and trained on the dataset. |
| **Are there missing elements?** | No critical functional bugs. Missing advanced optimizations: learning rate scheduling, staged backbone unfreezing, and a separate held-out test split. |
| **Is 75% accuracy good?** | **Yes, for a class-weighted baseline.** It prioritizes life-saving recall (78% Melanoma, 81% BCC) over superficial high accuracy. |

### Recommended Next Steps
1. **Compare Against EfficientNet-B0:** Run `configs/efficientnet_b0.yaml` to observe if mobile inverted convolutions improve the precision/recall trade-off.
2. **Probability Threshold Tuning:** Instead of raw `argmax`, tune decision thresholds per class to control false positive vs. false negative rates for melanoma.
3. **Focal Loss:** Replace simple inverse-frequency weights with Focal Loss ($\gamma = 2.0$) to prevent easy benign examples from dominating gradients while mitigating extreme false alarm rates.
