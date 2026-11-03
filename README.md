# FrugalAI

Skin-lesion classification on HAM10000 with a fine-tuned ResNet-18. This page summarizes the work so far. Quantization and efficiency benchmarking will be redone later.

## 1. Dataset

[HAM10000](https://doi.org/10.7910/DVN/DBW86T): 10,015 dermoscopy images of 7,470 lesions in 7 classes. The classes are heavily imbalanced:

| nv | mel | bkl | bcc | akiec | vasc | df |
|---|---|---|---|---|---|---|
| 6,705 | 1,113 | 1,099 | 514 | 327 | 142 | 115 |

## 2. Train / validation / test split

`make_splits.py` builds one fixed split that every run uses:

- **Split by lesion, not by image.** A lesion often has several images, and all of them land in the same partition. This stops near-duplicate images from leaking from training into evaluation. HAM10000 has no patient ID, so splitting by patient is not possible.
- **Stratified by diagnosis**, seed 42. Lesions are split 70% train, then the remaining 30% is halved into validation and test.

| Partition | Images | Lesions |
|---|---|---|
| Train | 6,981 | 5,229 |
| Validation | 1,532 | 1,120 |
| Test | 1,502 | 1,121 |

All 7 classes appear in every partition. Each class's share of every partition is within 0.53 percentage points of its share of the full dataset. The script checks the dataset and the split before writing anything, and all checks passed ([integrity_report.txt](resnet18_finetune/splits/integrity_report.txt)). It also records SHA-256 hashes of the split files ([split_config.json](resnet18_finetune/splits/split_config.json)).

## 3. Fine-tuning

`finetune_resnet18.py` fine-tunes an ImageNet-pretrained ResNet-18. Both runs share these settings:

- 224×224 input, batch 32, 25 epochs, Adam with learning rate 1e-4, seed 42
- **Class-weighted cross-entropy** to counter the class imbalance
- **Strong augmentation:** rotation at any angle, horizontal and vertical flips, random crops covering 60–100% of the image, and colour jitter. Dermoscopy images have no natural "up" direction, so none of these change the diagnosis.
- The checkpoint with the lowest validation loss is kept.

The two runs differ in a single setting, the learning-rate schedule:

| Run | Config | Learning-rate schedule |
|---|---|---|
| strong_aug | [strong_aug.yaml](resnet18_finetune/configs/strong_aug.yaml) | constant 1e-4 |
| strong_cosine25 | [strong_cosine25.yaml](resnet18_finetune/configs/strong_cosine25.yaml) | cosine, decaying from 1e-4 to ~0 |

Both runs trained on an NVIDIA A40 in about 18 minutes each.

## 4. Results (validation set, 1,532 images)

| Run | Accuracy | Macro F1 | Weighted F1 | Melanoma recall | Best epoch | Best val loss |
|---|---|---|---|---|---|---|
| strong_aug | 79.9% | 0.678 | 0.809 | 61.3% | 19 | 0.609 |
| **strong_cosine25** | **82.2%** | **0.710** | **0.827** | **66.5%** | 19 | **0.553** |

The cosine schedule beats the constant learning rate on every overall metric, with the same augmentation and the same number of epochs. Per class, it improves F1 on six of the seven classes; the exception is vasc (0.76 against 0.78), which has only 22 validation images.

These scores are on the validation set, which was also used to pick the best epoch, so they are slightly optimistic. The test set (1,502 images) is still unused and is reserved for the final evaluation.

Each run's folder in [resnet18_finetune/results/](resnet18_finetune/results/) contains:
- per-class classification reports
- confusion matrices, as counts and normalized
- the training history for every epoch
- Grad-CAM maps for 5 correct and 5 incorrect predictions
- the exact config and software environment

## Next steps

- Evaluate both models on the held-out test set.
- Redo INT8 quantization (post-training and quantization-aware) on top of the best model.
- Benchmark size, latency, memory and energy.

## Reproduce

See [resnet18_finetune/README.md](resnet18_finetune/README.md). The model checkpoints (`.pth`) are gitignored and are not in the repository.
