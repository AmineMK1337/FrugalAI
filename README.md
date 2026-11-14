# FrugalAI

Skin-lesion classification on HAM 10000 with a fine-tuned ResNet-18. This page summarizes the work so far: the data split, fine-tuning, and the evaluation of the FP32 model against its INT8 versions (weight-only quantization in [quantization/ptq/weight_only/](quantization/ptq/weight_only/), static post-training quantization in [quantization/ptq/static/](quantization/ptq/static/), quantization-aware training in [quantization/qat/](quantization/qat/)).

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

All 7 classes appear in every partition. Each class's share of every partition is within 0.53 percentage points of its share of the full dataset. The script checks the dataset and the split before writing anything, and all checks passed ([integrity_report.txt](resnet18_finetunning/splits/integrity_report.txt)). It also records SHA-256 hashes of the split files ([split_config.json](resnet18_finetunning/splits/split_config.json)).

## 3. Fine-tuning

`finetune_resnet18.py` fine-tunes an ImageNet-pretrained ResNet-18. Both runs share these settings:

- 224×224 input, batch 32, 25 epochs, Adam with learning rate 1e-4, seed 42
- **Class-weighted cross-entropy** to counter the class imbalance
- **Strong augmentation:** rotation at any angle, horizontal and vertical flips, random crops covering 60–100% of the image, and colour jitter. Dermoscopy images have no natural "up" direction, so none of these change the diagnosis.
- The checkpoint with the lowest validation loss is kept.

The two runs differ in a single setting, the learning-rate schedule:

| Run | Config | Learning-rate schedule |
|---|---|---|
| strong_aug | [strong_aug.yaml](resnet18_finetunning/configs/strong_aug.yaml) | constant 1e-4 |
| strong_cosine25 | [strong_cosine25.yaml](resnet18_finetunning/configs/strong_cosine25.yaml) | cosine, decaying from 1e-4 to ~0 |

Both runs trained on an NVIDIA A40 in about 18 minutes each.

## 4. Results (validation set, 1,532 images)

| Run | Accuracy | Macro F1 | Weighted F1 | Melanoma recall | Best epoch | Best val loss |
|---|---|---|---|---|---|---|
| strong_aug | 79.9% | 0.678 | 0.809 | 61.3% | 19 | 0.609 |
| **strong_cosine25** | **82.2%** | **0.710** | **0.827** | **66.5%** | 19 | **0.553** |

The cosine schedule beats the constant learning rate on every overall metric, with the same augmentation and the same number of epochs. Per class, it improves F1 on six of the seven classes; the exception is vasc (0.76 against 0.78), which has only 22 validation images.

These scores are on the validation set, which was also used to pick the best epoch, so they are slightly optimistic. The test set (1,502 images) is still unused and is reserved for the final evaluation.

Each run's folder in [resnet18_finetunning/results/](resnet18_finetunning/results/) contains:
- per-class classification reports
- confusion matrices, as counts and normalized
- the training history for every epoch
- Grad-CAM maps for 5 correct and 5 incorrect predictions
- the exact config and software environment

## 5. Evaluation machine and thread protocol

All models (FP32, weight-only INT8, static INT8 PTQ, INT8 QAT) are evaluated by [evaluation/evaluate.py](evaluation/evaluate.py) on the same validation images, on the same machine, on the CPU only, and in the same format, so the only difference between models is the precision (FP32 or INT8). The exact settings and file hashes of each run are saved in [evaluation/results/run_info.json](evaluation/results/run_info.json).

### The virtual machine

| Component | Value |
|---|---|
| Type | KVM virtual machine |
| CPU | Intel Xeon Gold 6348 @ 2.60 GHz, 32 vCPUs (1 thread per vCPU), AVX-512 with VNNI (INT8 instructions) |
| Memory | 62 GB |
| GPU | NVIDIA A40-24Q (virtual GPU): used for fine-tuning only, not for evaluation |
| OS | Ubuntu 23.10, Linux kernel 6.5 |
| Software | Python 3.11.6, PyTorch 2.12.0, quantized backend `x86` |

The VM exposes no power counter: there is no RAPL for the CPU, and the virtual GPU does not report power. Energy and CO₂e are therefore estimated by CodeCarbon, not measured (see below).

### Same format for every model: fused + TorchScript

Every model runs as TorchScript with each convolution fused with its BatchNorm and ReLU. The FP32 and weight-only models are loaded into `torchvision.models.quantization.resnet18` (`quantize=False`), fused with `fuse_model()` and traced with `torch.jit.trace`; the static PTQ and QAT models are already fused INT8 TorchScript. Before anything is measured, each fused model is compared with its unfused version on the validation set: predictions are 100% identical and logits differ by at most 3e-5 (float rounding) ([fused_vs_unfused_check_validation.csv](evaluation/results/comparison/performance/fused_vs_unfused_check_validation.csv)). Model size is still the size of each model's original file.

### Measuring at 1 and 4 threads

The thread count is the number of CPU cores PyTorch uses for one inference (`torch.set_num_threads`). Latency, throughput and peak memory depend on it, so each is measured at 1 and 4 threads. Energy and CO₂e are measured at 1 thread. Accuracy and the other performance metrics, file size and FLOPs do not depend on it, so they are computed once.

- **Main protocol: 1 thread.** It is the most stable and reproducible setting, close to a small device. The 4-thread results show how each model scales on a multi-core CPU.
- **One shared function** measures every model, with the same steps and settings.
- **One fresh process per measurement.** Each combination of model and thread count runs in a new Python process, so no cache or memory is carried over between runs. Inter-op parallelism is fixed at 1 thread.
- **3 repeats, interleaved.** The runs go repeat → thread count → model, all back to back in one session. A slowdown of the machine at one moment therefore affects every model, not just one. The reported value is the median of the 3 repeats.
- **Same inputs.** The validation images are decoded once and saved, and every run reads the same tensors.

In each run:

| Metric | How it is measured |
|---|---|
| Latency | One image (batch 1): 30 warm-up runs, then 200 timed runs. Median and 95th percentile, in ms. |
| Peak RAM | Increase in the process's resident memory (psutil, sampled every millisecond) while loading the model and running the batch-1 warm-up, in MB. |
| Throughput | All 1,532 validation images in batches of 32, in images per second. |
| Energy and CO₂e | At 1 thread: CodeCarbon (offline, process mode) over whole passes of the validation set (batch 32) lasting at least 60 s, scaled to 1,000 images. CO₂e uses Tunisia's grid intensity, 564 gCO₂e/kWh. |

**Limit of the energy estimate:** without a power counter, CodeCarbon assumes about 250 W per thread in use. Energy is therefore roughly that power × run time: it is an estimate, useful to compare models, not a measured value.

To rerun the evaluation (validation split):

```
python evaluation/evaluate.py
```

Results and plots are written to [evaluation/results/](evaluation/results/), by model (`fp32`, `weight_only_per_channel`, `weight_only_per_tensor`, `static_ptq`, `qat`) and metric type (`performance`, `efficiency`, `environment`), with cross-model comparisons in `comparison/`. Every quantized model is compared with FP32: its plots show FP32 next to it, `performance/change_vs_fp32` shows each metric's change with a paired-bootstrap CI and p-value, and each metric folder has a `vs_fp32_<split>.csv` table. Performance metrics are accuracy, balanced accuracy (= macro recall), precision, specificity, F1, AUC (macro) and melanoma recall, each with a 95% bootstrap CI.

## 6. Quantization-aware training (QAT)

[quantization/qat/qat.py](quantization/qat/qat.py) fine-tunes the FP32 model (strong_cosine25) with fake quantization, using the same INT8 scheme as static PTQ: x86 backend, per-channel symmetric INT8 weights, per-tensor INT8 activations with `reduce_range=True`. Train split: QAT training and observer initialization. Validation split: choosing the epoch. The test set is not used.

**Starting point = static PTQ.** The observers are initialized on the same 1,000 training images as static PTQ, and PTQ's activation and weight ranges are copied into them (the default QAT observers use a moving-average min/max, while PTQ used a histogram observer). Before training, the INT8 model gives exactly the static PTQ predictions for every seed.

**3-phase schedule, 10 epochs.** Weights train in every phase, with class-weighted cross-entropy, strong augmentation, batch 32 and Adam.

| Epochs | Phase | Learning rate | BatchNorm statistics | Observers (scales, zero points) |
|---|---|---|---|---|
| 1–2 | Adaptation | 1e-5, constant | updating | updating |
| 3–7 | Fine-tuning | cosine decay from 1e-5 | updating | updating |
| 8–10 | Stabilization | cosine continues toward 0 (3.8e-7 at epoch 10) | frozen | disabled |

**Selection.** The saved model ([resnet18_qat_int8.pt](quantization/qat/)) is seed 42 at its best INT8 validation macro F1 among epochs 8–10, the only epochs whose quantization parameters are final. Epoch 10 is also reported without selection. Seeds 42, 1 and 2 are run; seed 42 is the saved model.

**FP32 control.** The same 10 epochs, schedule and selection rule from the FP32 model, without fake quantization. QAT starts from a model that was already trained to convergence and trains it further, so comparing QAT with the original FP32 model mixes two effects: extra fine-tuning and quantization. Only QAT vs control isolates quantization.

**Validation results, mean over 3 seeds (min–max):**

| | Accuracy | Macro F1 | Melanoma recall |
|---|---|---|---|
| FP32 | 82.18 | 71.04 | 66.47 |
| Static PTQ | 82.11 (81.85–82.44) | 70.68 (70.32–71.00) | 66.28 (65.32–67.63) |
| FP32 control, best of epochs 8–10 | 82.59 (81.98–83.03) | 71.60 (70.71–72.26) | 71.10 (69.94–72.25) |
| QAT, best of epochs 8–10 | 82.94 (82.57–83.29) | 72.12 (70.95–73.36) | 73.22 (69.94–75.14) |
| QAT − control, best of epochs 8–10 | +0.35 (+0.20 to +0.59) | +0.53 (+0.24 to +1.11) | +2.12 (−2.31 to +4.62) |
| QAT − control, epoch 10 | −0.17 (−0.52 to +0.20) | +0.11 (−0.48 to +0.55) | +0.96 (−2.31 to +4.05) |

**Conclusion (QAT vs control).** The INT8 QAT model matches FP32 trained with the same schedule. With selection, the QAT − control difference is small and its sign varies from seed to seed for melanoma recall. At epoch 10, without selection, it is within ±0.6 points for accuracy and macro F1 on every seed. QAT beats the original FP32 model by about +1.1 points of macro F1 and +6.8 points of melanoma recall on average (best of epochs 8–10), but the FP32 control gains +0.6 and +4.6 from the same extra fine-tuning. At epoch 10, the control gets +0.55 of QAT's +0.65 macro-F1 gain. So most of the gain comes from training longer, not from quantization-aware training. The rest is smaller than the seed-to-seed spread. QAT is still a way to get that extra fine-tuning into an INT8 model with the speed of static PTQ, which is 7.4× faster than FP32 at 1 thread, 4× smaller, with an estimated 87% less energy per image. For this model, static PTQ already loses only about 0.3 points of macro F1, so QAT is not needed to recover from INT8 quantization. All of these are validation numbers, and validation also chose the epochs. The seed-to-seed spread (up to 2.4 points of macro F1) is as large as most differences here, so the test set is needed for the final comparison.

The previous QAT run (v1: one cosine schedule, best epoch chosen among 0–10) is kept in [quantization/qat/_v1_previous/](quantization/qat/_v1_previous/). Per-epoch logs, the training curves (QAT vs control, per seed and mean), the settings and the summary tables are in [quantization/qat/results/](quantization/qat/results/).

## Next steps

- Once every model is final, evaluate all of them once on the held-out test set (`python evaluation/evaluate.py --split test`).

## Reproduce

See [resnet18_finetunning/README.md](resnet18_finetunning/README.md). The model checkpoints (`.pth`) are gitignored and are not in the repository.
