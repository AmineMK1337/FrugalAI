# FrugalAI

Skin-lesion classification on HAM 10000 with a fine-tuned ResNet-18. This page summarizes the work so far: the data split, fine-tuning, and the evaluation of the FP32 baseline against its INT8 versions: weight-only quantization, per-channel and per-tensor ([quantization/ptq/weight_only/](quantization/ptq/weight_only/)), and quantization-aware training ([quantization/qat/](quantization/qat/)).

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

All four models (FP32 baseline, weight-only INT8 per-channel, weight-only INT8 per-tensor, INT8 QAT) are evaluated by [evaluation/evaluate.py](evaluation/evaluate.py) on the same validation images, on the same machine, on the CPU only, and in the same format, so the only difference between models is the precision (FP32 or INT8). The exact settings and file hashes of each run are saved in [evaluation/results/run_info.json](evaluation/results/run_info.json).

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

Every model runs as TorchScript with each convolution fused with its BatchNorm and ReLU. The FP32 and weight-only models are loaded into `torchvision.models.quantization.resnet18` (`quantize=False`), fused with `fuse_model()` and traced with `torch.jit.trace`; the QAT model is already fused INT8 TorchScript. Before anything is measured, each fused model is compared with its unfused version on the validation set: predictions are 100% identical and logits differ by at most 3e-5 (float rounding) ([fused_vs_unfused_check_validation.csv](evaluation/results/comparison/performance/fused_vs_unfused_check_validation.csv)). Model size is still the size of each model's original file.

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

Results and plots are written to [evaluation/results/](evaluation/results/), by model (`fp32`, `weight_only_per_channel`, `weight_only_per_tensor`, `qat`) and metric type (`performance`, `efficiency`, `environment`), with cross-model comparisons in `comparison/`. Every quantized model is compared with FP32: its plots show FP32 next to it, `performance/change_vs_fp32` shows each metric's change with a paired-bootstrap CI and p-value, and each metric folder has a `vs_fp32_<split>.csv` table. Performance metrics are accuracy, balanced accuracy (= macro recall), precision, specificity, F1, AUC (macro) and melanoma recall, each with a 95% bootstrap CI.

## 6. Quantization-aware training (QAT)

[quantization/qat/qat.py](quantization/qat/qat.py) fine-tunes the FP32 baseline (strong_cosine25) with fake quantization, so the model learns to work with INT8 weights and activations: x86 backend, per-channel symmetric INT8 weights, per-tensor INT8 activations with `reduce_range=True`. Train split: QAT training and observer initialization. Validation split: monitoring each epoch only. The test set is not used.

**Starting point.** The observers (which set the INT8 scales) are initialized on 1,000 training images, so training starts from a working INT8 model.

**3-phase schedule, 10 epochs, seed 42.** Weights train in every phase, with class-weighted cross-entropy, strong augmentation, batch 32 and Adam.

| Epochs | Phase | Learning rate | BatchNorm statistics | Observers (scales, zero points) |
|---|---|---|---|---|
| 1–2 | Adaptation | 1e-5, constant | updating | updating |
| 3–7 | Fine-tuning | cosine decay from 1e-5 | updating | updating |
| 8–10 | Stabilization | cosine continues toward 0 | frozen | disabled |

**Saved model.** [resnet18_qat_int8.pt](quantization/qat/) is the final model (epoch 10); no epoch is picked on validation. Per-epoch logs, the training curves and the settings are in [quantization/qat/results/](quantization/qat/results/).

## 7. Results: FP32 baseline vs quantized models (validation set, 1,532 images)

From [evaluation/results/comparison/](evaluation/results/comparison/), CPU, fused + TorchScript, latency at 1 thread (main protocol).

| Model | Accuracy | Macro F1 | Melanoma recall | Size (MB) | Latency (ms) | Speed-up | Peak RAM vs FP32 | Energy / 1,000 images (Wh, estimate) |
|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 82.18 | 71.04 | 66.47 | 44.80 | 50.74 | 1.00× | – | 3.31 |
| Weight-only INT8, per-channel | 82.70 | 71.03 | 64.74 | 11.31 | 50.89 | 1.00× | +2.6% | 3.38 |
| Weight-only INT8, per-tensor | 81.27 | 70.22 | 69.36 | 11.29 | 50.87 | 1.00× | +1.6% | 3.32 |
| INT8 QAT | 82.70 | 71.18 | 73.41 | 11.32 | 6.78 | 7.49× | −31.2% | 0.46 (−86%) |

- **Weight-only INT8** makes the file 4× smaller with no significant change in accuracy or macro F1 (paired bootstrap, p ≥ 0.05). It gives no speed or energy gain, because the weights are dequantized back to FP32 when the model loads and the computation stays in FP32.
- **INT8 QAT** is also 4× smaller and computes in INT8: 7.5× faster at 1 thread (6.9× at 4 threads), −31% peak RAM and an estimated −86% energy. Accuracy and macro F1 equal FP32 within noise (+0.52 and +0.14 points, p = 0.40 and 0.90). Melanoma recall is +6.94 points (95% paired-bootstrap CI +2.4 to +11.3). That gain comes from one training run (seed 42) and moves a lot from epoch to epoch (64–74% across epochs 1–10), so it needs to be confirmed on the test set.

### Why 8-bit weights lose almost nothing: bit-width sweep

[bitwidth_sweep.py](quantization/ptq/weight_only/bitwidth_sweep.py) applies the same weight-only integer scheme with 32, 24, 20 and 16 bits, then every width from 15 down to 2 (validation only; 8 bits reproduces the saved INT8 results exactly). The 16-bit float formats FP16 and BF16 are added for reference. Plot and full table (every bit width, with 95% bootstrap CIs): [results/bitwidth_sweep/](quantization/ptq/weight_only/results/bitwidth_sweep/).

| Bits | Weight error, per-channel / per-tensor | Macro F1, per-channel | Macro F1, per-tensor | Identical to FP32, per-channel / per-tensor |
|---|---|---|---|---|
| FP32 (32-bit float) | 0 | 71.04 | 71.04 | 100 / 100 |
| INT 32, 24, 20, 16 | ≤ 0.005% / ≤ 0.014% | 71.04 | 71.04 | 100 / 100 |
| FP16 / BF16 (float, for reference) | 0.02% / 0.17% | 71.04 / 70.94 | – | 100 / 99.74 |
| 12 | 0.08% / 0.22% | 70.98 | 71.37 | 99.93 / 99.87 |
| 10 | 0.33% / 0.89% | 71.49 | 71.20 | 99.67 / 98.89 |
| 9 | 0.66% / 1.79% | 71.60 | 71.38 | 98.69 / 98.50 |
| 8 | 1.3% / 3.6% | 71.03 | 70.22 | 97.72 / 96.54 |
| 7 | 2.7% / 7.2% | 70.64 | 69.66 |
| 6 | 5.4% / 14.7% | 65.67 | 66.15 |
| 5 | 11.2% / 30.3% | 57.45 | 48.39 |
| 4 | 23.8% / 60.7% | 36.69 | 11.24 |
| 3 | 51.3% / 89.3% | 14.42 | 11.41 |
| 2 | 91.6% / 99.1% | 11.41 | 11.41 |

From 32 down to 16 bits nothing changes: the predictions are 100% identical to FP32, because an FP32 weight carries only 24 bits of precision, and at 16 bits the rounding error is at most 0.014%. Between 15 and 9 bits a few images (at most 1.5%) change class, in both directions, and every metric stays at the FP32 level. Each bit removed doubles the weight error. At 8 and 7 bits the error (1–7%) stays within what the network tolerates, and macro F1 stays within the FP32 confidence interval. From 6 bits down, performance degrades, and it collapses at 4 bits: per-channel macro F1 falls to 37, and per-tensor ends up predicting "nv" for every image. That gives 66.45% accuracy (the share of nv in the validation set) but macro F1 11.4 and melanoma recall 0. Accuracy alone hides this collapse, which is why macro F1 and melanoma recall are the main metrics. Per-channel scales hold up about one bit longer than per-tensor scales.

## Next steps

- Once every model is final, evaluate all of them once on the held-out test set (`python evaluation/evaluate.py --split test`).

## Reproduce

See [resnet18_finetunning/README.md](resnet18_finetunning/README.md). The model checkpoints (`.pth`) are gitignored and are not in the repository.
