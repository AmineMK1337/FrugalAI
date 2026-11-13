# FrugalAI

Skin-lesion classification on HAM 10000 with a fine-tuned ResNet-18. This page summarizes the work so far: the data split, fine-tuning, and the evaluation of the FP32 model against its INT8 versions (weight-only quantization in [quantization/ptq/weight_only/](quantization/ptq/weight_only/), static post-training quantization in [quantization/ptq/static/](quantization/ptq/static/)). Quantization-aware training (QAT) is not done yet.

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

All models (FP32, weight-only INT8, static INT8 PTQ) are evaluated by [evaluation/evaluate.py](evaluation/evaluate.py) on the same validation images, on the same machine, on the CPU only, and in the same format, so the only difference between models is the precision (FP32 or INT8). The exact settings and file hashes of each run are saved in [evaluation/results/run_info.json](evaluation/results/run_info.json).

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

Every model runs as TorchScript with each convolution fused with its BatchNorm and ReLU. The FP32 and weight-only models are loaded into `torchvision.models.quantization.resnet18` (`quantize=False`), fused with `fuse_model()` and traced with `torch.jit.trace`; the static PTQ model is already fused TorchScript. Before anything is measured, each fused model is compared with its unfused version on the validation set: predictions are 100% identical and logits differ by at most 3e-5 (float rounding) ([fused_vs_unfused_check_validation.csv](evaluation/results/comparison/performance/fused_vs_unfused_check_validation.csv)). Model size is still the size of each model's original file.

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

Results and plots are written to [evaluation/results/](evaluation/results/), by model (`fp32`, `weight_only_per_channel`, `weight_only_per_tensor`, `static_ptq`) and metric type (`performance`, `efficiency`, `environment`), with cross-model comparisons in `comparison/`.

## Next steps

- Quantization-aware training (QAT) of the best model (strong_cosine25), then add it to the evaluation in the same way as the other models.
- Once every model is final, evaluate all of them once on the held-out test set (`python evaluation/evaluate.py --split test`).

## Reproduce

See [resnet18_finetunning/README.md](resnet18_finetunning/README.md). The model checkpoints (`.pth`) are gitignored and are not in the repository.
