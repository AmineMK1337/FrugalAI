# FrugalAI

Skin-lesion classification on HAM 10000 with a fine-tuned ResNet-18. This page summarizes the work so far: the data split, fine-tuning, and the evaluation of the FP32 baseline against its INT8 versions: weight-only quantization, per-channel and per-tensor ([quantization/ptq/weight_only/](quantization/ptq/weight_only/)), and quantization-aware training ([quantization/qat/](quantization/qat/)), plus structured (channel) pruning ([pruning/structured/](pruning/structured/)) and unstructured (weight) pruning ([pruning/unstructured/](pruning/unstructured/)), both in FP32.

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

All four models (FP32 baseline, weight-only INT8 per-channel, weight-only INT8 per-tensor, INT8 QAT) are evaluated by [evaluation/evaluate.py](evaluation/evaluate.py) on the same validation images, on the same machine, on the CPU only, and in the same format, so the only difference between models is the precision (FP32 or INT8). The exact settings and file hashes of each run are saved in [evaluation/results_quantization/run_info.json](evaluation/results_quantization/run_info.json).

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

Every model runs as TorchScript with each convolution fused with its BatchNorm and ReLU. The FP32 and weight-only models are loaded into `torchvision.models.quantization.resnet18` (`quantize=False`), fused with `fuse_model()` and traced with `torch.jit.trace`; the QAT model is already fused INT8 TorchScript. Before anything is measured, each fused model is compared with its unfused version on the validation set: predictions are 100% identical and logits differ by at most 3e-5 (float rounding) ([fused_vs_unfused_check_validation.csv](evaluation/results_quantization/comparison/performance/fused_vs_unfused_check_validation.csv)). Model size is still the size of each model's original file.

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

Results and plots are written to `evaluation/results/` by default (the quantization results are kept in [evaluation/results_quantization/](evaluation/results_quantization/)), by model (`fp32`, `weight_only_per_channel`, `weight_only_per_tensor`, `qat`) and metric type (`performance`, `efficiency`, `environment`), with cross-model comparisons in `comparison/`. Every quantized model is compared with FP32: its plots show FP32 next to it, `performance/change_vs_fp32` shows each metric's change with a paired-bootstrap CI and p-value, and each metric folder has a `vs_fp32_<split>.csv` table. Performance metrics are accuracy, balanced accuracy (= macro recall), precision, specificity, F1, AUC (macro) and melanoma recall, each with a 95% bootstrap CI.

`--models` picks which models to evaluate and `--results-dir` where to write them. The pruned models are evaluated this way in their own session, with FP32 re-measured there as the timing reference (section 8). FLOPs and parameters are counted on each model's own architecture.

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

From [evaluation/results_quantization/comparison/](evaluation/results_quantization/comparison/), CPU, fused + TorchScript, latency at 1 thread (main protocol).

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
| 7 | 2.7% / 7.2% | 70.64 | 69.66 | 95.30 / 91.91 |
| 6 | 5.4% / 14.7% | 65.67 | 66.15 | 89.10 / 89.36 |
| 5 | 11.2% / 30.3% | 57.45 | 48.39 | 84.60 / 73.17 |
| 4 | 23.8% / 60.7% | 36.69 | 11.24 | 63.84 / 51.11 |
| 3 | 51.3% / 89.3% | 14.42 | 11.41 | 45.23 / 62.34 |
| 2 | 91.6% / 99.1% | 11.41 | 11.41 | 62.34 / 62.34 |

From 32 down to 16 bits nothing changes: the predictions are 100% identical to FP32, because an FP32 weight carries only 24 bits of precision, and at 16 bits the rounding error is at most 0.014%. Between 15 and 9 bits a few images (at most 1.5%) change class, in both directions, and every metric stays at the FP32 level. Each bit removed doubles the weight error. At 8 and 7 bits the error (1–7%) stays within what the network tolerates, and macro F1 stays within the FP32 confidence interval. From 6 bits down, performance degrades, and it collapses at 4 bits: per-channel macro F1 falls to 37, and per-tensor ends up predicting "nv" for every image. That gives 66.45% accuracy (the share of nv in the validation set) but macro F1 11.4 and melanoma recall 0. Accuracy alone hides this collapse, which is why macro F1 and melanoma recall are the main metrics. Per-channel scales hold up about one bit longer than per-tensor scales.

## 8. Structured pruning (FP32)

[pruning/structured/prune.py](pruning/structured/prune.py) removes whole filters (output channels) from the convolutions, so the layers become physically smaller. Weights stay FP32; there is no quantization and no unstructured pruning. Data roles: train split = fine-tuning; validation split = choosing ratios and epochs; the test set is not used.

**How a pruned model is made** (diagram and measured macro F1 at each stage: [pruning_process.png](pruning/structured/results/report/pruning_process.png)):

1. **Start from the FP32 baseline**, loaded (`strict=True`) into `torchvision.models.quantization.resnet18(quantize=False)`, so the pruned models can be fused now and quantized later.
2. **Dependency groups.** [Torch-Pruning](https://github.com/VainF/Torch-Pruning) 1.6.1 builds the dependency graph. Convolutions whose outputs are added by a residual connection (the second conv of each block and the downsample conv) share their channel indices and are pruned together, so the additions stay consistent.
3. **Rank the filters: L1 filter norm.** Each output filter's importance is the L1 norm of its weights, the criterion of classic L1 filter pruning (Li et al., 2017, *Pruning Filters for Efficient ConvNets*). When several convolutions share a channel group (residual stream), the filter norms of those convolutions are averaged per channel. BatchNorm weights and the input weights of the next layers are not used.
4. **Remove the lowest-ranked filters.** Local pruning: the same target ratio (10, 20, 30 or 40%) in every prunable group. Two layers are excluded:
   - the stem `conv1`, which protects its whole channel group: the layer1 residual stream (outputs of `layer1.*.conv2`). In layer1, only the inner `conv1` of each block is pruned;
   - the classifier `fc`, which keeps its 7 outputs (its inputs follow layer4).

   The achieved ratio is recorded per layer (rounding): 10.2–10.9%, 20.1–20.3%, 30.1–31.2% and 40.0–40.6% in the pruned groups.
5. **Fine-tune**, with the same setup as the FP32 training: class-weighted cross-entropy (weights from the train split), strong augmentation, Adam, batch 32, lr 1e-4 with cosine decay over 10 epochs, seed 42 (the best ratio, 10%, and the control are repeated with seeds 1 and 2). The model is evaluated on validation right after pruning and after every epoch.
6. **Select and save** the epoch with the best validation macro F1 (epoch 10 is also reported), as the whole model (`torch.save`) and as TorchScript.
7. **Evaluate** with the shared protocol (section 5), in [evaluation/results_pruning/](evaluation/results_pruning/).

**FP32 control.** The unpruned model fine-tuned with exactly the same settings. Comparing pruned models with it separates the effect of pruning from the effect of the extra fine-tuning.

Settings: [settings.json](pruning/structured/results/settings.json), with a copy in each run's `summary.json`. Per-layer channels before/after: `pruning/structured/results/<run>/layers.csv`.

### Results (validation set)

All models were evaluated in one session by evaluate.py (CPU, fused + TorchScript; latency at 1 thread), FP32 included as the timing reference: [evaluation/results_pruning/](evaluation/results_pruning/). Change vs FP32: paired bootstrap, 95% CI. Energy is an estimate, about run time × an assumed power. Between models with the same architecture (FP32 and the three controls) it varies by up to ±5%, so only larger differences are meaningful.

| Model | Params | GMACs | Macro F1 right after pruning | Macro F1 after fine-tuning | Change vs FP32 (95% CI) | Melanoma recall | Latency (ms), 1 / 4 threads | Size (MB) | Peak RAM vs FP32 | Energy vs FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 11.18 M | 1.82 | – | 71.04 | – | 66.47 | 50.94 / 14.55 | 44.8 | – | – |
| FP32 control, seed 42 | 11.18 M | 1.82 | – | 72.65 | +1.60 (−1.62 to +4.86) | 71.10 | 50.75 / 14.67 | 44.8 | −1% | −2% |
| FP32 control, seed 1 | 11.18 M | 1.82 | – | 73.19 | +2.15 (−1.39 to +5.64) | 69.94 | 50.82 / 14.66 | 44.8 | +1% | −5% |
| FP32 control, seed 2 | 11.18 M | 1.82 | – | 70.98 | −0.07 (−3.33 to +3.43) | 70.52 | 50.86 / 14.69 | 44.8 | −1% | 0% |
| Pruned 10%, seed 42 | 9.05 M (−19%) | 1.54 (−16%) | 37.96 | 70.34 | −0.70 (−4.40 to +2.88) | 75.72 | 47.82 / 14.04 (1.07×) | 36.3 | −10% | −5% |
| Pruned 10%, seed 1 | 9.05 M | 1.54 | 37.96 | 69.42 | −1.63 (−6.03 to +2.47) | 71.10 | 47.20 / 14.06 (1.08×) | 36.3 | −12% | −7% |
| Pruned 10%, seed 2 | 9.05 M | 1.54 | 37.96 | 71.41 | +0.36 (−3.32 to +3.86) | 69.36 | 47.64 / 14.01 (1.07×) | 36.3 | −12% | −6% |
| Pruned 20%, seed 42 | 7.17 M (−36%) | 1.29 (−29%) | 16.74 | 67.25 | −3.79 (−7.64 to +0.23) | 72.25 | 43.17 / 13.26 (1.18×) | 28.8 | −20% | −15% |
| Pruned 30%, seed 42 | 5.52 M (−51%) | 1.06 (−42%) | 11.44 | 63.25 | −7.80 (−12.00 to −3.58) | 73.41 | 36.80 / 11.66 (1.38×) | 22.2 | −28% | −31% |
| Pruned 40%, seed 42 | 4.07 M (−64%) | 0.85 (−53%) | 11.41 | 60.66 | −10.39 (−14.25 to −6.64) | 61.85 | 31.92 / 10.43 (1.60×) | 16.4 | −35% | −37% |

"Right after pruning" comes from prune.py: pruning is deterministic, so it is the same for every seed. The CPU evaluation and prune.py's GPU validation differ by at most one prediction per model. Consolidated table, pruned − control by seed and plots: [pruning/structured/results/report/](pruning/structured/results/report/).

**Best ratio on validation (10%), 3 seeds** (seeds only change the fine-tuning: image order and augmentation):

| | FP32 | FP32 control, mean (min–max) | Pruned 10%, mean (min–max) | Pruned − control, by seed 42 / 1 / 2 (mean) |
|---|---|---|---|---|
| Macro F1 | 71.04 | 72.27 (70.98–73.19) | 70.39 (69.42–71.41) | −2.31 / −3.78 / +0.43 (−1.89) |
| Balanced accuracy | 71.50 | 73.03 (71.85–74.26) | 70.94 (69.08–72.56) | −1.80 / −5.18 / +0.71 (−2.09) |
| Melanoma recall | 66.47 | 70.52 (69.94–71.10) | 72.06 (69.36–75.72) | +4.62 / +1.16 / −1.16 (+1.54) |
| Accuracy | 82.18 | 82.35 (81.79–82.70) | 82.11 (81.53–82.90) | −0.78 / +0.33 / −0.26 (−0.24) |

Pruned − control above uses each model's selected epoch (CPU evaluation). At epoch 10 for both models, macro F1 pruned − control is −1.79 / −4.91 / +0.94 (mean −1.92; prune.py validation, [pruned_minus_control.csv](pruning/structured/results/report/pruned_minus_control.csv)).

- **Fine-tuning is essential.** Right after pruning, every model is broken: macro F1 11–38, and at 30–40% the model predicts "nv" for almost every image. Ten epochs recover most of the loss.
- **10% is nearly free compared with the original model.** On 3 seeds, macro F1 is 70.39 on average vs 71.04 for FP32 (−0.65), and no seed's difference is significant. The seed-to-seed spread is about 2 points, both for the pruned model and for the control, so one training run cannot resolve differences of this size.
- **Compared with the control, 10% probably costs about 2 points, but this is not established.** Pruned − control is −1.9 macro F1 on average, but the sign changes with the seed (−2.3, −3.8, +0.4).
- **From 20% the loss grows clearly with the ratio** (single seed, but much larger than the seed spread): −3.8, −7.8 and −10.4 points vs FP32 at 20, 30 and 40%, significant from 30%.
- **Speed-up is smaller than the compute cut.** GMACs fall by 16–53%, but latency only by 6–37% at 1 thread (1.07–1.60×) and 3–28% at 4 threads (1.04–1.40×). layer1, which runs at the highest resolution, is only half prunable (its residual stream is protected), and a ResNet-18 on this CPU is not limited by arithmetic alone. Peak RAM falls by 10–35%, and estimated energy by 5–37%.
- **Most pruned models were still improving at their last epochs** (best epoch 9 or 10 for 10–30%), so a longer fine-tuning could narrow the gap.
- **Melanoma recall** rises at 10% (+5.6 points vs FP32 on average), but the control rises too (+4.1), so most of it comes from the extra fine-tuning, not from pruning (pruned − control +1.5, with the sign changing by seed). Macro F1 falls at the same time, and the per-seed values vary by 6 points.
- **Compared with INT8 QAT** (section 7, a different evaluation session): QAT is 7.5× faster and 4× smaller with no loss of macro F1. Pruning gives 1.07× at 10% and 1.6× (2.7× smaller) at 40% with −10 points. For this model on this CPU, quantization is the far better trade-off. The two can also be combined: the pruned models keep the quantizable architecture.

## 9. Unstructured pruning (FP32)

[pruning/unstructured/prune_unstructured.py](pruning/unstructured/prune_unstructured.py) sets individual weights to zero instead of removing whole filters. The layers keep their shape: only the number of non-zero weights changes. Weights stay FP32, with no quantization and no structured pruning. Data roles are the same as in section 8: the train split is used for fine-tuning, the validation split for every choice, and the test set is never loaded.

**How a sparse model is made:**

1. **Start from the FP32 baseline**, loaded (`strict=True`) into `torchvision.models.quantization.resnet18(quantize=False)`, as for structured pruning.
2. **Rank all weights together: global L1 magnitude.** `torch.nn.utils.prune.global_unstructured` with `L1Unstructured` covers the weights of every convolution and of `fc`, the stem `conv1` included. The given fraction of weights with the smallest |w| is set to zero. The ranking is global across layers, so each layer ends up with a different sparsity: the large layers are pruned most. At 70% sparsity, per-layer sparsity runs from 33.8% (`conv1`) to 84.1% (`layer4.1.conv2`). Biases and BatchNorm are not touched.
3. **Fine-tune** with exactly the same settings as the structured runs and the FP32 control: class-weighted cross-entropy, strong augmentation, Adam, batch 32, lr 1e-4 with cosine decay over 10 epochs. The pruning masks stay active during fine-tuning, so pruned weights stay exactly zero; `prune.remove` then makes the zeros permanent.
4. **Select** the epoch with the best validation macro F1; epoch 10 is also reported. Ratios: 30, 50, 70 and 90% (seed 42). The best ratio, 70%, is repeated with seeds 1 and 2.
5. **Checks.** The run stops if a check fails. The achieved sparsity must equal the target. The zero count must be unchanged after fine-tuning, after `fuse_model()` and in the TorchScript export. The FP32 checkpoint, the split and the config are hashed before and after the run. Every check passed.
6. **Evaluate** with the shared protocol (section 5), all sparse models and FP32 in one session: [pruning/unstructured/results/evaluation/](pruning/unstructured/results/evaluation/).

**FP32 control.** The controls from section 8 are reused (same fine-tuning, unpruned, seeds 42, 1 and 2), not retrained.

Settings: [settings.json](pruning/unstructured/results/settings.json). Per-layer sparsity: `pruning/unstructured/results/<run>/layers.csv`. Consolidated table and plots: [pruning/unstructured/results/report/](pruning/unstructured/results/report/).

### Results (validation set)

CPU, fused + TorchScript, latency at 1 thread. Change vs FP32: paired bootstrap, 95% CI. Pruned − control comes from the validation pass of the training script (GPU), as in section 8. "Before fine-tuning" is the pruned model evaluated on the CPU right after pruning.

| Model | Non-zero params | Macro F1 before fine-tuning | Macro F1 after fine-tuning (epoch) | Change vs FP32 (95% CI) | Pruned − control | Melanoma recall | Compressed size, gzip (MB) | Latency (ms), 1 / 4 threads |
|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 11.18 M | – | 71.04 | – | – | 66.47 | 41.6 | 50.72 / 14.64 |
| 30% sparse, seed 42 | 7.83 M | 72.25 | 72.37 (7) | +1.33 (−2.36 to +5.26) | −0.11 | 69.94 | 32.0 | 50.87 / 14.68 |
| 50% sparse, seed 42 | 5.59 M | 71.16 | 72.76 (5) | +1.71 (−1.54 to +5.14) | +0.28 | 63.58 | 24.3 | 50.79 / 14.66 |
| 70% sparse, seed 42 | 3.36 M | 43.63 | 75.51 (9) | +4.46 (+1.00 to +8.22) | +3.03 | 70.52 | 16.2 | 50.76 / 14.65 |
| 70% sparse, seed 1 | 3.36 M | 43.63 | 74.65 (7) | +3.61 (−0.08 to +7.56) | +1.44 | 69.94 | 16.2 | 50.71 / 14.68 |
| 70% sparse, seed 2 | 3.36 M | 43.63 | 71.90 (8) | +0.86 (−2.38 to +4.12) | +1.03 | 67.63 | 16.2 | 50.82 / 14.73 |
| 90% sparse, seed 42 | 1.13 M | 11.41 | 71.03 (8) | −0.02 (−3.69 to +3.39) | −1.42 | 75.72 | 7.2 | 50.89 / 14.63 |

Every model has 11.18 M parameters in total, 1.82 GMACs and a 44.8 MB file, as FP32. The compressed size is the `state_dict` compressed with gzip (level 9).

**Best ratio on validation (70%), 3 seeds:**

| | FP32 | FP32 control, mean (min–max) | 70% sparse, mean (min–max) | Pruned − control, by seed 42 / 1 / 2 (mean) |
|---|---|---|---|---|
| Macro F1 | 71.04 | 72.27 (70.98–73.19) | 74.02 (71.90–75.51) | +3.03 / +1.44 / +1.03 (+1.83) |
| Balanced accuracy | 71.50 | 73.03 (71.85–74.26) | 73.86 (71.33–76.15) | +3.36 / −0.14 / −0.34 (+0.96) |
| Melanoma recall | 66.47 | 70.52 (69.94–71.10) | 69.36 (67.63–70.52) | −0.58 / 0.00 / −2.89 (−1.16) |
| Accuracy | 82.18 | 82.35 (81.79–82.70) | 83.49 (82.90–84.66) | +0.26 / +2.15 / +1.17 (+1.20) |

At epoch 10 for both models, macro F1 pruned − control is +2.02 / +0.69 / +2.58 (mean +1.76).

- **Up to 50%, unstructured pruning is free even without fine-tuning.** Right after pruning, macro F1 is 72.3 at 30% and 71.2 at 50%, the FP32 level. At 70% it falls to 43.6, and at 90% the model predicts "nv" for every image (macro F1 11.4).
- **After fine-tuning, no ratio up to 90% loses macro F1 against FP32.** Even at 90% sparsity (1.1 M non-zero weights), macro F1 is 71.03, against 71.04 for FP32.
- **70% gives no loss, and maybe a small gain.** On 3 seeds, macro F1 is 74.02 on average against 71.04 for FP32 (+3.0). Every seed beats its control, by +1.0 to +3.0 (mean +1.8). That is about the seed-to-seed spread (about 2 points), and only seed 42 is significant against FP32. The seed-42 result alone (+3.0 vs control) overstated the gain. Balanced accuracy is level with the control, and melanoma recall does not improve (−1.2 vs control on average).
- **No speed, memory or energy gain.** PyTorch runs the zeros through dense kernels, so latency stays within 0.6% of FP32 at every ratio, peak RAM within +3%, and estimated energy within −4%, inside the ±5% noise of this estimate. The file is the same 44.8 MB. The only real saving is storage once compressed: 16.2 MB at 70% (2.6× smaller than FP32 compressed) and 7.2 MB at 90% (5.7×). A speed-up would need sparse kernels or hardware that skip zeros.
- **Compared with structured pruning** (section 8): unstructured pruning keeps accuracy at far higher ratios (70–90% of weights against 10% of filters) but gives no speed-up, whereas structured pruning is 1.07× faster at 10% and 1.6× at 40%, at the cost of accuracy. Compared with INT8 QAT (section 7), which is 7.5× faster and 4× smaller with no loss, unstructured pruning only competes on compressed storage.

## Next steps

- Optional: quantize the pruned models (INT8 QAT or static PTQ) to combine both techniques: structured pruning + INT8 for speed, unstructured sparsity + INT8 for compressed size. Longer fine-tuning after structured pruning.
- Once every model is final, evaluate all of them once on the held-out test set (`python evaluation/evaluate.py --split test`).

## Reproduce

See [resnet18_finetunning/README.md](resnet18_finetunning/README.md). The model checkpoints (`.pth`) are gitignored and are not in the repository.
