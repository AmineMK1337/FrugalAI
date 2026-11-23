# Pruning a fine-tuned ResNet-18 for skin-lesion classification (HAM10000): full summary

## 1. Context

- **Task:** 7-class classification of dermoscopy images (HAM10000, 10,015 images). The classes are very imbalanced: nv (melanocytic nevus) is 67% of the data; the rarest classes (df, vasc) are about 1% each.
- **Split:** fixed, by lesion (no lesion in two partitions), stratified by class, seed 42. Train 6,981 / validation 1,532 / test 1,502 images. **All results below are on the validation set.** The test set has not been used yet.
- **Baseline (FP32):** ImageNet-pretrained ResNet-18 fine-tuned 25 epochs (Adam, lr 1e-4 with cosine decay, batch 32, class-weighted cross-entropy, strong augmentation: any-angle rotation, flips, random crops 60–100%, colour jitter). Validation macro F1 71.04, accuracy 82.18.
- **Main metrics:** macro F1 and melanoma recall. Accuracy alone is misleading here, because predicting "nv" for every image already gives 66.45% accuracy.
- **Goal:** make the model cheaper (size, latency, memory, energy) without losing macro F1 or melanoma recall. Pruning was tested in two forms, both in FP32 (no quantization).

## 2. Evaluation protocol (same for every model)

- CPU only (Intel Xeon Gold 6348 VM, AVX-512 VNNI, PyTorch 2.12). Each model is fused (Conv+BN+ReLU) and exported to TorchScript. Fused vs unfused predictions were checked identical.
- **Performance:** accuracy, balanced accuracy (= macro recall), macro precision, macro specificity, macro F1, macro AUC, melanoma recall. 95% CIs by bootstrap (1,000 resamples). Change vs FP32: paired bootstrap, with CI and two-sided p-value.
- **Latency:** batch 1, 30 warm-up + 200 timed runs, median. Measured at 1 thread (main protocol) and 4 threads. **Throughput:** batch 32. **Peak RAM:** increase in resident memory while loading the model and warming up.
- **Energy:** CodeCarbon estimate (no hardware power counter on this VM; it assumes about 250 W per active thread, so energy ≈ run time × power). Comparisons only. Between identical architectures it varies by up to ±5%, so smaller differences are noise.
- Each model runs in a fresh process, 3 interleaved repeats, median reported. Structured and unstructured models were evaluated in **two separate sessions**, each re-measuring FP32 as its own timing reference.
- **FP32 control:** the unpruned FP32 model fine-tuned again with exactly the same 10-epoch recipe as the pruned models (seeds 42, 1, 2). Extra fine-tuning alone improves the model slightly, so "pruned − control" isolates the effect of pruning; "pruned − FP32" mixes pruning with extra training.

## 3. Structured pruning (removing whole filters)

1. Start from the FP32 baseline.
2. Torch-Pruning 1.6.1 builds the dependency graph. Convolutions joined by residual additions share channel indices and are pruned together.
3. **Criterion:** L1 norm of each output filter (Li et al., 2017). For channel groups shared by several convs, the norms are averaged per channel.
4. **Local ratio:** the same fraction (10, 20, 30, 40%) of filters removed in every prunable group. Not pruned: the stem `conv1` (which protects the layer1 residual stream; only the inner conv of each layer1 block is pruned) and the classifier `fc`.
5. Layers become physically smaller (fewer parameters and MACs).
6. **Fine-tune 10 epochs:** class-weighted cross-entropy, strong augmentation, Adam, lr 1e-4 cosine, batch 32, seed 42. The best ratio (10%) and the control were repeated with seeds 1 and 2.
7. Keep the epoch with the best validation macro F1.

## 4. Unstructured pruning (zeroing individual weights)

1. Start from the FP32 baseline.
2. **Criterion:** global L1 magnitude (`torch.nn.utils.prune.global_unstructured`, `L1Unstructured`) over the weights of every conv and of `fc`, ranked together across layers. The smallest X% (30, 50, 70, 90%) are set to exactly zero. Biases and BatchNorm are untouched.
3. Large layers end up sparser: at 70% global sparsity, per-layer sparsity runs from 33.8% (stem `conv1`) to 84.1% (`layer4.1.conv2`).
4. **Layer shapes do not change:** same parameter count, same MACs, same file size. The zeros are still stored and computed by dense kernels.
5. **Fine-tune 10 epochs** with the same recipe as structured pruning. The pruning mask stays on during training, so zeroed weights stay zero. Then the mask is made permanent.
6. Keep the epoch with the best validation macro F1. The best ratio (70%) was repeated with seeds 1 and 2.
7. Checks passed: achieved sparsity = target, and zero counts unchanged after fine-tuning, after fusion and in TorchScript.

## 5. Results before fine-tuning (right after pruning, no training)

Pruning is deterministic, so these numbers are the same for every seed. All values in %.

| Model | Accuracy | Balanced acc. | Macro precision | Macro specificity | Macro F1 | Macro AUC | Melanoma recall | Predictions identical to FP32 |
|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 82.18 | 71.50 | 72.31 | 96.19 | 71.04 | 96.34 | 66.47 | 100 |
| Structured 10% | 69.71 | 36.25 | 50.37 | 92.15 | 37.96 | 85.69 | 43.93 | 70.8 |
| Structured 20% | 59.07 | 19.69 | 16.23 | 89.05 | 16.74 | 62.15 | 6.94 | 58.1 |
| Structured 30% | 64.95 | 14.04 | 9.90 | 85.65 | 11.44 | 49.72 | 0.00 | 61.6 |
| Structured 40% | 66.45 | 14.29 | 9.49 | 85.71 | 11.41 | 53.66 | 0.00 | 62.3 |
| Unstructured 30% | 83.49 | 72.18 | 73.35 | 96.41 | 72.25 | 96.35 | 63.01 | 96.4 |
| Unstructured 50% | 84.20 | 66.20 | 79.27 | 95.71 | 71.16 | 96.01 | 60.69 | 90.6 |
| Unstructured 70% | 78.33 | 37.76 | 70.42 | 92.42 | 43.63 | 93.94 | 47.40 | 77.1 |
| Unstructured 90% | 66.45 | 14.29 | 9.49 | 85.71 | 11.41 | 60.67 | 0.00 | 62.3 |

Structured "before" values come from the GPU validation pass of the training script; unstructured ones from the CPU evaluation (the two agree within one or two images).

**Note on unstructured 30–50% before fine-tuning:** accuracy goes up (83.5% and 84.2% vs 82.2%) even though no training happened. The non-zero weights were verified to be bit-identical to FP32. The cause is a shift toward the majority class: the baseline, trained with class-weighted loss, over-predicts rare classes (e.g. "mel" 204 times for 173 true melanomas), and pruning partly undoes this bias. At 50%, "nv" is predicted 1,060 times instead of 955: 60 more correct nv images and 31 fewer correct rare-class images. Accuracy rises, but balanced accuracy (−5.3, significant) and melanoma recall (−5.8, significant) fall. At 30%, melanoma recall falls by 3.5 points (significant) while macro F1 is unchanged.

## 6. Results after fine-tuning: performance (validation, CPU, all values in %)

Pruned − control is the macro F1 difference with the FP32 control of the same seed (CPU evaluation, selected epochs).

| Model | Accuracy | Balanced acc. | Macro precision | Macro specificity | Macro F1 (95% CI) | Macro AUC | Melanoma recall | Δ macro F1 vs FP32 (95% CI, p) | Pruned − control, macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 82.18 | 71.50 | 72.31 | 96.19 | 71.04 (66.27–74.96) | 96.34 | 66.47 | – | – |
| FP32 control, seed 42 | 82.70 | 72.97 | 74.81 | 96.46 | 72.65 | 96.46 | 71.10 | +1.60 (−1.62 to +4.86), p=0.30 | – |
| FP32 control, seed 1 | 82.57 | 74.26 | 73.49 | 96.28 | 73.19 | 95.69 | 69.94 | +2.15 (−1.39 to +5.64), p=0.25 | – |
| FP32 control, seed 2 | 81.79 | 71.85 | 72.99 | 96.11 | 70.98 | 95.97 | 70.52 | −0.07 (−3.33 to +3.43), p=0.99 | – |
| Structured 10%, seed 42 | 81.92 | 71.17 | 71.43 | 96.32 | 70.34 | 96.17 | 75.72 | −0.70 (−4.40 to +2.88), p=0.71 | −2.31 |
| Structured 10%, seed 1 | 82.90 | 69.08 | 71.84 | 96.20 | 69.42 | 96.04 | 71.10 | −1.63 (−6.03 to +2.47), p=0.45 | −3.78 |
| Structured 10%, seed 2 | 81.53 | 72.56 | 71.41 | 96.13 | 71.41 | 96.28 | 69.36 | +0.36 (−3.32 to +3.86), p=0.81 | +0.43 |
| Structured 20%, seed 42 | 78.98 | 68.67 | 69.00 | 95.79 | 67.25 | 95.57 | 72.25 | −3.79 (−7.64 to +0.23), p=0.06 | −5.40 |
| Structured 30%, seed 42 | 76.70 | 67.03 | 61.78 | 95.54 | 63.25 | 94.54 | 73.41 | −7.80 (−12.00 to −3.58), p=0.004 | −9.40 |
| Structured 40%, seed 42 | 73.17 | 66.88 | 58.03 | 95.10 | 60.66 | 92.83 | 61.85 | −10.39 (−14.25 to −6.64), p<0.001 | −11.99 |
| Unstructured 30%, seed 42 | 82.77 | 72.57 | 73.64 | 96.09 | 72.37 | 96.17 | 69.94 | +1.33 (−2.36 to +5.26), p=0.49 | −0.28 |
| Unstructured 50%, seed 42 | 81.98 | 74.91 | 72.06 | 96.21 | 72.76 | 95.97 | 63.58 | +1.71 (−1.54 to +5.14), p=0.28 | +0.11 |
| Unstructured 70%, seed 42 | 82.90 | 76.15 | 75.75 | 96.36 | 75.51 | 96.37 | 70.52 | +4.46 (+1.00 to +8.22), p=0.014 | +2.86 |
| Unstructured 70%, seed 1 | 84.66 | 74.11 | 77.08 | 96.45 | 74.65 | 96.16 | 69.94 | +3.61 (−0.08 to +7.56), p=0.056 | +1.46 |
| Unstructured 70%, seed 2 | 82.90 | 71.33 | 73.91 | 96.29 | 71.90 | 95.74 | 67.63 | +0.86 (−2.38 to +4.12), p=0.58 | +0.92 |
| Unstructured 90%, seed 42 | 79.63 | 75.87 | 68.95 | 96.09 | 71.03 | 95.49 | 75.72 | −0.02 (−3.69 to +3.39), p=0.99 | −1.62 |

**Three-seed summaries (macro F1):**
- FP32 control: mean 72.27 (70.98–73.19).
- Structured 10%: mean 70.39 (69.42–71.41); pruned − control mean −1.89 (−3.78 to +0.43, sign changes with seed).
- Unstructured 70%: mean 74.02 (71.90–75.51); pruned − control mean +1.75 (+0.92 to +2.86, same sign on all seeds). Melanoma recall mean 69.36 vs control mean 70.52.

## 7. Results after fine-tuning: efficiency (CPU)

GMACs per image. Latency in ms (median, batch 1). Throughput at batch 32, 1 thread. Energy is the CodeCarbon estimate per 1,000 images, 1 thread (CO₂e uses 564 gCO₂e/kWh, Tunisia's grid). Changes vs the FP32 of the same session.

**Session A: structured pruning**

| Model | Params | GMACs | File size (MB) | Latency 1 thread (p95) | Latency 4 threads | Speed-up 1 / 4 threads | Throughput (img/s) | Peak RAM (MB) | Energy (Wh / 1,000 img) | CO₂e (g / 1,000 img) |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 11.18 M | 1.82 | 44.80 | 50.94 (51.57) | 14.55 | 1.00 / 1.00 | 20.7 | 77.8 | 3.41 | 1.92 |
| FP32 control, seed 42 | 11.18 M | 1.82 | 44.82 | 50.75 (51.40) | 14.67 | 1.00 / 0.99 | 21.0 | 77.1 | 3.36 | 1.89 |
| FP32 control, seed 1 | 11.18 M | 1.82 | 44.82 | 50.82 (51.57) | 14.66 | 1.00 / 0.99 | 21.2 | 78.3 | 3.25 | 1.83 |
| FP32 control, seed 2 | 11.18 M | 1.82 | 44.82 | 50.86 (51.38) | 14.69 | 1.00 / 0.99 | 20.8 | 76.6 | 3.43 | 1.93 |
| Structured 10%, seed 42 | 9.05 M (−19%) | 1.54 (−16%) | 36.28 | 47.82 (48.41) | 14.04 | 1.07 / 1.04 | 21.6 | 70.4 (−9%) | 3.26 (−5%) | 1.84 |
| Structured 10%, seed 1 | 9.05 M | 1.54 | 36.28 | 47.20 (47.77) | 14.06 | 1.08 / 1.04 | 22.3 | 68.7 (−12%) | 3.17 (−7%) | 1.79 |
| Structured 10%, seed 2 | 9.05 M | 1.54 | 36.28 | 47.64 (48.16) | 14.01 | 1.07 / 1.04 | 21.7 | 68.8 (−11%) | 3.21 (−6%) | 1.81 |
| Structured 20%, seed 42 | 7.17 M (−36%) | 1.29 (−29%) | 28.75 | 43.17 (44.33) | 13.26 | 1.18 / 1.10 | 24.0 | 62.4 (−20%) | 2.91 (−15%) | 1.64 |
| Structured 30%, seed 42 | 5.52 M (−51%) | 1.06 (−42%) | 22.15 | 36.80 (37.76) | 11.66 | 1.38 / 1.25 | 30.0 | 56.1 (−28%) | 2.34 (−31%) | 1.32 |
| Structured 40%, seed 42 | 4.07 M (−64%) | 0.85 (−53%) | 16.37 | 31.92 (32.81) | 10.43 | 1.60 / 1.40 | 32.2 | 50.9 (−35%) | 2.15 (−37%) | 1.21 |

**Session B: unstructured pruning** (every model: 11.18 M parameters in total, 1.82 GMACs, 44.8 MB file; compressed size = `state_dict` gzip level 9)

| Model | Non-zero params | Sparsity | Compressed size (MB) | Latency 1 thread (p95) | Latency 4 threads | Speed-up 1 / 4 threads | Throughput (img/s) | Peak RAM (MB) | Energy (Wh / 1,000 img) | CO₂e (g / 1,000 img) |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 baseline | 11.18 M | 0% | 41.56 | 50.72 (51.26) | 14.64 | 1.00 / 1.00 | 21.1 | 76.7 | 3.39 | 1.91 |
| Unstructured 30%, seed 42 | 7.83 M | 30% | 31.98 | 50.87 (51.63) | 14.68 | 1.00 / 1.00 | 21.4 | 76.5 | 3.32 (−2%) | 1.87 |
| Unstructured 50%, seed 42 | 5.59 M | 50% | 24.30 | 50.79 (51.31) | 14.66 | 1.00 / 1.00 | 21.5 | 77.1 | 3.26 (−4%) | 1.84 |
| Unstructured 70%, seed 42 | 3.36 M | 70% | 16.20 | 50.76 (51.58) | 14.65 | 1.00 / 1.00 | 21.1 | 77.7 | 3.28 (−3%) | 1.85 |
| Unstructured 70%, seed 1 | 3.36 M | 70% | 16.20 | 50.71 (51.21) | 14.68 | 1.00 / 1.00 | 21.0 | 77.1 | 3.32 (−2%) | 1.87 |
| Unstructured 70%, seed 2 | 3.36 M | 70% | 16.20 | 50.82 (51.41) | 14.73 | 1.00 / 0.99 | 21.2 | 78.7 | 3.32 (−2%) | 1.87 |
| Unstructured 90%, seed 42 | 1.13 M | 90% | 7.23 | 50.89 (51.57) | 14.63 | 1.00 / 1.00 | 21.5 | 77.4 | 3.29 (−3%) | 1.86 |

For reference, FP32 compressed with gzip is 41.56 MB, so unstructured pruning saves 23% (30%), 42% (50%), 61% (70%) and 83% (90%) of compressed storage.

## 8. Reference point: INT8 quantization-aware training (different evaluation session)

INT8 QAT of the same baseline: macro F1 71.18 (+0.14 vs FP32, p=0.90), melanoma recall 73.41, 11.32 MB (4× smaller), 6.78 ms at 1 thread (7.5× faster), −31% peak RAM, about −86% estimated energy.

## 9. Main findings

1. **Fine-tuning is essential for structured pruning.** Right after pruning, even 10% of filters drops macro F1 from 71.0 to 38.0; at 30–40% the model predicts "nv" for nearly every image. Ten epochs recover most of the loss.
2. **Structured 10% is close to free vs FP32** (mean −0.65 macro F1 over 3 seeds, none significant), but **about −1.9 vs its control** (sign changes with seed, so not established). From 20%, the loss grows with the ratio and is significant from 30% (−7.8 at 30%, −10.4 at 40%).
3. **Structured speed-up is smaller than the compute reduction:** GMACs −16 to −53%, latency only −6 to −37% (1.07–1.60× at 1 thread). layer1 (highest resolution) is only half prunable because its residual stream is protected.
4. **Unstructured pruning keeps accuracy at much higher ratios.** Up to 30%, macro F1 is unchanged even without fine-tuning (though melanoma recall drops 3.5 points). After fine-tuning, every ratio up to 90% is at or above FP32 on macro F1.
5. **Unstructured 70% over 3 seeds:** macro F1 74.02 (+3.0 vs FP32, significant for seed 42 only), +1.75 vs control on average with the same sign on all seeds. The gain is about the size of the seed-to-seed spread (about 2 points), so it is "no loss, possibly a small gain". Melanoma recall does not improve vs the control (−1.2 on average).
6. **Unstructured pruning gives no runtime benefit** on this CPU with PyTorch dense kernels: latency, RAM and energy are within noise of FP32. The only gain is compressed storage (2.6× smaller at 70%, 5.7× at 90%).
7. **Melanoma recall is unstable:** it moves by up to 6 points between seeds of the same model, and much of the rise seen after pruning comes from the extra fine-tuning (the controls rise too).
8. **For this model on this CPU, INT8 QAT dominates both pruning methods** on speed and size with no accuracy loss. Pruning and quantization can be combined (the pruned models keep the quantizable architecture).

## 10. Limitations

- All results are on the validation set, which was also used to choose the ratio and the epoch (best macro F1), so the selected scores are slightly optimistic. The held-out test set has not been used yet.
- Only the best ratio of each method has 3 seeds; the other ratios have 1 seed.
- Energy is an estimate (time × assumed power), not a measurement.
- Structured and unstructured results come from two evaluation sessions; FP32 latency differs by 0.2 ms between them.
- Fine-tuning was limited to 10 epochs; most structured models were still improving at epochs 9–10.
- Structured pruning used a uniform per-layer ratio and the L1 filter criterion only; unstructured used global magnitude only (no iterative pruning, no other criteria).

## 11. Possible next steps

- Combine pruning and INT8 quantization (structured + INT8 for speed, unstructured sparsity + INT8 for compressed size).
- Longer fine-tuning after structured pruning; iterative or non-uniform pruning.
- Sparse kernels or hardware that skip zeros, to turn unstructured sparsity into speed.
- Final single evaluation of all chosen models on the test set.
