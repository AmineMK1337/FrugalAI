# FrugalAI

Fine-tuning ResNet-18 on the HAM10000 skin-lesion dataset and benchmarking it on three fronts: diagnostic **performance**, computational **efficiency**, and **environmental** footprint.

| Family | Metrics |
|---|---|
| Performance | Accuracy, Precision, Recall, Specificity, F1-Score, AUC (macro, weighted and per class) |
| Efficiency | Size, FLOPs, Latency, Throughput, Memory |
| Environment | Energy (kWh), CO₂e (kg) |

## What was done

1. **Correct dataset and a clean split.** The model was first trained on an ISIC archive export (11,720 images, diagnoses mapped from free-text fields). It was retrained on the official HAM10000 release (10,015 images, 7 classes). The split is made by lesion, 80/10/10, so the test images are never seen during training or model selection. HAM10000 is strongly imbalanced: melanocytic nevi are 66.9% of the images, dermatofibroma 1.1% (58×). Training therefore uses class-weighted cross-entropy, and macro metrics are reported.
2. **Data augmentation and training length.** Four FP32 setups were compared: standard vs strong augmentation, 10 epochs vs 25 epochs with a cosine schedule. Strong augmentation with 25 epochs (**model D**) is best on test: accuracy 81.5% and macro F1 0.686, against 79.0% and 0.654 for the starting setup. Neither change helps on its own. Details in [comparaison.md](comparaison.md).
3. **A reproducible evaluation pipeline** on CPU (4 threads):
   - predictive metrics with 95% bootstrap confidence intervals;
   - model size, MACs, latency (30 warm-up + 200 runs), throughput at batch 32;
   - memory measured in a fresh process;
   - CPU + RAM energy with CodeCarbon, 3 repeats of 60 s.

   Each run records how much CPU other programs used, and a check before and after the session flags drifting timings.
4. **INT8 post-training quantization (PTQ).** Calibration size (128 to 2,048 training images) and weight granularity (per-channel or per-tensor) were chosen on validation macro F1. The first attempt used the onednn default qconfig and lost up to 0.11 validation macro F1: full-range activations overflow in INT8 arithmetic on this AVX2-only CPU, which has no VNNI. The x86 qconfig (`reduce_range=True`) fixes it. The selected setting is per-channel weights with 128 calibration images.
5. **INT8 quantization-aware training (QAT).** 5 epochs from model D, the best INT8 epoch chosen on validation. A control run (the same 5 epochs in FP32, without quantization) gained as much on validation as QAT did. So QAT's small gain comes from the extra training, not from quantization.
6. **A test-set comparison** of FP32, PTQ and QAT on the same 1,035 images, with paired bootstrap tests of the differences.

## Repository layout

```
skin-lesion-classification/
├── finetune_resnet18.py      # fine-tunes ResNet-18 on HAM10000 (model D by default)
├── quantize_ptq.py           # INT8 post-training quantization (+ ablation on validation)
├── train_qat.py              # INT8 quantization-aware training (+ FP32 control run)
├── validation_summary.py     # validation scores of FP32 / PTQ / QAT / control
├── evaluate_model.ipynb      # test-set evaluation of one model, with plots
├── run_evaluations.py        # runs evaluate_model.ipynb for FP32, PTQ and QAT back to back
├── compare_models.ipynb      # FP32 vs INT8 comparison table, significance tests, plots
├── configs/                  # strong_cosine25.yaml = model D; the other three are the compared setups
├── src/ham10000/             # data, model, training, quantization, memory/speed probes
└── results/
    ├── strong_cosine25_resnet18/  # model D (FP32): checkpoint, training history, Grad-CAM
    ├── ptq_int8/, qat_int8/       # INT8 models, ablation, training history, settings
    ├── eval_FP32/, eval_INT8-*/   # test-set evaluation of each model
    ├── comparison/                # FP32 vs INT8 comparison
    └── <other runs>/              # the three FP32 setups model D was compared against
```

## Setup

```bash
cd skin-lesion-classification
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt   # also installs src/ham10000 as a package
```

On an NVIDIA GPU, replace the CPU build of PyTorch with the CUDA one:

```bash
pip install --force-reinstall --no-deps torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

To run the notebook in this venv, register it as a Jupyter kernel (the notebook expects this kernel name):

```bash
python -m ipykernel install --user --name skin-lesion-venv --display-name "Python (skin-lesion venv)"
```

### Data

Download [HAM10000](https://doi.org/10.7910/DVN/DBW86T) and place it in `dataset/` at the repository root, next to `skin-lesion-classification/`:

```
dataset/
├── HAM10000_metadata            # metadata CSV (10,015 rows)
├── HAM10000_images_part_1/      # 5,000 .jpg images
└── HAM10000_images_part_2/      # 5,015 .jpg images
```

If you store it elsewhere, update `metadata_csv` and `data_dir` in the files in `configs/`. `dataset/` is gitignored.

## Usage

All commands run from `skin-lesion-classification/`.

**1. Fine-tune**

```bash
python finetune_resnet18.py
```

Trains the chosen model, **model D** (`configs/strong_cosine25.yaml`): ImageNet-pretrained ResNet-18, 25 epochs, Adam with a cosine learning-rate schedule (1e-4 → 0), batch 32, class-weighted cross-entropy, and strong augmentation (any-angle rotation, horizontal and vertical flips, 60–100% crops, mild colour jitter). The checkpoint with the lowest validation loss is saved as `resnet18_baseline.pth`. Training history, validation metrics and Grad-CAM maps go to `results/strong_cosine25_resnet18/`. Checkpoints are gitignored, so run this before the notebook. Takes about 32 minutes on a GTX 1650.

Model D was picked from four training setups; see [comparaison.md](comparaison.md). The other setups can be rerun with `python finetune_resnet18.py --config configs/<name>.yaml --results-dir results/<name>`.

The split is made on `lesion_id`, so images of the same lesion never land in two splits: **80% train** (7,974 images), **10% validation** (1,006, picks the best epoch) and **10% test** (1,035, used only by the notebook, never seen during training).

**2. Quantize to INT8**

```bash
python quantize_ptq.py           # PTQ ablation, chosen on validation macro F1
python quantize_ptq.py --save    # save the selected PTQ model
python train_qat.py              # QAT, 5 epochs on GPU
python train_qat.py --control    # FP32 control: same 5 epochs, no quantization
python validation_summary.py     # compare all four on the validation split
```

Eager-mode quantization of torchvision's quantizable ResNet-18 with the **onednn engine and the x86 qconfig (`reduce_range=True`)**. The x86 engine needs FBGEMM, which this PyTorch build lacks. `reduce_range=True` keeps INT8 dot products from overflowing on CPUs without VNNI, such as this AVX2-only Ryzen; the ablation with the onednn default qconfig is kept in `results/ptq_int8/ptq_ablation_onednn_qconfig.csv` as evidence. The test split is never used for any of these choices.

**3. Evaluate on the test set**

```bash
python run_evaluations.py        # evaluate_model.ipynb for FP32, PTQ and QAT, back to back on CPU (4 threads)
```

Then run `compare_models.ipynb`. Keep the laptop on mains power and close busy programs first: latency and energy are sensitive to background load, and the evaluation warns when another program uses more than one CPU core or when speed drifts during the session.

## Results

Test set: 1,035 images never seen during training or model selection. Costs measured on CPU (Ryzen 7 5800H, 4 threads) in a session whose timings stayed stable within ±2%.

| | FP32 (model D) | INT8-PTQ | INT8-QAT |
|---|---|---|---|
| Accuracy | 81.5% | 81.3% | 80.9% |
| Macro F1 | 0.686 | 0.688 | 0.675 |
| Macro AUC | 0.953 | 0.951 | 0.955 |
| Melanoma recall | 65.7% | 67.7% | 71.7% |
| Model size | 42.7 MB | 10.8 MB | 10.8 MB |
| Latency, 1 image (median) | 24.1 ms | 6.8 ms | 6.7 ms |
| Throughput, batch 32 | 47.6 img/s | 186.3 img/s | 189.7 img/s |
| Peak memory, batch 32 | 394 MB | 96 MB | 96 MB |
| Energy, CPU + RAM | 0.294 Wh / 1,000 img | 0.088 Wh / 1,000 img | 0.089 Wh / 1,000 img |
| CO₂e (564 g/kWh, Tunisia) | 0.166 g / 1,000 img | 0.049 g / 1,000 img | 0.050 g / 1,000 img |

- **Accuracy:** no INT8 difference is statistically significant (paired bootstrap, `results/comparison/paired_bootstrap.csv`). INT8-PTQ gives the same prediction as FP32 for 96.2% of test images.
- **Costs:** INT8 is 4× smaller, uses about 4× less memory, and is roughly 3–4× faster and 3× cheaper in energy.
- **QAT brings no gain over PTQ here,** so PTQ is the recommended INT8 model.
- **Per-class scores for the rarest classes** (14 vasc and 20 df test images) move by 5–7 points per image, so per-class differences between models should not be over-read.

Full tables and plots: `results/comparison/`.

### Measurement caveats

The accuracy results are final. The cost ratios are slightly optimistic for INT8:

- **Runtime.** FP32 is timed as an eager PyTorch model and INT8 as TorchScript. FP32 in TorchScript is about 1.18× faster than eager, so the speed-up from quantization alone is **about 3.3×**.
- **Idle power.** The laptop draws about 35 W with no model running, and the slower model is charged more of that per image. Counting only the energy the model adds above that floor, the saving is **about 2.7×**; the 3.4× in the table is the total machine energy per image.
- **RAM power** is estimated by CodeCarbon as a constant 10 W, not measured. CPU power is measured (Windows EMI).
- **Background load.** Timings depend on what else is running. Busy programs (a browser, cloud sync) shifted latency by up to 80% between sessions, so measure on mains power with other programs closed.

## Next steps

- Benchmark FP32 as TorchScript, measure the no-model power floor, and interleave FP32/PTQ/QAT measurements, to make the cost ratios exact.
- Try a compression that removes operations rather than lowering their precision: structured pruning (then INT8), lower input resolution, or distillation into a smaller network.
- Measure training energy with CodeCarbon, so the lifecycle footprint is complete.
- Tune the melanoma decision threshold on validation, to trade false alarms for fewer missed melanomas.

## Credits

The training code in `skin-lesion-classification/` is adapted from [sairaveena28/skin-lesion-classification](https://github.com/sairaveena28/skin-lesion-classification), which builds on [sararajabzadeh/ham10000-classifier](https://github.com/sararajabzadeh/ham10000-classifier). HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

Educational project, not a clinical tool.
