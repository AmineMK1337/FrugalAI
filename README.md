# FrugalAI

Fine-tuning ResNet-18 on the HAM10000 skin-lesion dataset and benchmarking it on three fronts: diagnostic **performance**, computational **efficiency**, and **environmental** footprint.

| Family | Metrics |
|---|---|
| Performance | Accuracy, Precision, Recall, Specificity, F1-Score, AUC (macro, weighted and per class) |
| Efficiency | Size, FLOPs, Latency, Throughput, Memory |
| Environment | Energy (kWh), CO₂e (kg) |

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

If you store it elsewhere, update `metadata_csv` and `data_dir` in `configs/baseline.yaml`, and `DATA_DIR` / `METADATA_CSV` in the notebook's first cell. `dataset/` is gitignored.

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

Test set: 1,035 images never seen during training or model selection.

| | FP32 (model D) | INT8-PTQ | INT8-QAT |
|---|---|---|---|
| Accuracy | 81.5% | 81.3% | 80.9% |
| Macro F1 | 0.686 | 0.688 | 0.675 |
| Macro AUC | 0.953 | 0.951 | 0.955 |
| Model size | 42.7 MB | 10.8 MB | 10.8 MB |

No INT8 accuracy difference is statistically significant (paired bootstrap, `results/comparison/paired_bootstrap.csv`). QAT brings no gain over PTQ here: its small validation gain matched what 5 extra FP32 epochs give on their own. Latency, throughput, memory and energy are in `results/comparison/comparison_table.csv`; INT8 runs about 3x faster on CPU.

## Credits

The training code in `skin-lesion-classification/` is adapted from [sairaveena28/skin-lesion-classification](https://github.com/sairaveena28/skin-lesion-classification), which builds on [sararajabzadeh/ham10000-classifier](https://github.com/sararajabzadeh/ham10000-classifier). HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

Educational project, not a clinical tool.
