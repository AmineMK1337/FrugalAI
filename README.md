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
├── run_exp001.py          # fine-tunes ResNet-18 on HAM10000
├── evaluation.ipynb       # all benchmarks, with plots in the cell outputs
├── configs/baseline.yaml  # training hyperparameters
├── src/ham10000/          # dataset, model, training and Grad-CAM code
├── results/
│   ├── baseline_resnet18/ # training history, metrics, confusion matrices, Grad-CAM
│   └── metrics.json       # benchmark results exported by the notebook
└── report.md              # written audit of the ResNet-18 run
```

## Setup

```bash
cd skin-lesion-classification
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt   # also installs src/ham10000 as a package
```

### Data

Download [HAM10000](https://doi.org/10.7910/DVN/DBW86T) and place it under `skin-lesion-classification/data/`:

```
data/
├── ham10000_metadata_2026-10-02.csv
└── ISIC-images/            # the .jpg images
```

The original `HAM10000_images_part_1/` and `HAM10000_images_part_2/` folders are also recognised. If your metadata file has a different name, update `metadata_csv` in `configs/baseline.yaml` and `METADATA_CSV` in the notebook's first cell. `data/` is gitignored.

## Usage

All commands run from `skin-lesion-classification/`.

**1. Fine-tune**

```bash
python run_exp001.py
```

Trains for 10 epochs (Adam, lr 1e-4, batch 32, class-weighted cross-entropy, augmentation) on a lesion-level 80/20 split, and keeps the checkpoint with the lowest validation loss as `best_model.pth`. Training history, metrics and Grad-CAM maps go to `results/baseline_resnet18/`. Checkpoints are gitignored, so run this before the notebook.

**2. Benchmark**

Open `evaluation.ipynb` with the `venv` kernel and run all cells. It loads `best_model.pth`, evaluates the held-out test images, draws one figure per metric family, and exports everything to `results/metrics.json`.

## Results

Fine-tuned ResNet-18, 1,193 test images, CPU inference (AMD Ryzen 7 5800H).

| Performance | | Efficiency | | Environment | |
|---|---|---|---|---|---|
| Accuracy | 76.1% | Parameters | 11.18 M (42.7 MB) | Energy | 0.57 Wh (2,052 J) |
| Macro F1 | 0.658 | FLOPs | 3.65 GFLOPs / image | CO₂e | 0.32 g |
| Macro recall | 0.758 | Latency (batch 1) | 30.9 ± 2.0 ms | Per image | 1.72 J, 0.27 mg CO₂e |
| Macro specificity | 0.957 | Throughput (batch 32) | 40 images/s | Average power | 58 W |
| Macro AUC | 0.961 | Peak memory (batch 1) | 49 MB | Grid intensity | 564 g CO₂e/kWh (Tunisia) |

Latency and energy vary by roughly 10–15% between runs on a laptop CPU.

**Caveat on the test split.** The notebook's test images are half of the 20% hold-out that training used as its validation set, and that set chose the best epoch. So the test scores are slightly optimistic. For a strictly unseen test set, retrain on the 80% split with a separate validation subset.

## Credits

The training code in `skin-lesion-classification/` is adapted from [sairaveena28/skin-lesion-classification](https://github.com/sairaveena28/skin-lesion-classification), which builds on [sararajabzadeh/ham10000-classifier](https://github.com/sararajabzadeh/ham10000-classifier). HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

Educational project, not a clinical tool.
