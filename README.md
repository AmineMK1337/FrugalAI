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
python run_exp001.py
```

Trains ImageNet-pretrained ResNet-18 for 10 epochs (Adam, lr 1e-4, batch 32, class-weighted cross-entropy, augmentation) and keeps the checkpoint with the lowest validation loss as `best_model.pth`. Training history, validation metrics and Grad-CAM maps go to `results/baseline_resnet18/`. Checkpoints are gitignored, so run this before the notebook. Takes about 13 minutes on a GTX 1650.

The split is made on `lesion_id`, so images of the same lesion never land in two splits: **80% train** (7,974 images), **10% validation** (1,006, picks the best epoch) and **10% test** (1,035, used only by the notebook, never seen during training).

**2. Benchmark**

Open `evaluation.ipynb` with the `Python (skin-lesion venv)` kernel and run all cells. It loads `best_model.pth`, evaluates the test split, draws one figure per metric family, and exports everything to `results/metrics.json`.

## Results

Fine-tuned ResNet-18 (best epoch 9 of 10), evaluated on the 1,035 unseen test images. Inference on an NVIDIA GTX 1650 (Ryzen 7 5800H host).

| Performance | | Efficiency | | Environment | |
|---|---|---|---|---|---|
| Accuracy | 79.0% | Parameters | 11.18 M (42.7 MB) | Energy | 0.217 Wh (781 J) |
| Macro F1 | 0.654 | FLOPs | 3.65 GFLOPs / image | CO₂e | 0.122 g |
| Macro recall | 0.676 | Latency (batch 1) | 4.0 ± 0.3 ms | Per image | 0.75 J, 0.12 mg CO₂e |
| Macro specificity | 0.956 | Throughput (batch 32) | 694 images/s | Average power | 69 W |
| Macro AUC | 0.945 | Peak memory (batch 1) | 69 MB (GPU) | Grid intensity | 564 g CO₂e/kWh (Tunisia) |

Weighted F1 is 0.798. Per-class F1 ranges from 0.41 (dermatofibroma, 20 test images) to 0.89 (melanocytic nevi, 663 test images). Latency and energy vary slightly between runs.

## Credits

The training code in `skin-lesion-classification/` is adapted from [sairaveena28/skin-lesion-classification](https://github.com/sairaveena28/skin-lesion-classification), which builds on [sararajabzadeh/ham10000-classifier](https://github.com/sararajabzadeh/ham10000-classifier). HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

Educational project, not a clinical tool.
