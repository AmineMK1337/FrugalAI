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
├── finetune_resnet18.py   # fine-tunes ResNet-18 on HAM10000 (model D by default)
├── evaluation.ipynb       # all benchmarks, with plots in the cell outputs
├── configs/               # strong_cosine25.yaml = model D; the other three are the compared setups
├── src/ham10000/          # dataset, model, training and Grad-CAM code
├── results/
│   ├── strong_cosine25_resnet18/  # model D: training history, metrics, confusion matrices, Grad-CAM
│   ├── <other runs>/              # the three setups model D was compared against
│   ├── comparison.json            # test-set scores of all four setups
│   └── metrics.json               # benchmark results exported by the notebook
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
python finetune_resnet18.py
```

Trains the chosen model, **model D** (`configs/strong_cosine25.yaml`): ImageNet-pretrained ResNet-18, 25 epochs, Adam with a cosine learning-rate schedule (1e-4 → 0), batch 32, class-weighted cross-entropy, and strong augmentation (any-angle rotation, horizontal and vertical flips, 60–100% crops, mild colour jitter). The checkpoint with the lowest validation loss is saved as `resnet18_baseline.pth`. Training history, validation metrics and Grad-CAM maps go to `results/strong_cosine25_resnet18/`. Checkpoints are gitignored, so run this before the notebook. Takes about 32 minutes on a GTX 1650.

Model D was picked from four training setups; see [comparaison.md](comparaison.md). The other setups can be rerun with `python finetune_resnet18.py --config configs/<name>.yaml --results-dir results/<name>`.

The split is made on `lesion_id`, so images of the same lesion never land in two splits: **80% train** (7,974 images), **10% validation** (1,006, picks the best epoch) and **10% test** (1,035, used only by the notebook, never seen during training).

**2. Benchmark**

Open `evaluation.ipynb` with the `Python (skin-lesion venv)` kernel and run all cells. It loads `resnet18_baseline.pth`, evaluates the test split, draws one figure per metric family, and exports everything to `results/metrics.json`.

## Results

Model D (best epoch 25 of 25), evaluated on the 1,035 unseen test images. Inference on an NVIDIA GTX 1650 (Ryzen 7 5800H host).

| Performance | | Efficiency | | Environment | |
|---|---|---|---|---|---|
| Accuracy | 81.5% | Parameters | 11.18 M (42.7 MB) | Energy | 0.229 Wh (825 J) |
| Macro F1 | 0.686 | FLOPs | 3.65 GFLOPs / image | CO₂e | 0.129 g |
| Macro recall | 0.699 | Latency (batch 1) | 5.6 ± 0.4 ms | Per image | 0.80 J, 0.12 mg CO₂e |
| Macro specificity | 0.961 | Throughput (batch 32) | 721 images/s | Average power | 60 W |
| Macro AUC | 0.953 | Peak memory (batch 1) | 69 MB (GPU) | Grid intensity | 564 g CO₂e/kWh (Tunisia) |

Weighted F1 is 0.819. Per-class F1 ranges from 0.55 (actinic keratosis, 35 test images) to 0.91 (melanocytic nevi, 663 test images). Latency varies by a millisecond or two between runs on this GPU.

## Credits

The training code in `skin-lesion-classification/` is adapted from [sairaveena28/skin-lesion-classification](https://github.com/sairaveena28/skin-lesion-classification), which builds on [sararajabzadeh/ham10000-classifier](https://github.com/sararajabzadeh/ham10000-classifier). HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

Educational project, not a clinical tool.
