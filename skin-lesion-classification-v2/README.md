# FrugalAI: ResNet-18 on HAM10000 (stratified split)

Fine-tuning an ImageNet-pretrained ResNet-18 on the HAM10000 skin-lesion dataset (10,015 images, 7 classes), on a fixed lesion-level, class-stratified split.

## Layout

```
skin-lesion-classification-v2/
├── make_splits.py            # builds the fixed 70/15/15 split in splits/
├── finetune_resnet18.py      # fine-tunes ResNet-18 from a config
├── configs/
│   ├── strong_aug.yaml       # strong augmentation
│   └── strong_cosine25.yaml  # strong augmentation, 25 epochs, cosine LR schedule
├── splits/                   # split CSVs, summary, config and integrity report
├── src/ham10000/             # data, model, training, evaluation, Grad-CAM
└── results/
    ├── strong_aug_resnet18/
    └── strong_cosine25_resnet18/   # training history, metrics, confusion matrices, Grad-CAM
```

## Setup

```bash
cd skin-lesion-classification-v2
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt   # also installs src/ham10000 as a package
```

Download [HAM10000](https://doi.org/10.7910/DVN/DBW86T) into `dataset/` at the repository root (`HAM10000_metadata`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/`). `dataset/` is gitignored.

## Usage

```bash
python make_splits.py
python finetune_resnet18.py --config configs/strong_aug.yaml      --results-dir results/strong_aug_resnet18
python finetune_resnet18.py --config configs/strong_cosine25.yaml --results-dir results/strong_cosine25_resnet18
```

**Split.** Lesions, not images, are split, stratified by diagnosis (seed 42): 70% train (6,981 images), 15% validation (1,532), 15% test (1,502). Every image of a lesion stays in one partition. See `splits/split_summary.csv` and `splits/integrity_report.txt`.

**Training.** Adam, LR 1e-4, batch 32, class-weighted cross-entropy, strong augmentation. The checkpoint with the lowest validation loss is saved as `resnet18_baseline.pth` in the results folder. Checkpoints are gitignored.

## Results (validation split, 1,532 images)

| Run | Accuracy | Macro F1 | Weighted F1 | Best epoch |
|---|---|---|---|---|
| strong_aug | 79.9% | 0.678 | 0.809 | 19 |
| **strong_cosine25** | **82.2%** | **0.710** | **0.827** | 19 |

Per-class reports, confusion matrices and Grad-CAM maps are in each `results/` folder.

## Credits

Training code adapted from [sairaveena28/skin-lesion-classification](https://github.com/sairaveena28/skin-lesion-classification), which builds on [sararajabzadeh/ham10000-classifier](https://github.com/sararajabzadeh/ham10000-classifier). HAM10000: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

Educational project, not a clinical tool.
