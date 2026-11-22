| Model | Sparsity (target / achieved) | Non-zero params | Macro F1: before → selected (epoch) / epoch 10 | Δ macro F1 vs FP32 (95% CI) | Pruned − control (selected / epoch 10) | Mel. recall | State dict dense / gzip (MB) | Latency 1t / 4t (ms) | Peak RAM 1t (MB) | Energy (Wh/1000, est.) |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | 0 / 0.00% | 11,180,103 | 71.04 | – | – | 66.47 | 44.80 / 41.56 | 50.72 / 14.64 | 77 | 3.39 |
| sparse30_seed42 | 30 / 30.00% | 7,828,954 | 72.25 → 72.37 (7) / 71.11 | +1.33 (-2.36 to +5.26) | -0.11 / -1.02 | 63.01 → 69.94 | 44.80 / 31.98 | 50.87 / 14.68 | 76 | 3.32 |
| sparse50_seed42 | 50 / 50.00% | 5,594,855 | 71.16 → 72.76 (5) / 71.40 | +1.71 (-1.54 to +5.14) | +0.28 / -0.72 | 60.69 → 63.58 | 44.80 / 24.30 | 50.79 / 14.66 | 77 | 3.26 |
| sparse70_seed42 | 70 / 70.00% | 3,360,756 | 43.63 → 75.51 (9) / 74.15 | +4.46 (+1.00 to +8.22) | +3.03 / +2.02 | 47.40 → 70.52 | 44.80 / 16.20 | 50.76 / 14.65 | 78 | 3.28 |
| sparse90_seed42 | 90 / 90.00% | 1,126,657 | 11.41 → 71.03 (8) / 70.15 | -0.02 (-3.69 to +3.39) | -1.42 / -1.98 | 0.00 → 75.72 | 44.80 / 7.23 | 50.89 / 14.63 | 77 | 3.29 |
| sparse70_seed1 | 70 / 70.00% | 3,360,756 | 43.63 → 74.65 (7) / 73.89 | +3.61 (-0.08 to +7.56) | +1.44 / +0.69 | 47.40 → 69.94 | 44.80 / 16.20 | 50.71 / 14.68 | 77 | 3.32 |
| sparse70_seed2 | 70 / 70.00% | 3,360,756 | 43.63 → 71.90 (8) / 71.86 | +0.86 (-2.38 to +4.12) | +1.03 / +2.58 | 47.40 → 67.63 | 44.80 / 16.20 | 50.82 / 14.73 | 79 | 3.32 |

Before fine-tuning: CPU (this script); selected: CPU (evaluate.py) when evaluated, else GPU (prune_unstructured.py); epoch 10 and pruned − control: GPU validation of the training scripts (control = pruning/structured control_seed<seed>). Validation set, 1,532 images. Energy: CodeCarbon estimate, no power counter.
