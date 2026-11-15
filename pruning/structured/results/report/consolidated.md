| Model | Target | Channels − | Params − | GMACs − | Bal. acc. (before → after) | Macro F1 (before → after) | Mel. recall (before → after) | Epoch | Macro F1, epoch 10 | Size (MB) | Latency 1t / 4t (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 0% | 0.0% | 0.0% | 0.0% | 71.50 → 71.50 | 71.04 → 71.04 | 66.47 → 66.47 | – | 71.04 | 44.8 | 50.44 / 14.58 |
| control_seed42 | – | 0.0% | 0.0% | 0.0% | 71.50 → 72.79 | 71.04 → 72.48 | 66.47 → 71.10 | 7 | 72.13 | 44.8 | 50.44 / 15.38 |
| pruned10_seed42 | 10% | 9.8% | 19.1% | 15.5% | 36.25 → 71.17 | 37.96 → 70.34 | 43.93 → 75.72 | 10 | 70.34 | 36.3 | 47.61 / 14.09 |
| pruned20_seed42 | 20% | 19.4% | 35.9% | 29.3% | 19.69 → 68.67 | 16.74 → 67.25 | 6.94 → 72.25 | 10 | 67.25 | 28.8 | 42.44 / 13.21 |
| pruned30_seed42 | 30% | 29.0% | 50.7% | 42.0% | 14.04 → 67.03 | 11.44 → 63.25 | 0.00 → 73.41 | 10 | 63.25 | 22.2 | 36.30 / 11.57 |
| pruned40_seed42 | 40% | 38.6% | 63.6% | 53.1% | 14.29 → 66.80 | 11.41 → 60.80 | 0.00 → 61.85 | 7 | 60.39 | 16.4 | 31.56 / 10.39 |

FP32: no fine-tuning, so before = after. Performance: prune.py (validation). Size and latency: evaluate.py (CPU, fused + TorchScript; '–' = not evaluated yet).
