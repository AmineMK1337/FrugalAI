"""Calibration-size sweep for static INT8 PTQ, evaluated on the VALIDATION split.

Calibrates the same model (ptq_static.quantize: x86 backend, per-channel INT8 weights,
reduce_range activations) on stratified samples of TRAINING images of growing size, three
random samples per size (seeds 42, 1, 2; one run when all training images are used), and
compares each INT8 model with FP32. The test split is never loaded; nothing is saved but the tables.

Writes next to this script: calibration_sweep_runs.csv (one row per run) and
calibration_sweep_summary.csv (mean, min, max per size).

Usage: python quantization/ptq/static/benchmarking/calibration_sweep.py
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ptq_static as p
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
torch.manual_seed(0)
cfg = p.load_config(p.CONFIG)
ds = p.Ham10000Dataset(cfg.metadata_csv, cfg.data_dir, transform=p.eval_transform(cfg))
train_idx, _, _ = p.lesion_split(ds.data, cfg)
Xtr = torch.cat([x for x, _ in DataLoader(Subset(ds, train_idx), batch_size=64, num_workers=8)])
dx = ds.data.loc[train_idx, "dx"].to_numpy()
_, val_loader, c2i = p.build_dataloaders(cfg)
Xv, yv = map(torch.cat, zip(*val_loader)); yv = yv.numpy()
fp32 = p.load_fp32(7)
with torch.inference_mode():
    ref = torch.cat([fp32(b) for b in Xv.split(64)])
refp = ref.softmax(1); refpred = ref.argmax(1).numpy()
print("train images", len(Xtr), "val", len(yv), flush=True)

class Loader:  # p.quantize iterates (images, labels) batches
    def __init__(self, X): self.X = X
    def __iter__(self): return ((b, None) for b in self.X.split(32))

rows = []
for n in (100, 250, 500, 1000, 2000, 4000, len(Xtr)):
    for seed in ((42,) if n == len(Xtr) else (42, 1, 2)):
        pos = np.arange(len(Xtr)) if n == len(Xtr) else train_test_split(np.arange(len(Xtr)), train_size=n, stratify=dx, random_state=seed)[0]
        t = time.time()
        q, _ = p.quantize(fp32, Loader(Xtr[torch.as_tensor(np.sort(pos))]))
        with torch.inference_mode():
            lg = torch.cat([q(b) for b in Xv.split(64)])
        pr = lg.argmax(1).numpy(); prob = lg.softmax(1)
        rows.append({"n_calibration": n, "seed": seed, "accuracy": (pr == yv).mean() * 100,
                     "f1_macro": f1_score(yv, pr, average="macro") * 100,
                     "mel_recall": recall_score(yv, pr, labels=[c2i["mel"]], average=None)[0] * 100,
                     "identical_to_fp32": (pr == refpred).mean() * 100,
                     "softmax_mad_x1000": (prob - refp).abs().mean().item() * 1000,
                     "logit_mse": ((lg - ref) ** 2).mean().item(), "calib_s": time.time() - t})
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(HERE / "calibration_sweep_runs.csv", index=False)
print("FP32 accuracy", (refpred == yv).mean() * 100)
summary = df.drop(columns="seed").groupby("n_calibration").agg(["mean", "min", "max"])
summary.columns = ["_".join(c) for c in summary.columns]
summary.to_csv(HERE / "calibration_sweep_summary.csv")
print(summary.round(3).T.to_string())
