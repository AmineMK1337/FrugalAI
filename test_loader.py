"""Quick sanity-check: can we load the config and build dataloaders?"""
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent                   # .../FrugalAI
PROJECT = REPO / "resnet18_finetunning"

os.chdir(PROJECT)                                         # same as PTQ / QAT scripts
sys.path.insert(0, str(PROJECT / "src"))

from ham10000.utils import load_config
from ham10000.data import build_dataloaders

cfg = load_config("configs/strong_cosine25.yaml")
train_loader, val_loader, class_to_idx = build_dataloaders(cfg)

print(f"OK - train batches: {len(train_loader)}, val batches: {len(val_loader)}")
print(f"Classes: {class_to_idx}")
