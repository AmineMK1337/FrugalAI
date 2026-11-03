"""Measure the inference memory of one model in a fresh process.

Run as a subprocess (evaluate_model.ipynb does this), so the numbers are not
mixed with whatever the calling process loaded or allocated earlier:

  python src/ham10000/memory_probe.py --model PATH --config CFG --batch-size 32 --threads 4

Prints one JSON line (MB):
  baseline_mb       resident memory after imports, before the model is loaded
  after_load_mb     resident memory once the model is loaded
  model_mb          after_load_mb - baseline_mb
  inference_peak_mb highest resident memory sampled (1 ms) during inference
  activations_mb    inference_peak_mb - after_load_mb
  process_peak_mb   the OS peak resident counter for the whole process (monotonic)
  peak_over_baseline_mb  process_peak_mb - baseline_mb
"""
import argparse
import json
import os
import sys
import threading
import time

import psutil
import torch
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ham10000.quantize import set_engine  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

MB = 2**20


def process_peak_bytes(proc):
    info = proc.memory_info()
    if hasattr(info, "peak_wset"):  # Windows: peak working set
        return info.peak_wset
    import resource  # Linux / macOS: ru_maxrss is KB on Linux, bytes on macOS
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def load(path, config):
    try:
        return torch.jit.load(path, map_location="cpu").eval()
    except RuntimeError:
        # Same architecture as build_model, without downloading ImageNet weights that would be overwritten anyway
        model = models.resnet18(weights=None, num_classes=config.num_classes)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    set_engine()
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    proc = psutil.Process()
    torch.manual_seed(0)
    x = torch.randn(args.batch_size, 3, config.image_size, config.image_size)  # memory does not depend on pixel values
    baseline = proc.memory_info().rss

    model = load(args.model, config)
    after_load = proc.memory_info().rss

    peak = {"rss": after_load}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            time.sleep(0.001)

    with torch.inference_mode():
        model(x)  # warm-up: first call allocates the operator caches
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        for _ in range(args.runs):
            model(x)
        stop.set()
        sampler.join()

    result = {
        "batch_size": args.batch_size,
        "baseline_mb": baseline / MB,
        "after_load_mb": after_load / MB,
        "model_mb": (after_load - baseline) / MB,
        "inference_peak_mb": peak["rss"] / MB,
        "activations_mb": (peak["rss"] - after_load) / MB,
        "process_peak_mb": process_peak_bytes(proc) / MB,
        "peak_over_baseline_mb": (process_peak_bytes(proc) - baseline) / MB,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
