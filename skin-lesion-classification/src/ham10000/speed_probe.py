"""Quick batch-1 latency of one model in a fresh process, for session-drift checks.

run_evaluations.py calls this before and after the three notebook runs; if the
same model got noticeably faster or slower over the session, the background load
changed and timings from different runs are not comparable.

  python src/ham10000/speed_probe.py --model PATH --config CFG --threads 4

Prints one JSON line: median and p95 latency (ms) over --runs runs after --warmup runs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ham10000.quantize import set_engine  # noqa: E402
from ham10000.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    set_engine()
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    try:
        model = torch.jit.load(args.model, map_location="cpu").eval()
    except RuntimeError:
        model = models.resnet18(weights=None, num_classes=config.num_classes)
        model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
        model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, config.image_size, config.image_size)
    times = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({"model": args.model, "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                      "runs": args.runs}))


if __name__ == "__main__":
    main()
