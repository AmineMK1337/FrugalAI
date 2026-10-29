"""
Run evaluate_model.ipynb for FP32, INT8-PTQ and INT8-QAT back to back
======================================================================
Only MODEL_PATH and MODEL_LABEL in the notebook's config cell change between runs.
Each executed copy is saved next to its results; evaluate_model.ipynb keeps the FP32 run.
Keep the laptop on mains power so all timings come from comparable conditions.

Session-drift check: a quick fresh-process latency probe of FP32 and INT8-PTQ runs
before and after the three notebooks. If the same model's latency moved by more
than DRIFT_TOLERANCE, background load changed during the session and the
runs' timings are not directly comparable. Written to results/eval_session_drift.json
and checked by compare_models.ipynb.

Usage (from skin-lesion-classification/):
  python run_evaluations.py
"""
import json
import re
import subprocess
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

CONFIG_PATH = "configs/strong_cosine25.yaml"
NUM_THREADS = 4
RUNS = [
    ("FP32", "results/strong_cosine25_resnet18/resnet18_baseline.pth"),
    ("INT8-PTQ", "results/ptq_int8/resnet18_ptq_int8.pt"),
    ("INT8-QAT", "results/qat_int8/resnet18_qat_int8.pt"),
]
DRIFT_MODELS = RUNS[:2]
DRIFT_TOLERANCE = 0.10


def speed_probe(path):
    out = subprocess.run([sys.executable, "src/ham10000/speed_probe.py", "--model", path, "--config", CONFIG_PATH,
                          "--threads", str(NUM_THREADS)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    before = {label: speed_probe(path) for label, path in DRIFT_MODELS}
    print("drift probe before:", {k: round(v["median_ms"], 2) for k, v in before.items()}, flush=True)

    for label, path in RUNS:
        nb = nbformat.read("evaluate_model.ipynb", as_version=4)
        cfg = next(c for c in nb.cells if c.cell_type == "code" and "MODEL_LABEL =" in c.source)
        src = re.sub(r'^MODEL_PATH = .*$', f'MODEL_PATH = "{path}"', cfg.source, flags=re.M)
        src = re.sub(r'^MODEL_LABEL = "[^"]*"', f'MODEL_LABEL = "{label}"', src, flags=re.M)
        assert f"NUM_THREADS = {NUM_THREADS}" in src and f'"{label}"' in src and path in src
        cfg.source = src
        t0 = time.time()
        ExecutePreprocessor(timeout=3600, kernel_name="skin-lesion-venv").preprocess(nb, {"metadata": {"path": "."}})
        out = f"results/eval_{label}/evaluate_model_{label}.ipynb"
        nbformat.write(nb, out)
        if label == "FP32":
            nbformat.write(nb, "evaluate_model.ipynb")  # the master notebook keeps the FP32 run's outputs
        print(f"{label}: executed in {time.time() - t0:.0f} s -> {out}", flush=True)

    after = {label: speed_probe(path) for label, path in DRIFT_MODELS}
    print("drift probe after: ", {k: round(v["median_ms"], 2) for k, v in after.items()}, flush=True)
    drift = {label: after[label]["median_ms"] / before[label]["median_ms"] - 1 for label, _ in DRIFT_MODELS}
    report = {"tolerance": DRIFT_TOLERANCE, "before": before, "after": after, "relative_change": drift,
              "stable": all(abs(d) <= DRIFT_TOLERANCE for d in drift.values())}
    json.dump(report, open("results/eval_session_drift.json", "w"), indent=2)
    print("session drift:", {k: f"{v:+.1%}" for k, v in drift.items()}, "->", "stable" if report["stable"] else "DRIFTED")


if __name__ == "__main__":
    main()
