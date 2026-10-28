"""
Run evaluate_model.ipynb for FP32, INT8-PTQ and INT8-QAT back to back
======================================================================
Only MODEL_PATH and MODEL_LABEL in the notebook's config cell change between runs.
Each executed copy is saved next to its results; evaluate_model.ipynb keeps the FP32 run.
Keep the laptop on mains power so all timings come from comparable conditions.

Usage (from skin-lesion-classification/):
  python run_evaluations.py
"""
import re
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

RUNS = [
    ("FP32", "results/strong_cosine25_resnet18/resnet18_baseline.pth"),
    ("INT8-PTQ", "results/ptq_int8/resnet18_ptq_int8.pt"),
    ("INT8-QAT", "results/qat_int8/resnet18_qat_int8.pt"),
]


def main():
    for label, path in RUNS:
        nb = nbformat.read("evaluate_model.ipynb", as_version=4)
        cfg = next(c for c in nb.cells if c.cell_type == "code" and "MODEL_LABEL =" in c.source)
        src = re.sub(r'^MODEL_PATH = .*$', f'MODEL_PATH = "{path}"', cfg.source, flags=re.M)
        src = re.sub(r'^MODEL_LABEL = "[^"]*"', f'MODEL_LABEL = "{label}"', src, flags=re.M)
        assert "NUM_THREADS = 4" in src and f'"{label}"' in src and path in src
        cfg.source = src
        t0 = time.time()
        ExecutePreprocessor(timeout=3600, kernel_name="skin-lesion-venv").preprocess(nb, {"metadata": {"path": "."}})
        out = f"results/eval_{label}/evaluate_model_{label}.ipynb"
        nbformat.write(nb, out)
        if label == "FP32":
            nbformat.write(nb, "evaluate_model.ipynb")  # the master notebook keeps the FP32 run's outputs
        print(f"{label}: executed in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
