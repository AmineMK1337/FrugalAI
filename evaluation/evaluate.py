"""Evaluate every model in the same format, on the same data, inputs and hardware (CPU), and save the results
by model and metric type, each CSV next to its plots. The only difference between models is the precision.

Format:      every model runs as fused (Conv+BatchNorm+ReLU) TorchScript. FP32 and weight-only models are loaded
             into torchvision.models.quantization.resnet18 (quantize=False), fused with fuse_model() and traced
             with torch.jit.trace; static PTQ and QAT are already fused INT8 TorchScript. Before anything is measured, each
             fused model is checked against its unfused version (same predictions, logits equal up to rounding).
Performance: accuracy, precision, recall, specificity, F1, AUC (macro averages, and per class), each with a
             95% bootstrap CI and a paired bootstrap against FP32 (same resamples for both models).
Efficiency:  size (original file), FLOPs (ptflops MACs x 2, fused FP32 architecture), latency (batch 1, 30
             warm-up + 200 timed runs, median and p95), throughput (batch 32), peak RAM (psutil), at 1 (main) and
             4 threads. One shared function measures every model, in a fresh process per run, 3 repeats, models
             back to back in one session.
Environment: energy and CO2e per 1,000 images at 1 thread, ESTIMATED by CodeCarbon over at least 60 s per run.
             This VM has no RAPL counter, so CodeCarbon assumes a CPU power: compare models, do not quote as measured.

Nothing is trained, calibrated or modified; model files, split and config are hashed before and after the run.

Structured pruning (pruning/structured/filter/): pruned and FP32-control models are whole saved modules
(pruning/structured/filter/models/<run>_selected.pt), fused with fuse_model() and traced like FP32. FLOPs and parameters are
counted on each model's own architecture. They are evaluated in a separate session and folder, with FP32 re-measured
there as the timing reference:
  python evaluation/evaluate.py --models fp32 pruned20_seed42 ... --results-dir pruning/structured/filter/results/evaluation

Unstructured pruning (pruning/unstructured/): sparse<pct>_seed<s> are whole saved modules with exact zeros
(pruning/unstructured/models/<run>_selected.pt), fused and traced the same way; dense CPU kernels still multiply the
zeros. Storage columns for every model with FP32 weights: zeros and sparsity of the Conv2d/Linear weights, non-zero
parameters, and the torch.save(state_dict) size dense and gzip-compressed (level 9):
  python evaluation/evaluate.py --models fp32 sparse50_seed42 ... --results-dir pruning/unstructured/results/evaluation

Whole-block pruning (pruning/structured/whole_block/): blocks<k>_seed<s> are whole saved modules in which k
identity-shortcut BasicBlocks are nn.Identity (pruning/structured/whole_block/models/<run>_selected.pt), fused with
fuse_model() and traced like the other pruned models. Any Conv2d-BatchNorm2d(-ReLU) nn.Sequential that fuse_model()
does not reach (residual-branch pruning will keep a downsample shortcut as one) is fused too, so every model is
measured fully fused:
  python evaluation/evaluate.py --models fp32 control_seed42 blocks1_seed42 ... \
      --results-dir pruning/structured/whole_block/results/evaluation

Usage: python evaluation/evaluate.py [--split validation|test] [--repeats 3] [--country TUN] [--models ...]
       [--results-dir DIR]
Writes evaluation/results/<model>/<metric_type>/, evaluation/results/comparison/<metric_type>/ and
evaluation/results/run_info.json, then draws the plots (evaluation/plot_results.py).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # same hardware for every model: CPU

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # noqa: E402
from torchvision.models import resnet18  # noqa: E402
from torchvision.models.quantization import resnet18 as quantizable_resnet18  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT / "resnet18_finetunning"
WEIGHT_ONLY = ROOT / "quantization" / "ptq" / "weight_only"
QAT = ROOT / "quantization" / "qat"
PRUNING = ROOT / "pruning" / "structured" / "filter"
UNSTRUCTURED = ROOT / "pruning" / "unstructured"
WHOLE_BLOCK = ROOT / "pruning" / "structured" / "whole_block"
RESULTS = HERE / "results"
sys.path[:0] = [str(PROJECT / "src"), str(WEIGHT_ONLY)]
LAUNCH_DIR = Path.cwd()  # relative --results-dir is taken from where the script is launched
os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project

import quantize as q  # noqa: E402
from ham10000.data import build_dataloaders, build_test_loader  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = PROJECT / "results/strong_cosine25_resnet18/resnet18_baseline.pth"
PER_CHANNEL = WEIGHT_ONLY / "results/resnet18_w8_perchannel.pt"
PER_TENSOR = WEIGHT_ONLY / "results/resnet18_w8_pertensor.pt"
QAT_INT8 = QAT / "resnet18_qat_int8.pt"
STATIC = ROOT / "quantization" / "ptq" / "static"
STATIC_INT8 = STATIC / "resnet18_static_int8.pt"
NUM_CLASSES = 7
FORMAT = "fused + TorchScript"
BATCH_SIZE = 32
WARMUP, LATENCY_RUNS = 30, 200
THREADS = (1, 4)  # 1 = main protocol
ENERGY_THREADS, ENERGY_MIN_SECONDS = 1, 60
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0
ENGINE = "x86"
PERFORMANCE = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro",
               "auc_macro", "melanoma_recall")  # balanced accuracy = mean per-class recall (= recall_macro)
MELANOMA, MELANOMA_NAME = 4, "mel"  # class index (classes sorted alphabetically); checked in main()
LOGIT_TOLERANCE = 1e-3  # fused vs unfused: float rounding only gives differences far below this


# ── Models ───────────────────────────────────────────────────────────────────

def fp32_state():
    return torch.load(CHECKPOINT, map_location="cpu", weights_only=True)


def weight_only_state(path):
    return lambda: q.dequantize_weights(q.load_compressed(path))


def fused_torchscript(state):
    """FP32 weights in the quantizable ResNet-18 (quantize=False), BatchNorm and ReLU fused, traced."""
    model = quantizable_resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
    model.load_state_dict(state, strict=True)
    model.eval().fuse_model()
    with torch.no_grad():
        return torch.jit.trace(model, torch.zeros(1, 3, 224, 224))


def unfused_fp32():
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(fp32_state(), strict=True)
    return model.eval()


MODELS = {  # name: weights (None = already TorchScript), unfused reference, file on disk, note
    "fp32": dict(state=fp32_state, unfused=unfused_fp32, file=CHECKPOINT, note="FP32"),
    "weight_only_per_channel": dict(
        state=weight_only_state(PER_CHANNEL), unfused=lambda: q.load_dequantized_resnet18(PER_CHANNEL, NUM_CLASSES),
        file=PER_CHANNEL, note="INT8 weights on disk, dequantized to FP32 at load: computes in FP32"),
    "weight_only_per_tensor": dict(
        state=weight_only_state(PER_TENSOR), unfused=lambda: q.load_dequantized_resnet18(PER_TENSOR, NUM_CLASSES),
        file=PER_TENSOR, note="INT8 weights on disk, dequantized to FP32 at load: computes in FP32"),
    "ptq_static": dict(state=None, unfused=None, file=STATIC_INT8,
                       note="INT8 weights and activations (x86 engine), static PTQ, calibrated on 1,000 train images "
                            "(seed 42)"),
    "qat": dict(state=None, unfused=None, file=QAT_INT8,
                note="INT8 weights and activations (x86 engine), QAT, final model (epoch 10)"),
}
DEFAULT_MODELS = list(MODELS)


def saved_module(path):
    """A whole model saved by pruning/structured/filter/prune.py (own file: weights_only=False), eval mode."""
    return lambda: torch.load(path, map_location="cpu", weights_only=False).eval()


def fuse_sequentials(module):
    """Fuse, in place, every nn.Sequential that is exactly Conv2d-BatchNorm2d or Conv2d-BatchNorm2d-ReLU and that
    fuse_model() leaves unfused (it only reaches QuantizableBasicBlock modules). No-op for the other models."""
    for name, child in list(module.named_modules()):
        if isinstance(child, torch.nn.Sequential) and len(child) in (2, 3) and \
                isinstance(child[0], torch.nn.Conv2d) and isinstance(child[1], torch.nn.BatchNorm2d) and \
                (len(child) == 2 or isinstance(child[2], torch.nn.ReLU)):
            torch.ao.quantization.fuse_modules(child, [str(i) for i in range(len(child))], inplace=True)
    return module


def fused_module_torchscript(module):
    module.eval().fuse_model()
    fuse_sequentials(module)
    with torch.no_grad():
        return torch.jit.trace(module, torch.zeros(1, 3, 224, 224))


for _path in sorted((PRUNING / "models").glob("*_selected.pt")):  # pruned<ratio>_seed<s>, control_seed<s>
    _name = _path.name[:-len("_selected.pt")]
    MODELS[_name] = dict(
        state=None, module=saved_module(_path), unfused=saved_module(_path), file=_path,
        note=("FP32 control: unpruned, fine-tuned with the pruning schedule" if _name.startswith("control")
              else f"FP32, structured pruning (target {_name[6:].split('_')[0]}% channels)")
        + ", best validation macro F1 epoch")
for _path in sorted((UNSTRUCTURED / "models").glob("sparse*_selected.pt")):  # sparse<pct>_seed<s>
    _name = _path.name[:-len("_selected.pt")]
    MODELS[_name] = dict(
        state=None, module=saved_module(_path), unfused=saved_module(_path), file=_path,
        note=f"FP32, unstructured pruning (target {_name[6:].split('_')[0]}% of conv/linear weights set to zero), "
             "best validation macro F1 epoch")
for _path in sorted((WHOLE_BLOCK / "models").glob("blocks*_selected.pt")):  # blocks<k>_seed<s>[_rawstart]
    _name = _path.name[:-len("_selected.pt")]
    MODELS[_name] = dict(
        state=None, module=saved_module(_path), unfused=saved_module(_path), file=_path,
        note=f"FP32, whole-block pruning ({_name[6:].split('_')[0]} identity-shortcut BasicBlock(s) removed), "
             "best validation macro F1 epoch")

def export_torchscript(workdir):
    """Path of each model's fused TorchScript file. Static PTQ and QAT already are one; the others are traced here."""
    paths = {}
    for name, spec in MODELS.items():
        if "module" in spec:
            paths[name] = Path(workdir) / f"{name}.pt"
            torch.jit.save(fused_module_torchscript(spec["module"]()), paths[name])
        elif spec["state"] is None:
            paths[name] = spec["file"]
        else:
            paths[name] = Path(workdir) / f"{name}.pt"
            torch.jit.save(fused_torchscript(spec["state"]()), paths[name])
    return paths


def load_torchscript(path):
    torch.backends.quantized.engine = ENGINE
    return torch.jit.load(str(path), map_location="cpu").eval()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def logits(model, images):
    return torch.cat([model(b) for b in images.split(BATCH_SIZE)])


def check_fused(paths, images):
    """Each fused FP32 / weight-only model must give the predictions of its unfused version."""
    rows = []
    for name, spec in MODELS.items():
        if spec["unfused"] is None:
            continue
        before, after = logits(spec["unfused"](), images), logits(load_torchscript(paths[name]), images)
        rows.append({"model": name, "n_images": len(images),
                     "identical_predictions_pct": float((before.argmax(1) == after.argmax(1)).float().mean() * 100),
                     "max_abs_logit_difference": float((before - after).abs().max()),
                     "max_abs_probability_difference": float((before.softmax(1) - after.softmax(1)).abs().max())})
        print(f"fused check, {name}: {rows[-1]['identical_predictions_pct']:.2f}% identical predictions, "
              f"max logit difference {rows[-1]['max_abs_logit_difference']:.2e}", flush=True)
    table = pd.DataFrame(rows)
    bad = table[(table["identical_predictions_pct"] < 100) | (table["max_abs_logit_difference"] > LOGIT_TOLERANCE)]
    if len(bad):
        print(table.to_string(index=False))
        sys.exit(f"CHECK FAILED: fused models differ from their unfused versions by more than rounding: "
                 f"{list(bad['model'])}")
    return table


# ── Performance ──────────────────────────────────────────────────────────────

def scores(y, probs, n_classes):
    """Overall metrics (macro averages) and per-class arrays, from labels and class probabilities."""
    pred = probs.argmax(1)
    cm = np.bincount(y * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(0), cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        specificity = (cm.sum() - actual - predicted + tp) / (cm.sum() - actual)
    auc = np.array([roc_auc_score(y == i, probs[:, i]) if 0 < actual[i] < len(y) else np.nan
                    for i in range(n_classes)])
    overall = {"accuracy": tp.sum() / len(y), "precision_macro": precision.mean(),
               "recall_macro": np.nanmean(recall), "specificity_macro": specificity.mean(),
               "f1_macro": f1.mean(), "auc_macro": np.nanmean(auc), "balanced_accuracy": np.nanmean(recall),
               "melanoma_recall": recall[MELANOMA]}
    per_class = {"support": actual, "precision": precision, "recall": recall, "specificity": specificity,
                 "f1": f1, "auc": auc}
    return overall, per_class


def check_scores(y, probs, n_classes):
    """The hand-written metrics must agree with scikit-learn on the full split."""
    overall, _ = scores(y, probs, n_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y, probs.argmax(1), labels=list(range(n_classes)),
                                                               zero_division=0)
    for key, value in (("precision_macro", precision.mean()), ("recall_macro", recall.mean()),
                       ("f1_macro", f1.mean()),
                       ("auc_macro", roc_auc_score(y, probs, multi_class="ovr", average="macro"))):
        if abs(overall[key] - value) > 1e-9:
            sys.exit(f"CHECK FAILED: {key} {overall[key]} differs from scikit-learn {value}")


def bootstrap(probs_by_model, y, n_classes):
    """Same resampled index sets for every model, so differences vs FP32 are paired."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(0, len(y), size=(BOOTSTRAP, len(y)))
    stats = {}
    for name, probs in probs_by_model.items():
        rows = [scores(y[i], probs[i], n_classes)[0] for i in resamples]
        stats[name] = {metric: np.array([r[metric] for r in rows]) for metric in PERFORMANCE}
        print(f"bootstrap done: {name}", flush=True)
    return stats


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def check_against_earlier_results(overall):
    """Fusing changes only rounding, so performance must equal the results saved when the models were made
    (exactly for prediction-based metrics; AUC, which uses probabilities, up to 1e-4)."""
    tolerance = {m: 1e-9 for m in PERFORMANCE} | {"auc_macro": 1e-4}
    for name in (n for n in MODELS if n.startswith(("pruned", "control", "sparse", "blocks"))):
        # saved by prune.py from GPU logits: a prediction near a tie may flip on the CPU, so allow 2 images
        source = {"sparse": UNSTRUCTURED, "blocks": WHOLE_BLOCK}.get(name[:6], PRUNING)
        saved = pd.read_csv(source / "results" / name / "metrics.csv")
        saved = saved[saved["stage"].str.startswith("selected")].iloc[0]
        diff = abs(overall[name]["accuracy"] - saved["accuracy"]) * overall[name]["n_images"]
        print(f"{name}: accuracy differs from the prune.py validation result by {diff:.0f} image(s)", flush=True)
        if diff > 2:
            sys.exit(f"CHECK FAILED: {name} accuracy differs from {source}/results by {diff:.0f} images")
    if "fp32" not in MODELS:
        return
    weight_only = pd.read_csv(WEIGHT_ONLY / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("weight_only_per_channel", "per_channel"),
                      ("weight_only_per_tensor", "per_tensor")):
        if name not in MODELS:
            continue
        for metric, old_metric in (("accuracy", "accuracy"), ("f1_macro", "macro_f1")):
            if abs(overall[name][metric] - weight_only.loc[old, old_metric]) > tolerance[metric]:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {WEIGHT_ONLY}/results")
    qat = pd.read_csv(QAT / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("qat", "QAT")):
        if name not in MODELS:
            continue
        for metric in (m for m in PERFORMANCE if m in qat):
            if abs(overall[name][metric] - qat.loc[old, metric]) > tolerance[metric]:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {QAT}/results")
    static = pd.read_csv(STATIC / "results/validation_results.csv").set_index("model")
    if "ptq_static" in MODELS:  # the saved file is the seed-42 calibration
        for metric in (m for m in PERFORMANCE if m in static):
            if abs(overall["ptq_static"][metric] - static.loc["INT8_seed42", metric]) > tolerance[metric]:
                sys.exit(f"CHECK FAILED: ptq_static {metric} differs from {STATIC}/results")
    print("Performance matches the results saved when the models were made", flush=True)


# ── Efficiency and environment: one shared function, one fresh process per run ─

class PeakRSS:
    """Highest resident memory of this process (psutil), sampled every millisecond while active."""

    def __enter__(self):
        self.process, self.stop = psutil.Process(), threading.Event()
        self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.001)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


@torch.inference_mode()
def measure(model_path, threads, workdir, country):
    """The same measurement for every model: batch-1 latency and peak RAM, batch-32 throughput, and energy
    (CodeCarbon, at ENERGY_THREADS only) over full passes of the cached validation tensors."""
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    sample = torch.load(Path(workdir) / "sample.pt", weights_only=True)
    before = psutil.Process().memory_info().rss
    with PeakRSS() as memory:  # load + batch-1 warm-up
        model = load_torchscript(model_path)
        for _ in range(WARMUP):
            model(sample)
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        model(sample)
        times.append((time.perf_counter() - start) * 1000)
    result = {"latency_median_ms": float(np.median(times)), "latency_p95_ms": float(np.percentile(times, 95)),
              "peak_memory_mb": (memory.peak - before) / 2 ** 20, "latency_samples_ms": times}

    images = torch.load(Path(workdir) / "images.pt", weights_only=True)
    batches = [images[i:i + BATCH_SIZE].clone() for i in range(0, len(images), BATCH_SIZE)]
    for batch in (batches[0], batches[-1]) * 3:  # TorchScript specializes on the first calls of each shape
        model(batch)
    start = time.perf_counter()
    for batch in batches:
        model(batch)
    result["throughput_img_s"] = len(images) / (time.perf_counter() - start)

    if threads == ENERGY_THREADS:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code=country, measure_power_secs=1, tracking_mode="process",
                                          save_to_file=False, log_level="critical", allow_multiple_runs=True)
        tracker.start()
        start, n = time.perf_counter(), 0
        while time.perf_counter() - start < ENERGY_MIN_SECONDS:  # whole passes until at least 60 s
            for batch in batches:
                model(batch)
            n += len(images)
        seconds = time.perf_counter() - start
        tracker.stop()
        data = tracker.final_emissions_data
        result.update({"energy_seconds": seconds, "energy_images": n,
                       "energy_wh_per_1000": data.energy_consumed * 1000 * 1000 / n,
                       "co2e_g_per_1000": data.emissions * 1000 * 1000 / n,
                       "estimated_power_w": data.cpu_power + data.ram_power,
                       "grid_g_per_kwh": data.emissions / data.energy_consumed * 1000})
    return result


def worker(args):
    Path(args.out).write_text(json.dumps(measure(args.model_path, args.threads, args.workdir, args.country)))


def flops_and_params(name):
    """GFLOPs (ptflops MACs x 2) on the fused architecture, and parameters on the unfused one. Pruned and control
    models: their own architecture. Every other model has the FP32 architecture (ptflops cannot count TorchScript
    or INT8 ops)."""
    from ptflops import get_model_complexity_info
    spec = MODELS[name]
    if "module" in spec:
        model = spec["module"]()
    else:
        model = quantizable_resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
        model.load_state_dict(fp32_state(), strict=True)
    params = sum(p.numel() for p in model.parameters())
    model.eval().fuse_model()
    fuse_sequentials(model)
    macs, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False,
                                        verbose=False)
    return 2 * macs / 1e9, params


def storage(name):
    """Zeros and sparsity of the Conv2d/Linear weights, non-zero parameters, and the torch.save(state_dict) size,
    dense and gzip-compressed (level 9). Models with FP32 weights only (QAT: not applicable, NaN)."""
    import gzip
    import io
    spec = MODELS[name]
    if "module" in spec:
        model = spec["module"]()
    elif spec["state"] is not None:
        model = quantizable_resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
        model.load_state_dict(spec["state"](), strict=True)
    else:
        return {k: float("nan") for k in ("weights_prunable", "weights_zero", "sparsity_pct", "params_nonzero",
                                          "state_dict_dense_mb", "state_dict_gzip_mb")}
    weights = [m.weight for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    zeros = sum(int((w == 0).sum()) for w in weights)
    total = sum(w.numel() for w in weights)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    raw = buffer.getvalue()
    return {"weights_prunable": total, "weights_zero": zeros, "sparsity_pct": zeros / total * 100,
            "params_nonzero": sum(int((p != 0).sum()) for p in model.parameters()),
            "state_dict_dense_mb": len(raw) / 1e6, "state_dict_gzip_mb": len(gzip.compress(raw, 9)) / 1e6}


# ── Main ─────────────────────────────────────────────────────────────────────

def save_csv(table, folder, name, split):
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / f"{name}_{split}.csv", index=False)


def main(args):
    global MODELS, RESULTS
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; available: {list(MODELS)}")
    MODELS = {m: MODELS[m] for m in args.models}
    RESULTS = (LAUNCH_DIR / args.results_dir).resolve() if args.results_dir else RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    split = args.split
    config = load_config(CONFIG)
    tracked = {**{name: spec["file"] for name, spec in MODELS.items()},
               "split_csv": PROJECT / config.splits_csv, "config": PROJECT / CONFIG}
    hashes = {name: sha256(path) for name, path in tracked.items()}
    if split == "test":
        loader, _, class_to_idx = build_test_loader(config, num_workers=config.num_workers)
    else:
        _, loader, class_to_idx = build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    if classes[MELANOMA] != MELANOMA_NAME:
        sys.exit(f"CHECK FAILED: class {MELANOMA} is {classes[MELANOMA]}, not {MELANOMA_NAME}")
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    print(f"{split}: {len(y)} images, classes {classes}", flush=True)

    with tempfile.TemporaryDirectory(dir=RESULTS) as workdir:
        paths = export_torchscript(workdir)
        fused_check = check_fused(paths, images)
        if args.check_only:
            print(fused_check.to_string(index=False))
            return
        for folder in [RESULTS / m for m in MODELS] + [RESULTS / "comparison"]:
            shutil.rmtree(folder, ignore_errors=True)  # outputs of an earlier run of this script
        save_csv(fused_check, RESULTS / "comparison" / "performance", "fused_vs_unfused_check", split)

        # Performance, from the same TorchScript files that are timed below
        probs = {name: logits(load_torchscript(paths[name]), images).softmax(1).numpy() for name in MODELS}
        for name in MODELS:
            check_scores(y, probs[name], len(classes))
        boot = bootstrap(probs, y, len(classes))
        overall = {}
        for name in MODELS:
            point, per_class = scores(y, probs[name], len(classes))
            metrics = {"model": name, "format": FORMAT}
            for metric in PERFORMANCE:
                metrics[metric] = float(point[metric])
                metrics[f"{metric}_ci_low"], metrics[f"{metric}_ci_high"] = ci(boot[name][metric])
            for metric in PERFORMANCE if name != "fp32" else ():
                diff = boot[name][metric] - boot["fp32"][metric]
                metrics[f"{metric}_change_vs_fp32"] = metrics[metric] - overall["fp32"][metric]
                metrics[f"{metric}_change_ci_low"], metrics[f"{metric}_change_ci_high"] = ci(diff)
                metrics[f"{metric}_paired_bootstrap_p"] = float(min(1.0, 2 * min((diff <= 0).mean(),
                                                                                 (diff >= 0).mean())))
            metrics["n_images"] = len(y)
            metrics["note"] = MODELS[name]["note"]
            overall[name] = metrics
            folder = RESULTS / name / "performance"
            save_csv(pd.DataFrame([metrics]), folder, "metrics", split)
            save_csv(pd.DataFrame({"class": classes, **per_class}), folder, "per_class", split)
        if split == "validation":
            check_against_earlier_results(overall)

        # Efficiency and environment: every model and thread count, back to back, in this session
        rows, samples = [], []
        torch.save(images, Path(workdir) / "images.pt")
        torch.save(images[:1].clone(), Path(workdir) / "sample.pt")
        for repeat in range(args.repeats):
            for threads in THREADS:
                for name in MODELS:
                    result = Path(workdir) / "result.json"
                    subprocess.run([sys.executable, __file__, "--worker", "--model-path", str(paths[name]),
                                    "--threads", str(threads), "--workdir", workdir, "--out", str(result),
                                    "--country", args.country], check=True, stderr=subprocess.DEVNULL)
                    run = json.loads(result.read_text())
                    samples += [{"model": name, "threads": threads, "repeat": repeat, "run": i, "latency_ms": t}
                                for i, t in enumerate(run.pop("latency_samples_ms"))]
                    rows.append({"model": name, "format": FORMAT, "threads": threads, "repeat": repeat, **run})
                    print(f"repeat {repeat + 1}/{args.repeats}, {threads} thread(s), {name}: "
                          f"{run['latency_median_ms']:.2f} ms, {run['throughput_img_s']:.1f} img/s"
                          + (f", {run['energy_wh_per_1000']:.3f} Wh/1000 (estimate)" if "energy_wh_per_1000" in run
                             else ""), flush=True)
    runs = pd.DataFrame(rows)
    samples = pd.DataFrame(samples)
    medians = runs.drop(columns=["repeat", "format"]).groupby(["model", "threads"]).median()
    energy_columns = ["energy_seconds", "energy_images", "energy_wh_per_1000", "co2e_g_per_1000",
                      "estimated_power_w", "grid_g_per_kwh"]

    for name in MODELS:
        gflops, params = flops_and_params(name)
        efficiency = {"model": name, "format": FORMAT, "size_mb": MODELS[name]["file"].stat().st_size / 1e6,
                      "gflops": gflops, "params": params, **storage(name)}
        for t in THREADS:
            m = medians.loc[(name, t)]
            efficiency.update({f"latency_median_ms_{t}t": m["latency_median_ms"],
                               f"latency_p95_ms_{t}t": m["latency_p95_ms"],
                               f"throughput_img_s_{t}t": m["throughput_img_s"],
                               f"peak_memory_mb_{t}t": m["peak_memory_mb"]})
        efficiency["note"] = MODELS[name]["note"]
        m = medians.loc[(name, ENERGY_THREADS)]
        environment = {"model": name, "format": FORMAT, "threads": ENERGY_THREADS,
                       "energy_wh_per_1000": m["energy_wh_per_1000"], "co2e_g_per_1000": m["co2e_g_per_1000"],
                       "estimated_power_w": m["estimated_power_w"], "seconds_per_run": m["energy_seconds"],
                       "source": "estimate (CodeCarbon process mode; no RAPL counter on this VM)",
                       "grid_g_per_kwh": m["grid_g_per_kwh"], "country": args.country}
        mine = runs[runs["model"] == name]
        save_csv(pd.DataFrame([efficiency]), RESULTS / name / "efficiency", "metrics", split)
        save_csv(samples[samples["model"] == name], RESULTS / name / "efficiency", "latency_samples", split)
        save_csv(mine.drop(columns=energy_columns), RESULTS / name / "efficiency", "runs", split)
        save_csv(mine[mine["threads"] == ENERGY_THREADS][["model", "format", "threads", "repeat", *energy_columns]],
                 RESULTS / name / "environment", "runs", split)
        save_csv(pd.DataFrame([environment]), RESULTS / name / "environment", "metrics", split)

    write_comparison(split)
    after = {name: sha256(path) for name, path in tracked.items()}
    if after != hashes:
        sys.exit(f"CHECK FAILED: files changed during the run: {[k for k in hashes if hashes[k] != after[k]]}")
    (RESULTS / "run_info.json").write_text(json.dumps({
        "split": split, "n_images": len(y), "classes": classes, "device": "cpu",
        "format": FORMAT,
        "format_details": {
            "fp32": "weights in torchvision.models.quantization.resnet18 (quantize=False), fuse_model(), "
                    "torch.jit.trace",
            "weight_only_per_channel": "dequantized to FP32, then as fp32",
            "weight_only_per_tensor": "dequantized to FP32, then as fp32",
            "ptq_static": "saved fused INT8 TorchScript (torch.jit.script), loaded as is",
            "qat": "saved fused INT8 TorchScript (torch.jit.script), loaded as is",
            "pruned / control": "whole saved FP32 module (pruning/structured/filter/models), fuse_model(), torch.jit.trace",
            "sparse": "whole saved FP32 module with exact zeros (pruning/unstructured/models), fuse_model(), "
                      "torch.jit.trace; dense kernels",
            "blocks": "whole saved FP32 module, removed blocks are nn.Identity (pruning/structured/whole_block/models), "
                      "fuse_model() + Conv-BN(-ReLU) Sequential fusion, torch.jit.trace"},
        "models": list(MODELS),
        "fused_vs_unfused_check": fused_check.to_dict(orient="records"),
        "cpu": subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True,
                              text=True).stdout.split(":")[-1].strip(),
        "logical_cpus": os.cpu_count(), "threads": list(THREADS), "threads_main": 1, "interop_threads": 1,
        "repeats": args.repeats, "warmup_runs": WARMUP, "latency_runs": LATENCY_RUNS,
        "throughput_batch_size": BATCH_SIZE, "peak_memory": "psutil RSS, sampled every 1 ms during load + warm-up",
        "energy_threads": ENERGY_THREADS, "energy_min_seconds": ENERGY_MIN_SECONDS,
        "bootstrap_resamples": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "quantized_engine": ENGINE,
        "flops_counted_on": "fused FP32 architecture, for every model",
        "torch": torch.__version__, "python": platform.python_version(), "country_for_co2e": args.country,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashes,
    }, indent=2))
    print("Model files, split and config unchanged. Results in", RESULTS, flush=True)

    # Load evaluation/plot_results.py by path: another module on sys.path could share the name
    spec = importlib.util.spec_from_file_location("evaluation_plot_results", HERE / "plot_results.py")
    plot_results = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_results)
    plot_results.main(split, RESULTS, list(MODELS))


def write_comparison(split):
    out = RESULTS / "comparison"
    perf = pd.concat([pd.read_csv(RESULTS / m / "performance" / f"metrics_{split}.csv") for m in MODELS])
    save_csv(perf.drop(columns="note"), out / "performance", "table", split)

    eff = pd.concat([pd.read_csv(RESULTS / m / "efficiency" / f"metrics_{split}.csv") for m in MODELS])
    base = eff.set_index("model").loc["fp32"]
    eff["size_ratio_fp32_over_model"] = base["size_mb"] / eff["size_mb"]
    for t in THREADS:
        eff[f"speedup_vs_fp32_{t}t"] = base[f"latency_median_ms_{t}t"] / eff[f"latency_median_ms_{t}t"]
        eff[f"throughput_change_vs_fp32_pct_{t}t"] = (eff[f"throughput_img_s_{t}t"] / base[f"throughput_img_s_{t}t"]
                                                      - 1) * 100
        eff[f"memory_change_vs_fp32_pct_{t}t"] = (eff[f"peak_memory_mb_{t}t"] / base[f"peak_memory_mb_{t}t"] - 1) * 100
    save_csv(eff.drop(columns="note"), out / "efficiency", "table", split)

    env = pd.concat([pd.read_csv(RESULTS / m / "environment" / f"metrics_{split}.csv") for m in MODELS])
    base = env.set_index("model").loc["fp32"]
    for metric in ("energy_wh_per_1000", "co2e_g_per_1000"):
        env[f"{metric}_change_vs_fp32_pct"] = (env[metric] / base[metric] - 1) * 100
    save_csv(env, out / "environment", "table", split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--country", default="TUN", help="ISO-3 country of the machine, for CO2e only")
    parser.add_argument("--check-only", action="store_true", help="only run the fused vs unfused check")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workdir")
    parser.add_argument("--out")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="models to evaluate, in this order")
    parser.add_argument("--results-dir", help="output folder (default evaluation/results)")
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
