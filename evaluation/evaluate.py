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

Usage: python evaluation/evaluate.py [--split validation|test] [--repeats 3] [--country TUN]
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
STATIC = ROOT / "quantization" / "ptq" / "static"
QAT = ROOT / "quantization" / "qat"
RESULTS = HERE / "results"
sys.path[:0] = [str(PROJECT / "src"), str(WEIGHT_ONLY)]
os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project

import quantize as q  # noqa: E402
from ham10000.data import build_dataloaders, build_test_loader  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = PROJECT / "results/strong_cosine25_resnet18/resnet18_baseline.pth"
PER_CHANNEL = WEIGHT_ONLY / "results/resnet18_w8_perchannel.pt"
PER_TENSOR = WEIGHT_ONLY / "results/resnet18_w8_pertensor.pt"
STATIC_INT8 = STATIC / "resnet18_static_int8.pt"
QAT_INT8 = QAT / "resnet18_qat_int8.pt"
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
    "static_ptq": dict(state=None, unfused=None, file=STATIC_INT8,
                       note="INT8 weights and activations (x86 engine), seed-42 calibration"),
    "qat": dict(state=None, unfused=None, file=QAT_INT8,
                note="INT8 weights and activations (x86 engine), QAT v2 (3-phase schedule), seed 42, best of "
                     "epochs 8-10"),
}
# Seed summaries, read from the results of the script that made each model (nothing is retrained here):
# model folder: [(results file, model-name prefix there (+ seed number), name in the comparison table)]
SEED_SUMMARIES = {
    "static_ptq": [(STATIC / "results/validation_results.csv", "INT8_seed", "static_ptq")],
    "qat": [(QAT / "results/validation_results.csv", "QAT_seed", "qat"),
            (QAT / "results/validation_results.csv", "QAT_epoch10_seed", "qat_epoch10"),
            (QAT / "results/validation_results.csv", "control_selected_seed", "fp32_control_selected"),
            (QAT / "results/validation_results.csv", "control_epoch10_seed", "fp32_control_epoch10")],
}
QAT_MINUS_CONTROL = {  # comparison row: matching FP32-control row (same selection rule; same seed or seed mean)
    "qat": "fp32_control_selected_seed42", "qat_seed_mean": "fp32_control_selected_seed_mean",
    "qat_epoch10_seed42": "fp32_control_epoch10_seed42", "qat_epoch10_seed_mean": "fp32_control_epoch10_seed_mean"}
QAT_MINUS_CONTROL_METRICS = ("f1_macro", "accuracy", "melanoma_recall")


def export_torchscript(workdir):
    """Path of each model's fused TorchScript file. Static PTQ and QAT already are; the others are traced here."""
    paths = {}
    for name, spec in MODELS.items():
        if spec["state"] is None:
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


def seed_summary(name):
    """Per-seed rows and mean/min/max over seeds: static PTQ (3 calibration seeds), or QAT (selected and epoch 10)
    with its FP32 control (selected and epoch 10), from the results saved by the script that made them."""
    parts = []
    for path, prefix, label in SEED_SUMMARIES[name]:
        table = pd.read_csv(path)
        if "balanced_accuracy" not in table:  # older results: balanced accuracy is the macro recall
            table["balanced_accuracy"] = table["recall_macro"]
        seeds = table[table["model"].str.fullmatch(prefix + r"\d+")][["model", "seed", "epoch", *PERFORMANCE]
                                                                      if "epoch" in table else
                                                                      ["model", "seed", *PERFORMANCE]].copy()
        if len(seeds) == 0:
            sys.exit(f"CHECK FAILED: no '{prefix}<seed>' rows in {path}")
        seeds["model"] = f"{label}_seed" + seeds["seed"].astype(int).astype(str)
        numeric = seeds[list(PERFORMANCE)]
        stats = pd.DataFrame([{"model": f"{label}_seed_{s}", **getattr(numeric, s)().to_dict()}
                              for s in ("mean", "min", "max")])
        parts += [seeds, stats]
    return pd.concat(parts, ignore_index=True)


def check_against_earlier_results(overall):
    """Fusing changes only rounding, so performance must equal the results saved when the models were made
    (exactly for prediction-based metrics; AUC, which uses probabilities, up to 1e-4)."""
    tolerance = {m: 1e-9 for m in PERFORMANCE} | {"auc_macro": 1e-4}
    weight_only = pd.read_csv(WEIGHT_ONLY / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("weight_only_per_channel", "per_channel"),
                      ("weight_only_per_tensor", "per_tensor")):
        for metric, old_metric in (("accuracy", "accuracy"), ("f1_macro", "macro_f1")):
            if abs(overall[name][metric] - weight_only.loc[old, old_metric]) > tolerance[metric]:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {WEIGHT_ONLY}/results")
    static = pd.read_csv(STATIC / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("static_ptq", "INT8_seed42")):
        for metric in (m for m in PERFORMANCE if m in static):
            if abs(overall[name][metric] - static.loc[old, metric]) > tolerance[metric]:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {STATIC}/results")
    qat = pd.read_csv(QAT / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("qat", "QAT_seed42")):
        for metric in (m for m in PERFORMANCE if m in qat):
            if abs(overall[name][metric] - qat.loc[old, metric]) > tolerance[metric]:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {QAT}/results")
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


def flops():
    """Same for every model: the fused FP32 architecture (ptflops cannot count TorchScript or INT8 ops)."""
    from ptflops import get_model_complexity_info
    model = quantizable_resnet18(weights=None, quantize=False, num_classes=NUM_CLASSES)
    model.load_state_dict(fp32_state(), strict=True)
    model.eval().fuse_model()
    macs, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False,
                                        verbose=False)
    return 2 * macs / 1e9


# ── Main ─────────────────────────────────────────────────────────────────────

def save_csv(table, folder, name, split):
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / f"{name}_{split}.csv", index=False)


def main(args):
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
            for name in SEED_SUMMARIES:
                save_csv(seed_summary(name), RESULTS / name / "performance", "seed_summary", split)

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
    gflops = flops()

    for name in MODELS:
        efficiency = {"model": name, "format": FORMAT, "size_mb": MODELS[name]["file"].stat().st_size / 1e6,
                      "gflops": gflops}
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
            "static_ptq": "saved fused INT8 TorchScript (torch.jit.script), loaded as is",
            "qat": "saved fused INT8 TorchScript (torch.jit.script), loaded as is (QAT v2, 3-phase schedule)"},
        "fp32_control": "reference rows only (FP32 fine-tuned with the QAT schedule, no fake quantization), from "
                        "quantization/qat/results/validation_results.csv; not timed",
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
    plot_results.main(split)


def write_comparison(split):
    out = RESULTS / "comparison"
    perf = pd.concat([pd.read_csv(RESULTS / m / "performance" / f"metrics_{split}.csv") for m in MODELS])
    every_seed = []
    for name in SEED_SUMMARIES:
        seeds = RESULTS / name / "performance" / f"seed_summary_{split}.csv"
        if seeds.exists():
            seeds = pd.read_csv(seeds)
            every_seed.append(seeds)
            perf = pd.concat([perf, seeds[seeds["model"].str.contains(r"_seed_(?:mean|min|max)$")]],
                             ignore_index=True)
    # FP32 control as reference rows, and QAT - control (the comparison that isolates fake quantization)
    lookup = pd.concat([perf, *every_seed], ignore_index=True).drop_duplicates("model").set_index("model")
    extra = [m for m in ("qat_epoch10_seed42", "fp32_control_selected_seed42", "fp32_control_epoch10_seed42")
             if m in lookup.index and m not in set(perf["model"])]
    perf = pd.concat([perf, lookup.loc[extra].reset_index()], ignore_index=True)
    for metric in QAT_MINUS_CONTROL_METRICS:
        perf[f"{metric}_qat_minus_control"] = np.nan
    for qat_row, control_row in QAT_MINUS_CONTROL.items():
        if qat_row in lookup.index and control_row in lookup.index:
            for metric in QAT_MINUS_CONTROL_METRICS:
                perf.loc[perf["model"] == qat_row, f"{metric}_qat_minus_control"] = (
                    lookup.loc[qat_row, metric] - lookup.loc[control_row, metric])
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
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
