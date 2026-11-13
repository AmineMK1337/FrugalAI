"""Evaluate every model on the same data, same inputs and same hardware (CPU), and save the results by
model and metric type, each CSV next to its plots.

Performance: accuracy, precision, recall, specificity, F1, AUC (macro averages, and per class), each with a
             95% bootstrap CI and a paired bootstrap against FP32 (same resamples for both models).
Efficiency:  size, FLOPs (ptflops MACs x 2), latency (batch 1, median and p95), throughput (batch 32),
             peak memory, at 1, 2, 4 and 8 threads. Fresh process per run, 20 warm-up + 300 timed runs,
             3 repeats, models and thread counts interleaved so drift is not tied to one of them.
Environment: energy and CO2e per 1,000 images, at 1, 2, 4 and 8 threads, ESTIMATED by CodeCarbon. This VM has
             no RAPL counter, so CodeCarbon assumes a CPU power: compare models, do not quote as measured.

Models: fp32, weight_only_per_channel, weight_only_per_tensor, static_ptq (seed-42 calibration). To add a
model, add an entry to MODELS. Nothing is trained, calibrated or modified; model files, split and config are
hashed before and after the run.

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
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # same hardware for every model: CPU

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT / "resnet18_finetunning"
WEIGHT_ONLY = ROOT / "quantization" / "ptq" / "weight_only"
STATIC = ROOT / "quantization" / "ptq" / "static"
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
NUM_CLASSES = 7
BATCH_SIZE = 32
WARMUP, LATENCY_RUNS = 20, 300
THREADS = (1, 2, 4, 8)  # 1 = main protocol
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0
ENGINE = "x86"
PERFORMANCE = ("accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro", "auc_macro")


def load_fp32():
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def load_weight_only(path):
    return lambda: q.load_dequantized_resnet18(path, NUM_CLASSES)


def load_static():
    torch.backends.quantized.engine = ENGINE
    return torch.jit.load(str(STATIC_INT8), map_location="cpu").eval()


MODELS = {  # name: loader, file on disk, note
    "fp32": dict(load=load_fp32, file=CHECKPOINT, note="FP32 eager PyTorch"),
    "weight_only_per_channel": dict(load=load_weight_only(PER_CHANNEL), file=PER_CHANNEL,
                                    note="INT8 weights on disk, dequantized to FP32 at load: computes in FP32"),
    "weight_only_per_tensor": dict(load=load_weight_only(PER_TENSOR), file=PER_TENSOR,
                                   note="INT8 weights on disk, dequantized to FP32 at load: computes in FP32"),
    "static_ptq": dict(load=load_static, file=STATIC_INT8,
                       note="INT8 TorchScript (x86), BatchNorm fused, seed-42 calibration; FLOPs counted on the "
                            "FP32 architecture because ptflops cannot count quantized TorchScript"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
               "f1_macro": f1.mean(), "auc_macro": np.nanmean(auc)}
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


def static_seed_summary():
    """The 3 calibration seeds of static PTQ, from quantization/ptq/static/results (nothing is recalibrated)."""
    table = pd.read_csv(STATIC / "results/validation_results.csv")
    seeds = table[table["model"] != "FP32"][["model", *PERFORMANCE]].copy()
    seeds["model"] = "static_ptq_seed" + table.loc[seeds.index, "seed"].astype(int).astype(str)
    numeric = seeds.drop(columns="model")
    stats = pd.DataFrame([{"model": f"static_ptq_seed_{s}", **getattr(numeric, s)().to_dict()}
                          for s in ("mean", "min", "max")])
    return pd.concat([seeds, stats], ignore_index=True)


def check_against_earlier_results(overall):
    """The models did not change, so performance must equal the results saved when they were made."""
    weight_only = pd.read_csv(WEIGHT_ONLY / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("weight_only_per_channel", "per_channel"),
                      ("weight_only_per_tensor", "per_tensor")):
        for metric, old_metric in (("accuracy", "accuracy"), ("f1_macro", "macro_f1")):
            if abs(overall[name][metric] - weight_only.loc[old, old_metric]) > 1e-9:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {WEIGHT_ONLY}/results")
    static = pd.read_csv(STATIC / "results/validation_results.csv").set_index("model")
    for name, old in (("fp32", "FP32"), ("static_ptq", "INT8_seed42")):
        for metric in PERFORMANCE:
            if abs(overall[name][metric] - static.loc[old, metric]) > 1e-9:
                sys.exit(f"CHECK FAILED: {name} {metric} differs from {STATIC}/results")
    print("Performance matches the results saved when the models were made", flush=True)


# ── Efficiency and environment: one fresh process per run ────────────────────

def memory_kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1])
    raise RuntimeError(f"{field} not found")


@torch.inference_mode()
def worker(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sample = torch.load(Path(args.workdir) / "sample.pt", weights_only=True)
    before = memory_kb("VmRSS")
    model = MODELS[args.worker]["load"]()

    for _ in range(WARMUP):
        model(sample)
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        model(sample)
        times.append((time.perf_counter() - start) * 1000)
    peak_mb = (memory_kb("VmHWM") - before) / 1024  # load + batch-1 inference, before the image tensor is opened

    from codecarbon import OfflineEmissionsTracker
    images = torch.load(Path(args.workdir) / "images.pt", weights_only=True, mmap=True)
    batches = [images[i:i + BATCH_SIZE].clone() for i in range(0, len(images), BATCH_SIZE)]
    tracker = OfflineEmissionsTracker(country_iso_code=args.country, measure_power_secs=1, tracking_mode="process",
                                      save_to_file=False, log_level="critical", allow_multiple_runs=True)
    tracker.start()
    start = time.perf_counter()
    for batch in batches:
        model(batch)
    seconds = time.perf_counter() - start
    tracker.stop()
    data = tracker.final_emissions_data
    per_1000 = 1000 / len(images)
    Path(args.out).write_text(json.dumps({
        "latency_median_ms": float(np.median(times)), "latency_p95_ms": float(np.percentile(times, 95)),
        "throughput_img_s": len(images) / seconds, "peak_memory_mb": peak_mb,
        "energy_wh_per_1000": data.energy_consumed * 1000 * per_1000,
        "co2e_g_per_1000": data.emissions * 1000 * per_1000,
        "estimated_power_w": data.cpu_power + data.ram_power,
        "grid_g_per_kwh": data.emissions / data.energy_consumed * 1000,
        "latency_samples_ms": times,
    }))


def flops(name):
    from ptflops import get_model_complexity_info
    counted = name if name != "static_ptq" else "fp32"  # quantized TorchScript cannot be counted
    macs, _ = get_model_complexity_info(MODELS[counted]["load"](), (3, 224, 224), as_strings=False,
                                        print_per_layer_stat=False, verbose=False)
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
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    print(f"{split}: {len(y)} images, classes {classes}", flush=True)
    for folder in [RESULTS / m for m in MODELS] + [RESULTS / "comparison"]:
        shutil.rmtree(folder, ignore_errors=True)  # outputs of an earlier run of this script

    # Performance
    probs = {}
    with torch.inference_mode():
        for name, spec in MODELS.items():
            model = spec["load"]()
            probs[name] = torch.cat([model(b).softmax(1) for b in images.split(BATCH_SIZE)]).numpy()
    for name in MODELS:
        check_scores(y, probs[name], len(classes))
    boot = bootstrap(probs, y, len(classes))
    overall = {}
    for name in MODELS:
        point, per_class = scores(y, probs[name], len(classes))
        metrics = {"model": name}
        for metric in PERFORMANCE:
            metrics[metric] = float(point[metric])
            metrics[f"{metric}_ci_low"], metrics[f"{metric}_ci_high"] = ci(boot[name][metric])
        for metric in PERFORMANCE if name != "fp32" else ():
            diff = boot[name][metric] - boot["fp32"][metric]
            metrics[f"{metric}_change_vs_fp32"] = metrics[metric] - overall["fp32"][metric]
            metrics[f"{metric}_change_ci_low"], metrics[f"{metric}_change_ci_high"] = ci(diff)
            metrics[f"{metric}_paired_bootstrap_p"] = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
        metrics["n_images"] = len(y)
        metrics["note"] = MODELS[name]["note"]
        overall[name] = metrics
        folder = RESULTS / name / "performance"
        save_csv(pd.DataFrame([metrics]), folder, "metrics", split)
        save_csv(pd.DataFrame({"class": classes, **per_class}), folder, "per_class", split)
    if split == "validation":
        check_against_earlier_results(overall)
        save_csv(static_seed_summary(), RESULTS / "static_ptq" / "performance", "seed_summary", split)

    # Efficiency and environment: every model and thread count, back to back, in this session
    rows, samples = [], []
    with tempfile.TemporaryDirectory(dir=RESULTS) as workdir:
        torch.save(images, Path(workdir) / "images.pt")
        torch.save(images[:1].clone(), Path(workdir) / "sample.pt")
        for repeat in range(args.repeats):
            for threads in THREADS:
                for name in MODELS:
                    result = Path(workdir) / "result.json"
                    subprocess.run([sys.executable, __file__, "--worker", name, "--threads", str(threads),
                                    "--workdir", workdir, "--out", str(result), "--country", args.country],
                                   check=True, stderr=subprocess.DEVNULL)
                    run = json.loads(result.read_text())
                    samples += [{"model": name, "threads": threads, "repeat": repeat, "run": i, "latency_ms": t}
                                for i, t in enumerate(run.pop("latency_samples_ms"))]
                    rows.append({"model": name, "threads": threads, "repeat": repeat, **run})
                    print(f"repeat {repeat + 1}/{args.repeats}, {threads} thread(s), {name}: "
                          f"{run['latency_median_ms']:.2f} ms, {run['throughput_img_s']:.1f} img/s", flush=True)
    runs = pd.DataFrame(rows)
    samples = pd.DataFrame(samples)
    medians = runs.drop(columns="repeat").groupby(["model", "threads"]).median()

    for name in MODELS:
        efficiency = {"model": name, "size_mb": MODELS[name]["file"].stat().st_size / 1e6, "gflops": flops(name)}
        environment = {"model": name}
        for t in THREADS:
            m = medians.loc[(name, t)]
            efficiency.update({f"latency_median_ms_{t}t": m["latency_median_ms"],
                               f"latency_p95_ms_{t}t": m["latency_p95_ms"],
                               f"throughput_img_s_{t}t": m["throughput_img_s"],
                               f"peak_memory_mb_{t}t": m["peak_memory_mb"]})
            environment.update({f"energy_wh_per_1000_{t}t": m["energy_wh_per_1000"],
                                f"co2e_g_per_1000_{t}t": m["co2e_g_per_1000"],
                                f"estimated_power_w_{t}t": m["estimated_power_w"]})
        efficiency["note"] = MODELS[name]["note"]
        environment.update({"source": "estimated (CodeCarbon process mode; no RAPL counter on this VM)",
                            "grid_g_per_kwh": medians.loc[(name, 1), "grid_g_per_kwh"], "country": args.country})
        save_csv(pd.DataFrame([efficiency]), RESULTS / name / "efficiency", "metrics", split)
        save_csv(samples[samples["model"] == name], RESULTS / name / "efficiency", "latency_samples", split)
        save_csv(runs[runs["model"] == name].drop(columns=["energy_wh_per_1000", "co2e_g_per_1000",
                                                           "estimated_power_w", "grid_g_per_kwh"]),
                 RESULTS / name / "efficiency", "runs", split)
        save_csv(runs[runs["model"] == name][["model", "threads", "repeat", "energy_wh_per_1000", "co2e_g_per_1000",
                                              "estimated_power_w", "grid_g_per_kwh"]],
                 RESULTS / name / "environment", "runs", split)
        save_csv(pd.DataFrame([environment]), RESULTS / name / "environment", "metrics", split)

    write_comparison(split)
    after = {name: sha256(path) for name, path in tracked.items()}
    if after != hashes:
        sys.exit(f"CHECK FAILED: files changed during the run: {[k for k in hashes if hashes[k] != after[k]]}")
    (RESULTS / "run_info.json").write_text(json.dumps({
        "split": split, "n_images": len(y), "classes": classes, "device": "cpu",
        "cpu": subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True,
                              text=True).stdout.split(":")[-1].strip(),
        "logical_cpus": os.cpu_count(), "threads": list(THREADS), "threads_main": 1, "repeats": args.repeats,
        "warmup_runs": WARMUP, "latency_runs": LATENCY_RUNS, "throughput_batch_size": BATCH_SIZE,
        "bootstrap_resamples": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "quantized_engine": ENGINE,
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
    seeds = RESULTS / "static_ptq" / "performance" / f"seed_summary_{split}.csv"
    if seeds.exists():
        seeds = pd.read_csv(seeds)
        perf = pd.concat([perf, seeds[seeds["model"].str.startswith("static_ptq_seed_")]], ignore_index=True)
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
    for t in THREADS:
        for metric in ("energy_wh_per_1000", "co2e_g_per_1000"):
            env[f"{metric}_change_vs_fp32_pct_{t}t"] = (env[f"{metric}_{t}t"] / base[f"{metric}_{t}t"] - 1) * 100
    save_csv(env, out / "environment", "table", split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--country", default="TUN", help="ISO-3 country of the machine, for CO2e only")
    parser.add_argument("--worker")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workdir")
    parser.add_argument("--out")
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
