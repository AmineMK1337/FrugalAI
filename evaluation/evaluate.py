"""Evaluate every model on the same data, same inputs and same hardware (CPU).

Performance: accuracy, precision, recall, specificity, F1, AUC (macro, and per class).
Efficiency:  file size, FLOPs, latency, throughput, memory (one thread, fresh process per run).
Environment: energy and CO2e, ESTIMATED by CodeCarbon. This VM has no CPU power counter, so the
             estimate is roughly a fixed CPU power x run time: compare models, do not quote it as measured.

FLOPs: ptflops counts MACs; FLOPs here use the common convention 1 MAC = 2 FLOPs.
To add a model (e.g. real INT8 PTQ or QAT), add an entry to MODELS.

Usage: python evaluation/evaluate.py [--split validation|test] [--repeats 3] [--country TUN]
Writes to evaluation/results/<split>/.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # same hardware for every model: CPU

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "resnet18_finetunning"
WEIGHT_ONLY = ROOT / "quantization" / "ptq" / "weight_only"
sys.path[:0] = [str(PROJECT / "src"), str(WEIGHT_ONLY)]
os.chdir(PROJECT)  # config and split paths are relative to the fine-tuning project

import quantize as q  # noqa: E402
from ham10000.data import build_dataloaders, build_test_loader  # noqa: E402
from ham10000.utils import load_config  # noqa: E402

CONFIG = "configs/strong_cosine25.yaml"
CHECKPOINT = PROJECT / "results/strong_cosine25_resnet18/resnet18_baseline.pth"
NUM_CLASSES = 7
BATCH_SIZE = 32
WARMUP, LATENCY_RUNS = 20, 300


def load_fp32():
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def load_weight_only(path):
    return lambda: q.load_dequantized_resnet18(path, NUM_CLASSES)


PER_CHANNEL = WEIGHT_ONLY / "results/resnet18_w8_perchannel.pt"
PER_TENSOR = WEIGHT_ONLY / "results/resnet18_w8_pertensor.pt"
MODELS = {  # name: (loader, file on disk)
    "FP32": (load_fp32, CHECKPOINT),
    "weight_only_per_channel": (load_weight_only(PER_CHANNEL), PER_CHANNEL),
    "weight_only_per_tensor": (load_weight_only(PER_TENSOR), PER_TENSOR),
}


# ── Performance ──────────────────────────────────────────────────────────────

def performance(name, probs, y, classes):
    pred = probs.argmax(1)
    ids = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=ids, zero_division=0)
    cm = confusion_matrix(y, pred, labels=ids)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    specificity = tn / (tn + cm.sum(0) - np.diag(cm))
    auc = np.array([roc_auc_score(y == i, probs[:, i]) for i in ids])
    per_class = pd.DataFrame({"model": name, "class": classes, "support": support, "precision": precision,
                              "recall": recall, "specificity": specificity, "f1": f1, "auc": auc})
    summary = {"model": name, "accuracy": accuracy_score(y, pred), "precision_macro": precision.mean(),
               "recall_macro": recall.mean(), "specificity_macro": specificity.mean(), "f1_macro": f1.mean(),
               "auc_macro": auc.mean()}
    return summary, per_class


# ── Efficiency and environment: one fresh process per run ────────────────────

def memory_kb(field):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1])
    raise RuntimeError(f"{field} not found")


@torch.inference_mode()
def worker(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sample = torch.load(Path(args.workdir) / "sample.pt", weights_only=True)
    before = memory_kb("VmRSS")
    model = MODELS[args.worker][0]()

    for _ in range(WARMUP):
        model(sample)
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        model(sample)
        times.append((time.perf_counter() - start) * 1000)
    peak_mb = (memory_kb("VmHWM") - before) / 1024  # measured before the image tensor is opened

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
        "estimated_cpu_power_w": data.cpu_power, "grid_g_per_kwh": data.emissions / data.energy_consumed * 1000,
    }))


def efficiency_static(name):
    from ptflops import get_model_complexity_info
    model = MODELS[name][0]()
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False,
                                             verbose=False)
    size = MODELS[name][1].stat().st_size
    return {"model": name, "size_mb": size / 1e6, "params": params, "gmacs": macs / 1e9, "gflops": 2 * macs / 1e9}


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    out = ROOT / "evaluation" / "results" / args.split
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    hashes = (q.sha256_file(CHECKPOINT), q.sha256_file(config.splits_csv))
    if args.split == "test":
        loader, _, class_to_idx = build_test_loader(config, num_workers=config.num_workers)
    else:
        _, loader, class_to_idx = build_dataloaders(config)
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    images, labels = map(torch.cat, zip(*loader))
    y = labels.numpy()
    print(f"{args.split}: {len(y)} images, classes {classes}", flush=True)

    summaries, per_classes, predictions = [], [], {}
    with torch.inference_mode():
        for name, (load, _) in MODELS.items():
            model = load()
            probs = torch.cat([model(b).softmax(1) for b in images.split(BATCH_SIZE)]).numpy()
            predictions[name] = probs.argmax(1)
            summary, per_class = performance(name, probs, y, classes)
            summary["identical_to_fp32_pct"] = (predictions[name] == predictions["FP32"]).mean() * 100
            summaries.append(summary)
            per_classes.append(per_class)
    performance_table = pd.DataFrame(summaries)
    performance_table.to_csv(out / "performance.csv", index=False)
    pd.concat(per_classes).to_csv(out / "performance_per_class.csv", index=False)

    if args.split == "validation":  # the earlier weight-only run must give the same accuracy
        earlier = pd.read_csv(WEIGHT_ONLY / "results/validation_results.csv").set_index("model")["accuracy"]
        names = {"FP32": "FP32", "weight_only_per_channel": "per_channel", "weight_only_per_tensor": "per_tensor"}
        for row in summaries:
            if abs(row["accuracy"] - earlier[names[row["model"]]]) > 1e-9:
                sys.exit(f"CHECK FAILED: {row['model']} accuracy differs from the earlier run")
        print("Accuracy matches the earlier weight-only run for all models", flush=True)

    rows = []
    with tempfile.TemporaryDirectory(dir=out) as workdir:
        torch.save(images, Path(workdir) / "images.pt")
        torch.save(images[:1].clone(), Path(workdir) / "sample.pt")
        for repeat in range(args.repeats):  # rotate the models so drift is not tied to one of them
            for name in MODELS:
                result = Path(workdir) / "result.json"
                subprocess.run([sys.executable, __file__, "--worker", name, "--workdir", workdir, "--out", str(result),
                                "--country", args.country], check=True, stderr=subprocess.DEVNULL)
                rows.append({"model": name, "repeat": repeat, **json.loads(result.read_text())})
                print(f"repeat {repeat + 1}/{args.repeats} {name}: {rows[-1]['latency_median_ms']:.1f} ms, "
                      f"{rows[-1]['throughput_img_s']:.1f} img/s", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    medians = runs.drop(columns="repeat").groupby("model", sort=False).median().reset_index()
    static = pd.DataFrame([efficiency_static(name) for name in MODELS])
    efficiency_cols = ["latency_median_ms", "latency_p95_ms", "throughput_img_s", "peak_memory_mb"]
    environment_cols = ["energy_wh_per_1000", "co2e_g_per_1000", "estimated_cpu_power_w", "grid_g_per_kwh"]
    efficiency = static.merge(medians[["model", *efficiency_cols]], on="model")
    environment = medians[["model", *environment_cols]]
    efficiency.to_csv(out / "efficiency.csv", index=False)
    environment.to_csv(out / "environment.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "split": args.split, "n_images": len(y), "repeats": args.repeats, "threads": 1, "device": "cpu",
        "batch_size_throughput": BATCH_SIZE, "latency_runs": LATENCY_RUNS, "country_for_co2e": args.country,
        "torch": torch.__version__, "cpu": subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True,
                                                          text=True).stdout.split(":")[-1].strip(),
        "python": platform.python_version(), "checkpoint_sha256": hashes[0],
    }, indent=2))

    if (q.sha256_file(CHECKPOINT), q.sha256_file(config.splits_csv)) != hashes:
        sys.exit("CHECK FAILED: the checkpoint or split file changed during the run")
    pd.set_option("display.width", 250, "display.max_columns", None)
    for title, table in (("Performance", performance_table), ("Efficiency", efficiency), ("Environment", environment)):
        print(f"\n== {title} ==\n{table.set_index('model').T.round(4).to_string()}")
    spread = runs.groupby("model", sort=False)[["latency_median_ms", "throughput_img_s"]].agg(["min", "max"]).round(2)
    print(f"\n== Spread across repeats ==\n{spread.to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--country", default="TUN", help="ISO-3 country of the machine, for CO2e only")
    parser.add_argument("--worker")
    parser.add_argument("--workdir")
    parser.add_argument("--out")
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
