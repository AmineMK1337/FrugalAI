"""ResNet18: FP32 vs INT8 (post-training static quantization) on an ImageFolder dataset.

Measures classification metrics, model size, parameters, GFLOPs (FP32 only),
latency, throughput, and CodeCarbon energy/CO2e estimates (inference only).
"""
import copy
import os
import random
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
import codecarbon
from codecarbon import OfflineEmissionsTracker
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
# Quantization-ready ResNet18: same weights/layout as the standard model, but its
# residual additions use FloatFunctional so they can be quantized.
from torchvision.models.quantization import resnet18 as quantizable_resnet18

# ============================== CONFIGURATION ==============================
DATA_DIR = "dataset"              # contains train/ val/ test/
BATCH_SIZE = 32
NUM_CLASSES = 2
IMAGE_SIZE = 224
NUM_WORKERS = 0                   # keep 0 on Windows unless you know you need more
NUM_INFERENCE_IMAGES = 256        # same images used for latency and energy, FP32 and INT8
MODEL_CHECKPOINT_PATH = None      # e.g. "resnet18_baseline.pth" (plain state_dict); None = ImageNet weights
DEVICE = "cpu"                    # INT8 runs on CPU, so the comparison is done on CPU

SEED = 42
CALIB_SOURCE = "train"            # "train" or "val" -- NEVER "test"
CALIB_BATCHES = 10
TIMING_REPEATS = 5                # latency is averaged over several repetitions
ENERGY_REPEATS = 5                # passes over the inference images inside the tracker
COUNTRY_ISO_CODE = "TUN"          # for CodeCarbon's carbon-intensity lookup (offline mode)
OUTPUT_CSV = "results_quantization.csv"
# ===========================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------ Step 1: data ------------------------------
def get_loaders():
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    gen = torch.Generator().manual_seed(SEED)
    loaders, datasets_ = {}, {}
    for split in ["train", "val", "test"]:
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=tf)
        shuffle = split == "train"  # only used for picking calibration batches
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                                    num_workers=NUM_WORKERS, generator=gen if shuffle else None)
        datasets_[split] = ds
        print(f"{split}: {len(ds)} images, classes={ds.class_to_idx}")
    if len(datasets_["test"].classes) != NUM_CLASSES:
        raise SystemExit(f"NUM_CLASSES={NUM_CLASSES} but found {datasets_['test'].classes}")
    return loaders


def get_inference_batches(test_loader, n_images):
    """Cache the first n test images in memory so data loading is excluded from timing/energy."""
    chunks, count = [], 0
    for x, _ in test_loader:
        chunks.append(x)
        count += len(x)
        if count >= n_images:
            break
    x = torch.cat(chunks)[:n_images]
    return list(torch.split(x, BATCH_SIZE))


# ----------------------------- Step 2: FP32 model -----------------------------
def build_fp32_model():
    weights = None if MODEL_CHECKPOINT_PATH else ResNet18_Weights.DEFAULT
    model = quantizable_resnet18(weights=weights, quantize=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if MODEL_CHECKPOINT_PATH:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu"))
    else:
        print("WARNING: no checkpoint -> final layer is randomly initialised. "
              "Metrics are meaningless; use this only to test the pipeline.")
    return model.eval()


# --------------------------- Step 3: evaluation ---------------------------
def compute_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(1)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    sens_list, spec_list = [], []
    for i in labels:  # one-vs-rest sensitivity/specificity per class
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if tp + fn else np.nan)
        spec_list.append(tn / (tn + fp) if tn + fp else np.nan)
    if NUM_CLASSES == 2:   # binary: class index 1 is the "positive" class
        sens, spec = sens_list[1], spec_list[1]
    else:                  # multi-class: macro average
        sens, spec = np.nanmean(sens_list), np.nanmean(spec_list)

    try:
        if NUM_CLASSES == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError as e:  # e.g. a class missing from the test set
        print("ROC-AUC not computed:", e)
        auc = np.nan

    return {"accuracy": float((y_pred == y_true).mean()), "precision_macro": p,
            "recall_macro": r, "f1_macro": f1, "sensitivity": sens,
            "specificity": spec, "roc_auc": auc}, cm


@torch.no_grad()
def evaluate_model(model, loader, device, warmup_batches=2):
    """Classification metrics + model-only forward time over the full test set."""
    first_x, _ = next(iter(loader))
    for _ in range(warmup_batches):
        model(first_x.to(device))
    sync(device)

    y_true, y_prob, model_time = [], [], 0.0
    for x, y in loader:
        x = x.to(device)
        sync(device)
        t0 = time.perf_counter()
        logits = model(x)
        sync(device)
        model_time += time.perf_counter() - t0
        y_prob.append(torch.softmax(logits, 1).cpu())
        y_true.append(y)

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy().astype(np.float64)
    metrics, cm = compute_metrics(y_true, y_prob)
    metrics["eval_ms_per_image"] = 1000 * model_time / len(y_true)
    return metrics, cm


@torch.no_grad()
def measure_latency(model, batches, device, repeats=TIMING_REPEATS):
    """Repeated timing on the cached images (model forward only)."""
    n_images = sum(len(b) for b in batches)
    for b in batches[:2]:                      # warm-up
        model(b.to(device))
    sync(device)
    per_image_ms = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        for b in batches:
            model(b.to(device))
        sync(device)
        per_image_ms.append(1000 * (time.perf_counter() - t0) / n_images)
    mean_ms = float(np.mean(per_image_ms))
    return {"latency_ms_per_image": mean_ms, "latency_std_ms": float(np.std(per_image_ms)),
            "throughput_img_s": 1000 / mean_ms}


# ------------------------ Step 4: INT8 quantization ------------------------
def quantize_static_ptq(fp32_model, calib_loader):
    """Eager-mode static PTQ: fuse -> prepare -> calibrate -> convert (CPU only)."""
    supported = torch.backends.quantized.supported_engines
    print(f"Supported quantized engines on this build: {supported}")
    # Prefer x86 backends; qnnpack is mainly for ARM. Only use what this build supports.
    engine = next((e for e in ["fbgemm", "x86", "onednn", "qnnpack"] if e in supported), None)
    if engine is None:
        raise RuntimeError(f"No usable quantized engine in {supported}")
    torch.backends.quantized.engine = engine
    model = copy.deepcopy(fp32_model).cpu().eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fuse_model(is_qat=False)                       # Conv+BN(+ReLU) fusion
        model.qconfig = torch.ao.quantization.get_default_qconfig(engine)
        torch.ao.quantization.prepare(model, inplace=True)   # insert observers
        with torch.no_grad():                                # calibration (no test data!)
            for i, (x, _) in enumerate(calib_loader):
                if i >= CALIB_BATCHES:
                    break
                model(x)
        torch.ao.quantization.convert(model, inplace=True)   # -> INT8 modules
    for w in caught:
        if "deprecat" in str(w.message).lower():
            print("PYTORCH DEPRECATION WARNING:", w.message)
    print(f"Static PTQ done with backend '{engine}' using {CALIB_BATCHES} calibration batches.")
    return model


def quantize_dynamic_fallback(fp32_model):
    """Fallback: dynamic quantization only converts nn.Linear (the final fc layer in ResNet18)."""
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)


# --------------------- Steps 6-7: size, params, GFLOPs ---------------------
def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fp32_gflops(model):
    """GFLOPs ~= 2 x multiply-accumulates (torchinfo). Only valid for the FP32 model."""
    try:
        from torchinfo import summary
        info = summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), verbose=0, device="cpu")
        model.eval()
        return 2 * info.total_mult_adds / 1e9
    except Exception as e:
        print("GFLOPs not computed (pip install torchinfo):", e)
        return float("nan")


# ------------------------ Step 8: energy / carbon ------------------------
@torch.no_grad()
def measure_energy(model, batches, device, tag):
    """CodeCarbon estimate for the inference loop only."""
    n_images = sum(len(b) for b in batches) * ENERGY_REPEATS
    for b in batches[:2]:                      # warm-up BEFORE the tracker starts
        model(b.to(device))
    tracker = OfflineEmissionsTracker(
        project_name=f"resnet18_{tag}", country_iso_code=COUNTRY_ISO_CODE,
        output_dir=".", output_file="codecarbon_emissions.csv", save_to_file=True,
        measure_power_secs=1, log_level="error")
    tracker.start()
    try:
        for _ in range(ENERGY_REPEATS):
            for b in batches:
                model(b.to(device))
        sync(device)
    finally:
        co2_kg = tracker.stop()
    energy_kwh = getattr(tracker.final_emissions_data, "energy_consumed", None)
    co2_kg = float("nan") if co2_kg is None else float(co2_kg)
    energy_kwh = float("nan") if energy_kwh is None else float(energy_kwh)
    co2_g_per_image = co2_kg * 1000 / n_images
    return {"energy_images": n_images, "energy_kwh_total": energy_kwh,
            "energy_kwh_per_image": energy_kwh / n_images,
            "co2e_g_total": co2_kg * 1000, "co2e_g_per_image": co2_g_per_image,
            "co2e_g_per_1000_images": co2_g_per_image * 1000}


def pct_drop(fp32_value, int8_value):
    return 100 * (fp32_value - int8_value) / fp32_value if fp32_value else float("nan")


# ================================== MAIN ==================================
def main():
    set_seed(SEED)
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"torchvision {torchvision.__version__} | scikit-learn {sklearn.__version__} | "
          f"codecarbon {codecarbon.__version__}")
    device = DEVICE
    if device != "cpu":
        print("INT8 (fbgemm/qnnpack) runs on CPU -> forcing CPU for a fair FP32 vs INT8 comparison.")
        device = "cpu"
    print(f"Device: {device} | torch threads: {torch.get_num_threads()}")

    loaders = get_loaders()
    print(f"Test images: {len(loaders['test'].dataset)}")
    batches = get_inference_batches(loaders["test"], NUM_INFERENCE_IMAGES)
    print(f"Inference/energy images (same for both models): {sum(len(b) for b in batches)}")

    fp32 = build_fp32_model().to(device)
    try:
        int8, qtype = quantize_static_ptq(fp32, loaders[CALIB_SOURCE]), "static PTQ INT8"
    except Exception as e:
        print(f"\nStatic quantization FAILED: {type(e).__name__}: {e}")
        print("Falling back to dynamic quantization (only the final Linear layer is INT8; "
              "conv layers stay FP32, so expect little size/speed gain).")
        try:
            int8, qtype = quantize_dynamic_fallback(fp32), "dynamic INT8 (fc only)"
        except Exception as e2:
            raise SystemExit(f"Dynamic quantization also failed: {e2}\n"
                             "Check your PyTorch version/platform (see 'Common errors').")

    fp32_total, fp32_trainable = count_params(fp32)
    gflops = fp32_gflops(fp32)
    rows, cms = [], {}
    for name, q, model, tag in [("ResNet18 FP32", "none", fp32, "fp32"),
                                ("ResNet18 INT8", qtype, int8, "int8")]:
        print(f"\n=== {name} ===")
        metrics, cm = evaluate_model(model, loaders["test"], device)
        cms[tag] = cm
        row = {"model": name, "quantization": q, **metrics}
        row.update(measure_latency(model, batches, device))
        # Quantized modules hide weights in packed params, so reuse the FP32 architecture counts.
        row.update({"params_total": fp32_total, "params_trainable": fp32_trainable,
                    "size_mb": model_size_mb(model),
                    "gflops_fp32_only": gflops if tag == "fp32" else float("nan")})
        row.update(measure_energy(model, batches, device, tag))
        rows.append(row)

    fp32_size, int8_size = rows[0]["size_mb"], rows[1]["size_mb"]
    for r in rows:
        r["compression_ratio"] = fp32_size / r["size_mb"]
        r["size_reduction_pct"] = pct_drop(fp32_size, r["size_mb"])

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    for tag, cm in cms.items():
        names = loaders["test"].dataset.classes
        pd.DataFrame(cm, index=[f"true_{c}" for c in names],
                     columns=[f"pred_{c}" for c in names]).to_csv(f"confusion_matrix_{tag}.csv")
    pd.options.display.float_format = "{:.6g}".format
    pd.set_option("display.width", 200)
    print("\n================ RESULTS (CodeCarbon values are ESTIMATES) ================")
    print(df.set_index("model").T.to_string())
    print(f"\nSaved {OUTPUT_CSV}, confusion_matrix_fp32.csv, confusion_matrix_int8.csv, "
          "codecarbon_emissions.csv")

    a, b = rows[0], rows[1]
    print("\n================ FP32 vs INT8 ================")
    for m in ["accuracy", "sensitivity", "specificity", "f1_macro", "roc_auc"]:
        print(f"{m:>12} drop: {pct_drop(a[m], b[m]):.2f} %")
    print(f"Size reduction:  {b['size_reduction_pct']:.1f} % (compression {b['compression_ratio']:.2f}x)")
    print(f"Latency speedup: {a['latency_ms_per_image'] / b['latency_ms_per_image']:.2f}x")
    print(f"Energy reduction: {pct_drop(a['energy_kwh_total'], b['energy_kwh_total']):.1f} %")
    print(f"Carbon reduction: {pct_drop(a['co2e_g_total'], b['co2e_g_total']):.1f} %")


if __name__ == "__main__":   # required on Windows when NUM_WORKERS > 0
    main()