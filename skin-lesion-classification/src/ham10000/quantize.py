"""Helpers shared by quantize_ptq.py, train_qat.py and validation_summary.py.

Eager-mode quantization of torchvision's quantizable ResNet-18.

Engine: the x86 quantized engine dispatches to FBGEMM, and the installed PyTorch
build is compiled without FBGEMM (supported_engines == ['onednn']), so the
oneDNN engine is used.
Qconfig: the x86 defaults (get_default_qconfig("x86") / get_default_qat_qconfig("x86")).
They differ from the oneDNN defaults only in reduce_range=True for activations,
which keeps INT8 dot products from overflowing on CPUs without VNNI (AVX2 only).
"""
import os

import numpy as np
import torch
import torch.ao.nn.quantized as nnq
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision.models.quantization import resnet18 as quantizable_resnet18

from ham10000.data import Ham10000Dataset, eval_transform, lesion_split

ENGINE = "onednn"
QCONFIG_BACKEND = "x86"
LABEL = "onednn engine, x86 qconfig (reduce_range=True)"


def set_engine():
    torch.backends.quantized.engine = ENGINE


def qconfig_info(qconfig):
    """Describe a QConfig for settings.json: repr plus the activation reduce_range flag."""
    act = qconfig.activation()
    return {"qconfig": str(qconfig), "activation_observer": type(act).__name__,
            "reduce_range": bool(getattr(act, "reduce_range", getattr(getattr(act, "activation_post_process", None), "reduce_range", None)))}


def cpu_instruction_support():
    """CPU model and the SIMD features that matter for INT8 kernels, read from CPUID."""
    import cpuinfo  # py-cpuinfo
    info = cpuinfo.get_cpu_info()
    flags = set(info.get("flags", []))
    return {
        "cpu": info.get("brand_raw"),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": bool(flags & {"avx512_vnni", "avx512vnni"}),
        "avx_vnni": bool(flags & {"avx_vnni", "avxvnni"}),
        "source": f"py-cpuinfo {cpuinfo.CPUINFO_VERSION_STRING}",
    }


def engine_info():
    return {"engine": ENGINE, "supported_engines": list(torch.backends.quantized.supported_engines),
            "engine_note": "x86 engine unavailable: this PyTorch build has no FBGEMM, so oneDNN is the only quantized engine",
            "label": LABEL}


def load_quantizable_fp32(state_dict_path, num_classes):
    """torchvision quantizable ResNet-18 holding the fine-tuned FP32 weights."""
    model = quantizable_resnet18(weights=None, quantize=False, num_classes=num_classes)
    model.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=True), strict=True)
    return model


def eval_split_loader(config, split, indices=None, batch_size=64, num_workers=4):
    """Loader over the train or val split with the deterministic eval transform (no augmentation, no shuffle)."""
    ds = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=eval_transform(config))
    train_idx, val_idx, _ = lesion_split(ds.data, config)
    rows = {"train": train_idx, "val": val_idx}[split]
    if indices is not None:
        rows = [rows[i] for i in indices]
    return DataLoader(Subset(ds, rows), batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.inference_mode()
def predict(model, loader, device="cpu"):
    """Return (y_true, y_pred) for a classification model."""
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def val_scores(y_true, y_pred, class_to_idx):
    mel = class_to_idx["mel"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mel_recall": float(recall_score(y_true == mel, y_pred == mel, zero_division=0)),
    }


def check_int8(model, path, num_classes, image_size):
    """Sanity checks on a converted model and its TorchScript file; returns a dict of the results."""
    n_qconv = sum(isinstance(m, nnq.Conv2d) for m in model.modules())
    with torch.inference_mode():
        out_shape = tuple(torch.jit.load(path)(torch.randn(2, 3, image_size, image_size)).shape)
    size_mb = os.path.getsize(path) / 2**20
    checks = {"quantized_conv2d_modules": n_qconv, "output_shape": list(out_shape), "file_size_mb": size_mb}
    assert n_qconv > 0, "no QuantizedConv2d modules after convert"
    assert out_shape == (2, num_classes), f"unexpected output shape {out_shape}"
    assert 8 < size_mb < 15, f"INT8 file is {size_mb:.1f} MB, expected about 11 MB"
    return checks
