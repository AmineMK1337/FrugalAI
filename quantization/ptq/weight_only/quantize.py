"""Weight-only INT8 quantization: compress Conv2d/Linear weights, run inference in FP32.

scale = max|w| / 127 and q = clamp(round(w / scale), -127, 127), per output channel or
per tensor. Biases and BatchNorm entries are kept as they are. Loading dequantizes
(w = scale * q) back into a standard floating-point model, so there is no speedup:
this saves file size only.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

FORMAT_VERSION = 1
QMAX = 127
EPS32 = torch.finfo(torch.float32).eps


class WeightCheckFailed(RuntimeError):
    """A required check failed: stop and report, do not carry on."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_names(model):
    """State-dict keys of every Conv2d and Linear weight."""
    return [f"{name}.weight" for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]


def _scale_for(weight, per_channel):
    """FP32 scale(s). An all-zero channel or tensor gets scale 1, so its q is simply 0."""
    max_abs = weight.abs().flatten(1).amax(1) if per_channel else weight.abs().max()
    scale = (max_abs / QMAX).to(torch.float32)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def _broadcastable(scale, weight):
    """Per-channel scale reshaped along axis 0 (output channels for Conv2d and Linear)."""
    return scale.view(-1, *[1] * (weight.dim() - 1)) if scale.dim() else scale


def quantize_weights(model, per_channel):
    """Compress a floating-point model into INT8 weights + FP32 scales + everything else unchanged."""
    state = model.state_dict()
    names = weight_names(model)
    weights = {}
    for name in names:
        weight = state[name].detach()
        scale = _scale_for(weight, per_channel)
        # Divide in float64 by the scale that is actually stored, so rounding picks the nearest level
        ratio = weight.double() / _broadcastable(scale, weight).double()
        q = ratio.round().clamp(-QMAX, QMAX).to(torch.int8)
        weights[name] = {"q": q, "scale": scale.clone()}
    other = {name: tensor.detach().clone() for name, tensor in state.items() if name not in weights}
    return {"format_version": FORMAT_VERSION, "scheme": "per_channel" if per_channel else "per_tensor",
            "weights": weights, "other": other}


def dequantize_weights(compressed):
    """Rebuild a full FP32 state_dict: w = scale * q."""
    state = dict(compressed["other"])
    for name, entry in compressed["weights"].items():
        state[name] = _broadcastable(entry["scale"], entry["q"]) * entry["q"].to(torch.float32)
    return state


def save_compressed(compressed, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(compressed, path)


def load_compressed(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_dequantized_resnet18(path, num_classes):
    """A standard FP32 ResNet-18 in eval mode, with weights rebuilt from a compressed file."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.load_state_dict(dequantize_weights(load_compressed(path)), strict=True)
    return model.eval()


# ── Checks ───────────────────────────────────────────────────────────────────

def check_format(compressed, original_state):
    """INT8 in range, valid scales, the right keys, and non-weight entries bit-identical."""
    if set(compressed["weights"]) | set(compressed["other"]) != set(original_state):
        raise WeightCheckFailed("compressed keys do not match the original state_dict")
    if set(compressed["weights"]) & set(compressed["other"]):
        raise WeightCheckFailed("a key is stored both as a weight and as an unchanged entry")
    per_channel = compressed["scheme"] == "per_channel"
    for name, entry in compressed["weights"].items():
        q, scale = entry["q"], entry["scale"]
        if q.dtype != torch.int8 or q.shape != original_state[name].shape:
            raise WeightCheckFailed(f"{name}: q has dtype {q.dtype}, shape {tuple(q.shape)}")
        if q.min() < -QMAX or q.max() > QMAX:
            raise WeightCheckFailed(f"{name}: q out of range [{q.min()}, {q.max()}]")
        if scale.dtype != torch.float32 or not (torch.isfinite(scale).all() and (scale > 0).all()):
            raise WeightCheckFailed(f"{name}: bad scale (dtype {scale.dtype})")
        expected = (q.shape[0],) if per_channel else ()
        if tuple(scale.shape) != expected:
            raise WeightCheckFailed(f"{name}: scale shape {tuple(scale.shape)}, expected {expected}")
    for name, tensor in compressed["other"].items():
        if tensor.dtype != original_state[name].dtype or not torch.equal(tensor, original_state[name]):
            raise WeightCheckFailed(f"{name}: an unquantized entry changed")


def weight_error_rows(compressed, original_state):
    """Per-layer errors. Raises if any dequantized weight is further than scale/2 from the original.

    Errors are measured in float64. The limit allows for w_hat being stored in float32:
    scale/2 plus 2 float32 epsilons of |w|, which is at most 3e-5 of a scale step.
    """
    restored = dequantize_weights(compressed)
    rows = []
    for name, entry in compressed["weights"].items():
        w = original_state[name].double()
        w_hat = restored[name].double()
        half_step = _broadcastable(entry["scale"], entry["q"]).double().expand_as(w) / 2
        error = (w - w_hat).abs()
        limit = half_step + 2 * float(EPS32) * w.abs()
        if (error > limit).any():
            raise WeightCheckFailed(f"{name}: {int((error > limit).sum())} weights are further than scale/2")
        rows.append({
            "variant": compressed["scheme"], "layer": name, "n_weights": w.numel(),
            "scale_min": entry["scale"].min().item(), "scale_max": entry["scale"].max().item(),
            "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
            "relative_error": ((w - w_hat).norm() / w.norm()).item(),
            "max_error_over_half_scale": (error / half_step).max().item(),
            "pct_q_zero": (entry["q"] == 0).double().mean().item() * 100,
            "n_zero_channels": int((w.abs().flatten(1).amax(1) == 0).sum()),
        })
    return rows


def check_saved_file(path, compressed):
    """The file on disk holds exactly the same integers, scales and unchanged entries."""
    loaded = load_compressed(path)
    if loaded["scheme"] != compressed["scheme"] or set(loaded["weights"]) != set(compressed["weights"]):
        raise WeightCheckFailed(f"{path}: reloaded file differs")
    for name, entry in compressed["weights"].items():
        if not (torch.equal(loaded["weights"][name]["q"], entry["q"])
                and torch.equal(loaded["weights"][name]["scale"], entry["scale"])):
            raise WeightCheckFailed(f"{path}: {name} differs after reload")
    for name, tensor in compressed["other"].items():
        if not torch.equal(loaded["other"][name], tensor):
            raise WeightCheckFailed(f"{path}: {name} differs after reload")
