"""Utilities for Static Quantization-Aware Training (QAT) ResNet-18.

Provides model preparation, module fusion, QAT setup, conversion,
checkpoint saving/loading, and weight inspection.
"""
from pathlib import Path
import hashlib
import torch
import torchvision.models.quantization as qmodels


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_qat_resnet18(num_classes, pretrained_state_dict=None):
    """Instantiate a quantizable ResNet-18, fuse modules, and insert QAT observers."""
    model = qmodels.resnet18(weights=None, num_classes=num_classes)
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict, strict=True)
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def convert_qat_model(model):
    """Convert a prepared/trained QAT model into an actual quantized INT8 model."""
    model.eval()
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    return quantized_model


def get_quantized_weights_dict(quantized_model):
    """Extract dequantized weight tensors and scales from converted INT8 layers."""
    weights = {}
    for name, module in quantized_model.named_modules():
        if hasattr(module, "weight") and callable(module.weight):
            qw = module.weight()
            if isinstance(qw, torch.Tensor) and qw.is_quantized:
                weights[f"{name}.weight"] = {
                    "qweight": qw,
                    "dequantized": qw.dequantize(),
                    "scale": qw.q_per_channel_scales() if qw.qscheme() in (
                        torch.per_channel_affine, torch.per_channel_symmetric
                    ) else qw.q_scale(),
                }
    return weights

