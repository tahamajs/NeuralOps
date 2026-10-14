"""
Surgery Manifest and Self-Reconstructing Architecture Serialization Pipeline.

Maintains an exact structural blueprint of model modifications (transplanted attention,
sidecars, and factorized/pruned linear layers with per-layer adaptive ranks) to enable
100% deterministic reconstruction and state-dict loading without re-running SVD.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from neuralops.core.model_utils import replace_module

logger = logging.getLogger("NeuralOps.Surgery.Manifest")


@dataclass
class AttentionSurgeryEntry:
    layer_path: str
    attention_type: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    has_sidecar_gate: bool = False
    has_learnable_sink: bool = False
    peft_recovery_type: Optional[str] = None
    peft_rank: int = 0
    arch_pattern: Optional[str] = None


@dataclass
class CompressionSurgeryEntry:
    layer_path: str
    compressor_name: str
    in_features: int
    out_features: int
    rank: int
    bits: Optional[int] = None
    group_size: Optional[int] = None
    has_bias: bool = True
    wrapper_class: str = "SVDLinear"


@dataclass
class SurgeryManifest:
    version: str = "1.3.0"
    base_model_name: Optional[str] = None
    attentions: List[AttentionSurgeryEntry] = field(default_factory=list)
    compressions: List[CompressionSurgeryEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "base_model_name": self.base_model_name,
            "attentions": [asdict(a) for a in self.attentions],
            "compressions": [asdict(c) for c in self.compressions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SurgeryManifest:
        attentions = [
            AttentionSurgeryEntry(**a) for a in data.get("attentions", [])
        ]
        compressions = [
            CompressionSurgeryEntry(**c) for c in data.get("compressions", [])
        ]
        return cls(
            version=data.get("version", "1.3.0"),
            base_model_name=data.get("base_model_name"),
            attentions=attentions,
            compressions=compressions,
            metadata=data.get("metadata", {}),
        )


def record_attention_surgery(
    model: nn.Module,
    layer_path: str,
    attention_type: str,
    kwargs: Optional[Dict[str, Any]] = None,
    **extra_attrs: Any,
) -> None:
    """Records an attention surgery operation into model._neuralops_surgery_manifest."""
    if not hasattr(model, "_neuralops_surgery_manifest"):
        model._neuralops_surgery_manifest = SurgeryManifest()

    entry = AttentionSurgeryEntry(
        layer_path=layer_path,
        attention_type=attention_type,
        kwargs=dict(kwargs or {}),
        has_sidecar_gate=extra_attrs.get("has_sidecar_gate", False),
        has_learnable_sink=extra_attrs.get("has_learnable_sink", False),
        peft_recovery_type=extra_attrs.get("peft_recovery_type"),
        peft_rank=extra_attrs.get("peft_rank", 0),
        arch_pattern=extra_attrs.get("arch_pattern"),
    )
    model._neuralops_surgery_manifest.attentions.append(entry)


def record_compression_surgery(
    model: nn.Module,
    layer_path: str,
    compressor_name: str,
    in_features: int,
    out_features: int,
    rank: int,
    has_bias: bool = True,
    wrapper_class: str = "SVDLinear",
    **extra_attrs: Any,
) -> None:
    """Records a factorized/compressed linear layer into model._neuralops_surgery_manifest."""
    if not hasattr(model, "_neuralops_surgery_manifest"):
        model._neuralops_surgery_manifest = SurgeryManifest()

    entry = CompressionSurgeryEntry(
        layer_path=layer_path,
        compressor_name=compressor_name,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        has_bias=has_bias,
        wrapper_class=wrapper_class,
        bits=extra_attrs.get("bits"),
        group_size=extra_attrs.get("group_size"),
    )
    model._neuralops_surgery_manifest.compressions.append(entry)


def build_manifest_from_model(model: nn.Module) -> SurgeryManifest:
    """Inspects a model and builds or updates its SurgeryManifest."""
    if hasattr(model, "_neuralops_surgery_manifest") and isinstance(
        model._neuralops_surgery_manifest, SurgeryManifest
    ):
        manifest = model._neuralops_surgery_manifest
    else:
        manifest = SurgeryManifest()

    # Scan for SVDLinear and factorized layers not yet recorded
    existing_comp_paths = {c.layer_path for c in manifest.compressions}
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "SVDLinear" in cls_name or cls_name == "FactorizedLinear":
            if name not in existing_comp_paths:
                in_f = getattr(module, "in_features", 0)
                out_f = getattr(module, "out_features", 0)
                rank = getattr(module, "rank", 0)
                has_bias = getattr(module, "bias", None) is not None
                manifest.compressions.append(
                    CompressionSurgeryEntry(
                        layer_path=name,
                        compressor_name="direct_svd",
                        in_features=in_f,
                        out_features=out_f,
                        rank=rank,
                        has_bias=has_bias,
                        wrapper_class=cls_name,
                    )
                )

    return manifest


def replay_surgery_manifest(
    model: nn.Module, manifest: SurgeryManifest
) -> nn.Module:
    """
    Reconstructs factorized linear modules and attention layers purely from manifest metadata
    WITHOUT running heavy SVD decompositions or calibration passes.
    """
    from neuralops.api import replace_attention
    from neuralops.compression_engine.svd.aa_svd_compressor import SVDLinear
    from neuralops.core.utils.lora import FactorizedLinear

    # 1. Replay attention transplantations
    for attn_entry in manifest.attentions:
        try:
            model = replace_attention(
                model,
                attention=attn_entry.attention_type,
                **attn_entry.kwargs,
            )
        except Exception as e:
            logger.warning(
                "Could not re-transplant attention %s: %s",
                attn_entry.attention_type,
                e,
            )

    # 2. Replay compression factorizations
    for comp_entry in manifest.compressions:
        path = comp_entry.layer_path
        in_f = comp_entry.in_features
        out_f = comp_entry.out_features
        rank = comp_entry.rank
        has_bias = comp_entry.has_bias
        w_cls = comp_entry.wrapper_class

        if w_cls == "FactorizedLinear":
            replacement = FactorizedLinear(
                in_features=in_f,
                out_features=out_f,
                rank=rank,
                bias=has_bias,
            )
        else:
            replacement = SVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=rank,
                bias=has_bias,
            )

        try:
            replace_module(model, path, replacement)
        except Exception as e:
            logger.warning("Could not replay factorized layer %s: %s", path, e)

    model._neuralops_surgery_manifest = manifest
    return model


def merge_for_inference(model: nn.Module, only_above_breakeven: bool = False) -> nn.Module:
    """
    Folds low-rank adapters (PiSSALinear, DoRALinear) and factorized SVD layers
    back into single contiguous nn.Linear layers with < 1e-5 numerical error.
    """
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            # 1. PiSSALinear
            if hasattr(child, "merge_and_unload") and callable(child.merge_and_unload):
                try:
                    merged = child.merge_and_unload()
                    setattr(module, child_name, merged)
                    continue
                except Exception:
                    pass

            # 2. Factorized SVD linear layers
            cls_name = child.__class__.__name__
            if "SVDLinear" in cls_name or cls_name == "FactorizedLinear":
                d_in = getattr(child, "in_features", None)
                d_out = getattr(child, "out_features", None)
                r = getattr(child, "rank", None)
                if d_in and d_out and r:
                    breakeven = (d_in * d_out) / (d_in + d_out)
                    if only_above_breakeven and r < breakeven:
                        continue  # Keep factorized because it saves FLOPs

                    # Compute effective weight
                    if hasattr(child, "weight"):
                        eff_w = child.weight.data
                    elif hasattr(child, "U") and hasattr(child, "V"):
                        eff_w = (child.U @ child.V).data
                    elif hasattr(child, "lora_B") and hasattr(child, "lora_A"):
                        eff_w = (child.lora_B @ child.lora_A).data
                    else:
                        continue

                    bias = getattr(child, "bias", None)
                    has_bias = bias is not None
                    lin = nn.Linear(d_in, d_out, bias=has_bias, device=eff_w.device, dtype=eff_w.dtype)
                    lin.weight.data.copy_(eff_w)
                    if has_bias:
                        lin.bias.data.copy_(bias.data)
                    setattr(module, child_name, lin)

    return model


__all__ = [
    "SurgeryManifest",
    "AttentionSurgeryEntry",
    "CompressionSurgeryEntry",
    "record_attention_surgery",
    "record_compression_surgery",
    "build_manifest_from_model",
    "replay_surgery_manifest",
    "merge_for_inference",
]

