"""
CoMe: Compress LLMs through Layer Concatenation and Shedding with Learnable Inter-Layer Interpolation Adapters.
Registered in COMPRESSOR_REGISTRY as 'come'.

Mathematical Formulation:
1. Layer Shedding & Uniform Strided Selection:
   $$\\mathcal{I}_{\\text{keep}} = \\{i_0, i_1, \\dots, i_{K-1}\\}, \\quad i_k = \\left\\lfloor k \\cdot \\frac{L-1}{K-1} \\right\\rceil$$
2. Learnable Linear Interpolation Bridge between Non-Consecutive Layers:
   $$\\forall k \\text{ s.t. } i_{k+1} > i_k + 1: \\quad h_{k}^{\\text{bridged}} = h_k + \\alpha_k \\left( W_{\\text{bridge}}^{(k)} h_k - h_k \\right)$$
   $$W_{\\text{bridge}} \\approx I, \\quad \\alpha_k \\in \\mathbb{R} \\text{ (learnable mixing coefficient)}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


class CoMeBridge(nn.Module):
    """
    Learnable linear interpolation adapter bridging the representation gap between non-consecutive layers.
    Computes: y = x + alpha * (W_bridge @ x - x) with W_bridge initialized to Identity.
    """

    def __init__(self, hidden_size: int, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(hidden_states, (tuple, list)):
            x = hidden_states[0]
            if isinstance(x, torch.Tensor):
                bridged = x + self.alpha * (self.linear(x) - x)
                return (bridged, *hidden_states[1:])
            return hidden_states

        if isinstance(hidden_states, torch.Tensor):
            return hidden_states + self.alpha * (self.linear(hidden_states) - hidden_states)

        return hidden_states


class CoMeBridgedLayer(nn.Module):
    """
    Wraps a retained transformer layer and appends a learnable CoMeBridge when skipping layers.
    """

    def __init__(self, layer: nn.Module, bridge: Optional[CoMeBridge] = None) -> None:
        super().__init__()
        self.layer = layer
        self.bridge = bridge

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(*args, **kwargs)
        if self.bridge is not None:
            out = self.bridge(out)
        return out


class CoMeCompressor(BaseCompressor):
    """
    CoMe Compressor: Layer shedding with learnable linear interpolation adapters/bridges.

    Config parameters:
        target_layers : int, optional – target number of layers after shedding
        sparsity      : float, default 0.3 – fraction of layers to remove
        alpha_init    : float, default 0.1 – initial bridge interpolation weight
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        target_layers: Optional[int] = None,
        sparsity: float = 0.3,
        alpha_init: float = 0.1,
        recalibrate_ln: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="come", **kwargs)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.target_layers = target_layers
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.alpha_init = alpha_init
        self.recalibrate_ln = recalibrate_ln
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}

    def _get_hidden_size(self, layer: nn.Module) -> int:
        """Infers the hidden feature dimension from a transformer layer.

        Priority:
        1. Explicit attribute on the layer itself (hidden_size / d_model / dim / embed_dim).
        2. Same attributes on layer.config (HuggingFace style).
        3. Output-projection Linear whose out_features equals the model hidden dim
           (o_proj / out_proj / output / dense).  These project *back* to hidden_size.
        4. Majority-vote over all Linear.in_features – avoids MLP intermediate sizes
           (e.g. 2816 = up-proj intermediate) dominating when they appear first.
        5. Hard-coded default 768.
        """
        # 1. Direct attribute on the layer object
        for attr in ["hidden_size", "d_model", "dim", "embed_dim"]:
            if hasattr(layer, attr):
                return int(getattr(layer, attr))

        # 2. Via layer.config (HuggingFace TransformerLayer carries a config ref)
        if hasattr(layer, "config"):
            for attr in ["hidden_size", "d_model", "dim"]:
                if hasattr(layer.config, attr):
                    return int(getattr(layer.config, attr))

        # 3. Output-projection Linears project FROM all_head_size TO hidden_size;
        #    their out_features is the true hidden_size.
        _OUTPUT_PROJ_KEYS = ("o_proj", "out_proj", "output", "dense")
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear) and any(k in name for k in _OUTPUT_PROJ_KEYS):
                return m.out_features

        # 4. Majority vote over in_features of all Linear layers.
        #    MLP intermediate projections (e.g. gate_proj, up_proj) produce large
        #    outlier sizes; the hidden_size typically has the highest count.
        sizes = [m.in_features for _, m in layer.named_modules() if isinstance(m, nn.Linear)]
        if sizes:
            from collections import Counter
            return Counter(sizes).most_common(1)[0][0]

        # 5. LayerNorm fallback
        for m in layer.modules():
            if isinstance(m, nn.LayerNorm):
                ns = m.normalized_shape
                return ns[0] if isinstance(ns, (tuple, list)) else int(ns)

        return 768

    def _find_layer_container(self) -> Tuple[Optional[nn.Module], Optional[str], Optional[nn.ModuleList]]:
        """Finds the ModuleList containing transformer layers."""
        candidates = [
            ("transformer.layer", lambda m: getattr(getattr(m, "transformer", None), "layer", None)),
            ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
            ("transformer.blocks", lambda m: getattr(getattr(m, "transformer", None), "blocks", None)),
            ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
            ("model.decoder.layers", lambda m: getattr(getattr(getattr(m, "model", None), "decoder", None), "layers", None)),
            ("model.encoder.layers", lambda m: getattr(getattr(getattr(m, "model", None), "encoder", None), "layers", None)),
            ("encoder.layer", lambda m: getattr(getattr(m, "encoder", None), "layer", None)),
            ("decoder.layer", lambda m: getattr(getattr(m, "decoder", None), "layer", None)),
            ("layers", lambda m: getattr(m, "layers", None)),
            ("h", lambda m: getattr(m, "h", None)),
            ("blocks", lambda m: getattr(m, "blocks", None)),
        ]

        for name, getter in candidates:
            res = getter(self.model)
            if isinstance(res, (nn.ModuleList, list)) and len(res) > 0:
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                return parent, parts[-1], res

        # Generic search for any child ModuleList of repeated blocks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 2:
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                return parent, parts[-1], module

        return None, None, None

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Applies layer shedding with learnable inter-layer interpolation adapters."""
        parent, attr_name, layers = self._find_layer_container()
        bridges_inserted = 0

        if layers is not None and parent is not None and attr_name is not None:
            orig_len = len(layers)
            keep_len = max(1, int(orig_len * (1.0 - self.sparsity)))
            if self.target_layers is not None:
                keep_len = min(orig_len, max(1, self.target_layers))

            if keep_len < orig_len:
                # Uniformly sampled layer indices
                kept_indices = np.round(np.linspace(0, orig_len - 1, keep_len)).astype(int).tolist()
                # Deduplicate while preserving order
                seen = set()
                kept_indices = [x for x in kept_indices if not (x in seen or seen.add(x))]

                bridged_module_list = nn.ModuleList()
                for i, layer_idx in enumerate(kept_indices):
                    layer = layers[layer_idx]
                    # Check if jumping across dropped layers
                    is_gap = (i < len(kept_indices) - 1 and kept_indices[i + 1] > layer_idx + 1)
                    if is_gap:
                        hidden_size = self._get_hidden_size(layer)
                        p = next(layer.parameters(), None)
                        bridge = CoMeBridge(hidden_size=hidden_size, alpha_init=self.alpha_init)
                        if p is not None:
                            bridge = bridge.to(device=p.device, dtype=p.dtype)
                        bridged_module_list.append(CoMeBridgedLayer(layer, bridge=bridge))
                        bridges_inserted += 1
                    else:
                        bridged_module_list.append(layer)

                setattr(parent, attr_name, bridged_module_list)
        else:
            # SVD prune linear layers if architecture lacks explicit layer list
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    W = module.weight.data.float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    r = max(1, int(len(S) * (1.0 - self.sparsity)))
                    module.weight.data = ((U[:, :r] * S[:r]) @ Vt[:r, :]).to(dtype=module.weight.dtype)

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                device=self.target_device,
            )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "bridges_inserted": bridges_inserted,
            "target_layers": self.target_layers,
            "sparsity": self.sparsity,
            "recalibrate_ln": self.recalibrate_ln,
            "method": "come_layer_shedding_with_bridges",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("come")
def come_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for CoMe compressor."""
    target_layers = getattr(comp_config, "come_target_layers", kwargs.get("target_layers", None))
    sparsity = getattr(comp_config, "come_sparsity", kwargs.get("sparsity", 0.3))
    alpha_init = getattr(comp_config, "come_alpha_init", kwargs.get("alpha_init", 0.1))
    tokenizer = getattr(model, "tokenizer", kwargs.get("tokenizer", None))
    recalibrate_ln = getattr(comp_config, "come_recalibrate_ln", kwargs.get("recalibrate_ln", True))

    compressor = CoMeCompressor(
        model,
        tokenizer=tokenizer,
        target_layers=target_layers,
        sparsity=sparsity,
        alpha_init=alpha_init,
        recalibrate_ln=recalibrate_ln,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

__all__ = [
    "CoMeBridge",
    "CoMeBridgedLayer",
    "CoMeCompressor",
    "come_factory",
]
