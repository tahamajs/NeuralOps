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
    Delegates attribute lookups to the underlying transformer layer.
    """

    def __init__(self, layer: nn.Module, bridge: Optional[CoMeBridge] = None) -> None:
        super().__init__()
        self.layer = layer
        self.bridge = bridge

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def __getitem__(self, key: Any) -> Any:
        return self.layer[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.layer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.layer(*args, **kwargs)
        if self.bridge is not None:
            out = self.bridge(out)
        return out


class CoMeLoRALinear(nn.Module):
    """
    Linear layer containing frozen base weights with optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        self.weight = nn.Parameter(
            weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> CoMeLoRALinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.detach()
        w = raw_w.t().clone() if is_conv1d else raw_w.clone()
        bias = linear.bias.detach() if hasattr(linear, "bias") and linear.bias is not None else None
        in_features = getattr(linear, "in_features", w.shape[1])
        out_features = getattr(linear, "out_features", w.shape[0])
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight=w,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=w.device,
            dtype=w.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(x_flat, self.weight, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


class CoMeCompressor(BaseCompressor):
    """
    CoMe Compressor: Layer shedding with learnable linear interpolation adapters/bridges.

    Config parameters:
        target_layers : int, optional – target number of layers after shedding
        sparsity      : float, default 0.3 – fraction of layers to remove
        alpha_init    : float, default 0.1 – initial bridge interpolation weight
        lora_r        : int, default 0 – rank for optional post-compression LoRA adapters
        lora_alpha    : float, default 1.0 – alpha scaling for LoRA adapters
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        target_layers: Optional[int] = None,
        sparsity: float = 0.3,
        alpha_init: float = 0.1,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        protect_critical: bool = True,
        recalibrate_ln: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="come", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        if conf is not None:
            target_layers = getattr(conf, "come_target_layers", getattr(conf, "target_layers", target_layers))
            sparsity = getattr(conf, "come_sparsity", getattr(conf, "sparsity", sparsity))
            alpha_init = getattr(conf, "come_alpha_init", getattr(conf, "alpha_init", alpha_init))
            lora_r = getattr(conf, "lora_r", getattr(conf, "lora_rank", lora_r))
            lora_alpha = getattr(conf, "lora_alpha", lora_alpha)
            skip_attention_qkv = getattr(conf, "skip_attention_qkv", skip_attention_qkv)
            protect_critical = getattr(conf, "protect_critical", protect_critical)
            recalibrate_ln = getattr(conf, "come_recalibrate_ln", getattr(conf, "recalibrate_ln", recalibrate_ln))

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.target_layers = target_layers
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.alpha_init = float(alpha_init)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.protect_critical = bool(protect_critical)
        self.recalibrate_ln = bool(recalibrate_ln)
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.stats: Dict[str, Any] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _get_hidden_size(self, layer: nn.Module) -> int:
        """Infers the hidden feature dimension from a transformer layer."""
        # 0. Via self.model.config (top-level model config is canonical)
        if hasattr(self, "model") and hasattr(self.model, "config"):
            for attr in ["hidden_size", "d_model", "dim"]:
                if hasattr(self.model.config, attr):
                    return int(getattr(self.model.config, attr))

        # 1. Direct attribute on the layer object
        for attr in ["hidden_size", "d_model", "dim", "embed_dim"]:
            if hasattr(layer, attr):
                return int(getattr(layer, attr))

        # 2. Via layer.config (HuggingFace TransformerLayer carries a config ref)
        if hasattr(layer, "config"):
            for attr in ["hidden_size", "d_model", "dim"]:
                if hasattr(layer.config, attr):
                    return int(getattr(layer.config, attr))

        # 3. Output-projection Linears project FROM all_head_size TO hidden_size
        _OUTPUT_PROJ_KEYS = ("o_proj", "out_proj", "output", "dense")
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear) and any(k in name for k in _OUTPUT_PROJ_KEYS):
                return m.out_features

        # 4. Majority vote over in_features of all Linear layers
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
        """Finds the ModuleList containing transformer layers across diverse architectures."""
        candidates = [
            ("roberta.encoder.layer", lambda m: getattr(getattr(getattr(m, "roberta", None), "encoder", None), "layer", None)),
            ("bert.encoder.layer", lambda m: getattr(getattr(getattr(m, "bert", None), "encoder", None), "layer", None)),
            ("distilbert.transformer.layer", lambda m: getattr(getattr(getattr(m, "distilbert", None), "transformer", None), "layer", None)),
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

        # Generic search prioritizing largest ModuleList named layer/h/block
        best_match = None
        max_score = -1
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) >= 2:
                parts = name.split(".")
                last = parts[-1].lower()
                is_layer_name = any(k in last for k in ["layer", "h", "block"])
                score = len(module) + (1000 if is_layer_name else 0)
                if score > max_score:
                    max_score = score
                    parent = self.model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    best_match = (parent, parts[-1], module)

        if best_match is not None:
            return best_match

        return None, None, None

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Applies layer shedding with learnable inter-layer interpolation adapters."""
        from framework.utils.compat import should_compress_linear
        from framework.core.model_utils import replace_module

        parent, attr_name, layers = self._find_layer_container()
        bridges_inserted = 0
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []

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
                self.compressed_layers.append(f"{attr_name} (shed {orig_len - keep_len} layers)")
        else:
            # Fallback: SVD compress linear layers if architecture lacks explicit layer list
            seen_weights: set[int] = set()
            for name, module in list(self.model.named_modules()):
                cls_name = module.__class__.__name__.lower()
                is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
                if is_linear_like and not isinstance(module, CoMeLoRALinear):
                    if self.protect_critical and any(skip in name.lower() for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
                        continue
                    if not should_compress_linear(self.model, name):
                        continue
                    if self._is_gqa_asymmetric(name, module):
                        continue
                    if not hasattr(module, "weight") or module.weight is None:
                        continue
                    w_id = id(module.weight)
                    if w_id in seen_weights:
                        self.skipped_layers.append(name)
                        continue
                    seen_weights.add(w_id)

                    try:
                        raw_w = module.weight.data
                        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                        W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                        r = max(1, int(len(S) * (1.0 - self.sparsity)))
                        W_comp = (U[:, :r] * S[:r]) @ Vt[:r, :]
                        if is_conv1d:
                            module.weight.data.copy_(W_comp.t().to(dtype=module.weight.dtype, device=raw_w.device))
                        else:
                            module.weight.data.copy_(W_comp.to(dtype=module.weight.dtype, device=raw_w.device))
                        self.compressed_layers.append(name)
                    except Exception as e:
                        logger.warning("CoMe fallback SVD failed on %s: %s", name, e)
                        self.fallback_layers.append(name)

        # Post-shedding trainable LoRA adapter insertion on retained layers if lora_r > 0
        if self.lora_r > 0:
            seen_weights: set[int] = set()
            for name, module in list(self.model.named_modules()):
                cls_name = module.__class__.__name__.lower()
                is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
                if is_linear_like and not isinstance(module, (CoMeLoRALinear, CoMeBridge)):
                    if self.protect_critical and any(skip in name.lower() for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
                        continue
                    if not should_compress_linear(self.model, name):
                        continue
                    if self._is_gqa_asymmetric(name, module):
                        continue
                    if not hasattr(module, "weight") or module.weight is None:
                        continue
                    w_id = id(module.weight)
                    if w_id in seen_weights:
                        self.skipped_layers.append(name)
                        continue
                    seen_weights.add(w_id)

                    lora_mod = CoMeLoRALinear.from_linear(
                        module,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    replace_module(self.model, name, lora_mod)

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            try:
                recalibrate_layernorms(
                    self.model,
                    tokenizer=self.tokenizer,
                    device=self.target_device,
                )
            except Exception as e:
                logger.warning("CoMe LayerNorm recalibration skipped: %s", e)

        # Post-compression forward sanity check
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.debug("CoMe post-compression forward sanity check exception: %s", e)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "bridges_inserted": bridges_inserted,
            "target_layers": self.target_layers,
            "sparsity": self.sparsity,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "recalibrate_ln": self.recalibrate_ln,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "forward_check_success": forward_check_success,
            "method": "come_layer_shedding_with_bridges",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("come")
@COMPRESSOR_REGISTRY.register("come_compressor")
@COMPRESSOR_REGISTRY.register("come_factory")
@COMPRESSOR_REGISTRY.register("come_compressor_factory")
def come_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for CoMe compressor."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    target_layers = kwargs.pop("target_layers", getattr(comp_config, "come_target_layers", getattr(comp_config, "target_layers", None)) if comp_config else None)
    sparsity = kwargs.pop("sparsity", getattr(comp_config, "come_sparsity", getattr(comp_config, "sparsity", 0.3)) if comp_config else 0.3)
    alpha_init = kwargs.pop("alpha_init", getattr(comp_config, "come_alpha_init", getattr(comp_config, "alpha_init", 0.1)) if comp_config else 0.1)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", getattr(comp_config, "lora_rank", 0)) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    skip_attention_qkv = kwargs.pop("skip_attention_qkv", getattr(comp_config, "skip_attention_qkv", True) if comp_config else True)
    protect_critical = kwargs.pop("protect_critical", getattr(comp_config, "protect_critical", True) if comp_config else True)
    tokenizer = getattr(model, "tokenizer", kwargs.pop("tokenizer", None))
    recalibrate_ln = getattr(comp_config, "come_recalibrate_ln", kwargs.get("recalibrate_ln", True)) if comp_config else kwargs.get("recalibrate_ln", True)

    compressor = CoMeCompressor(
        model,
        tokenizer=tokenizer,
        target_layers=target_layers,
        sparsity=sparsity,
        alpha_init=alpha_init,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        protect_critical=protect_critical,
        recalibrate_ln=recalibrate_ln,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "CoMeBridge",
    "CoMeBridgedLayer",
    "CoMeLoRALinear",
    "CoMeCompressor",
    "come_factory",
]
