"""
CoMe: Compress LLMs through Layer Concatenation and Shedding with Learnable Inter-Layer Interpolation Adapters.
Registered in COMPRESSOR_REGISTRY as 'come' and 'come_compressor'.

Mathematical Formulation:
1. Layer Shedding & Uniform Strided Selection:
   $$\\mathcal{I}_{\\text{keep}} = \\{i_0, i_1, \\dots, i_{K-1}\\}, \\quad i_k = \\left\\lfloor k \\cdot \\frac{L-1}{K-1} \\right\\rceil$$
2. Learnable Linear Interpolation Bridge between Non-Consecutive Layers:
   $$\\forall k \\text{ s.t. } i_{k+1} > i_k + 1: \\quad h_{k}^{\\text{bridged}} = h_k + \\alpha_k \\left( W_{\\text{bridge}}^{(k)} h_k - h_k \\right)$$
   $$W_{\\text{bridge}} \\approx I, \\quad \\alpha_k \\in \\mathbb{R} \\text{ (learnable mixing coefficient)}$$
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from None, dict, or namespace object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CoMeBridge(nn.Module):
    """
    Learnable low-rank linear interpolation adapter bridging the representation gap between non-consecutive layers.
    Computes: y = x + alpha * (U(V(x)) - x) where V: hidden -> bridge_rank, U: bridge_rank -> hidden.
    Initialized such that U is N(0, 0.02^2) and V is 0, yielding identity at initialization.
    Robustly handles Tensor, tuple, list, dict, and BaseModelOutput returns.
    """

    def __init__(self, hidden_size: int, bridge_rank: int = 64, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bridge_rank = min(bridge_rank, hidden_size)
        self.V = nn.Linear(hidden_size, self.bridge_rank, bias=False)
        self.U = nn.Linear(self.bridge_rank, hidden_size, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.V.weight)
            nn.init.normal_(self.U.weight, std=0.02)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _bridge_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim >= 2 and t.shape[-1] == self.hidden_size:
            v_w = self.V.weight.to(device=t.device, dtype=t.dtype)
            u_w = self.U.weight.to(device=t.device, dtype=t.dtype)
            bottleneck = F.linear(t, v_w)
            recon = F.linear(bottleneck, u_w)
            return t + self.alpha.to(device=t.device, dtype=t.dtype) * (recon - t)
        return t

    def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        # Tuple return (common in PyTorch transformers: (hidden_states, ...))
        if isinstance(hidden_states, tuple):
            if len(hidden_states) > 0 and isinstance(hidden_states[0], torch.Tensor):
                return (self._bridge_tensor(hidden_states[0]), *hidden_states[1:])
            return hidden_states

        # List return
        if isinstance(hidden_states, list):
            if len(hidden_states) > 0 and isinstance(hidden_states[0], torch.Tensor):
                return [self._bridge_tensor(hidden_states[0])] + hidden_states[1:]
            return hidden_states

        # Dict return
        if isinstance(hidden_states, dict):
            out_dict = dict(hidden_states)
            for k in ["last_hidden_state", "hidden_states", "x"]:
                if k in out_dict and isinstance(out_dict[k], torch.Tensor):
                    out_dict[k] = self._bridge_tensor(out_dict[k])
                    return out_dict
            return hidden_states

        # BaseModelOutput or objects with last_hidden_state
        modified = False
        if hasattr(hidden_states, "last_hidden_state"):
            t = getattr(hidden_states, "last_hidden_state")
            if isinstance(t, torch.Tensor):
                setattr(hidden_states, "last_hidden_state", self._bridge_tensor(t))
                modified = True
                
        if hasattr(hidden_states, "hidden_states"):
            hs = getattr(hidden_states, "hidden_states")
            if isinstance(hs, tuple) and len(hs) > 0 and isinstance(hs[-1], torch.Tensor):
                new_hs = list(hs)
                new_hs[-1] = self._bridge_tensor(new_hs[-1])
                setattr(hidden_states, "hidden_states", tuple(new_hs))
                modified = True
                
        if modified:
            return hidden_states

        # Raw tensor return
        if isinstance(hidden_states, torch.Tensor):
            return self._bridge_tensor(hidden_states)

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
            if "layer" in self.__dict__ or ("_modules" in self.__dict__ and "layer" in self._modules):
                return getattr(self.layer, name)
            raise

    def __getitem__(self, key: Any) -> Any:
        return self.layer[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.layer

    def forward(
        self,
        hidden_states: Any = None,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        # Reconstruct args/kwargs to pass down safely without duplicate positional/keyword collision
        if hidden_states is None and "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        if hidden_states is not None:
            args = (hidden_states,) + args

        attn_mask = kwargs.pop("attention_mask", None)
        if attn_mask is not None:
            if getattr(attn_mask, "dtype", None) == torch.int64:
                attn_mask = attn_mask.to(torch.bool)
            kwargs["attention_mask"] = attn_mask
            
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
                torch.empty((self.lora_r, in_features), dtype=target_dtype, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=target_dtype, device=target_device)
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
        target_mod = getattr(linear, "base_layer", getattr(linear, "linear", linear))
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.detach()

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
            w = raw_w.t().contiguous().clone()
        else:
            out_f = raw_w.shape[0]
            in_f = raw_w.shape[1]
            w = raw_w.clone()

        bias = target_mod.bias.detach().clone() if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None else None
        return cls(
            in_features=in_f,
            out_features=out_f,
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
        out = F.linear(x_flat, self.weight.to(x_flat.dtype), self.bias.to(x_flat.dtype) if self.bias is not None else None)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.to(self.lora_A.dtype), self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


@COMPRESSOR_REGISTRY.register("come")
@COMPRESSOR_REGISTRY.register("come_compressor")
class CoMeCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    CoMe Compressor: Layer shedding with learnable low-rank linear interpolation adapters/bridges.

    Config parameters:
        target_layers : int, optional – target number of layers after shedding
        sparsity      : float, default 0.3 – fraction of layers to remove
        bridge_rank   : int, default 64 – bottleneck rank for inter-layer interpolation bridge (U @ V)
        alpha_init    : float, default 0.1 – initial bridge interpolation weight
        lora_r        : int, default 16 – rank for optional post-compression LoRA adapters
        lora_alpha    : float, default 1.0 – alpha scaling for LoRA adapters
        target_attention_only : bool, default False - only bridge/adapt attention projections
        allow_forward_failure : bool, default False - whether to suppress forward check failure
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        target_layers: Optional[int] = None,
        sparsity: float = 0.3,
        bridge_rank: int = 64,
        alpha_init: float = 0.1,
        lora_r: int = 16,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        protect_critical: bool = True,
        target_attention_only: bool = False,
        recalibrate_ln: bool = True,
        allow_forward_failure: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="come", **kwargs)
        conf = kwargs.get("comp_config", kwargs.get("cfg", None))

        target_layers = _cfg_get(conf, "come_target_layers", _cfg_get(conf, "target_layers", target_layers))
        if target_layers is not None:
            target_layers = int(target_layers)
        sparsity = float(_cfg_get(conf, "come_sparsity", _cfg_get(conf, "sparsity", sparsity)))
        bridge_rank = int(_cfg_get(conf, "come_bridge_rank", _cfg_get(conf, "bridge_rank", bridge_rank)))
        alpha_init = float(_cfg_get(conf, "come_alpha_init", _cfg_get(conf, "alpha_init", alpha_init)))
        lora_r = int(_cfg_get(conf, "come_lora_r", _cfg_get(conf, "lora_r", _cfg_get(conf, "lora_rank", lora_r))))
        lora_alpha = float(_cfg_get(conf, "come_lora_alpha", _cfg_get(conf, "lora_alpha", lora_alpha)))
        skip_attention_qkv = bool(_cfg_get(conf, "come_skip_attention_qkv", _cfg_get(conf, "skip_attention_qkv", skip_attention_qkv)))
        protect_critical = bool(_cfg_get(conf, "protect_critical", protect_critical))
        target_attention_only = bool(
            _cfg_get(conf, "come_target_attention_only", _cfg_get(conf, "target_attention_only", target_attention_only))
        )
        recalibrate_ln = bool(_cfg_get(conf, "come_recalibrate_ln", _cfg_get(conf, "recalibrate_ln", recalibrate_ln)))
        allow_forward_failure = bool(
            _cfg_get(conf, "allow_forward_failure", _cfg_get(kwargs, "allow_forward_failure", allow_forward_failure))
        )

        self.tokenizer = tokenizer or kwargs.get("tokenizer", getattr(model, "tokenizer", None))
        self.target_layers = target_layers
        self.sparsity = max(0.0, min(sparsity, 0.95))
        self.bridge_rank = bridge_rank
        self.alpha_init = alpha_init
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.skip_attention_qkv = skip_attention_qkv
        self.protect_critical = protect_critical
        self.target_attention_only = target_attention_only
        self.recalibrate_ln = recalibrate_ln
        self.allow_forward_failure = allow_forward_failure
        self.target_modules = target_modules
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.comp_config = conf

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.stats: Dict[str, Any] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        if self.target_attention_only:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, (CoMeLoRALinear, CoMeBridge, CoMeBridgedLayer)):
            return False

        if ".bridge." in name or name.endswith(".bridge"):
            return False

        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor", "head", "pre_classifier"]):
            return False

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only:
            ffn_patterns = ["ffn", "mlp", "lin1", "lin2", "intermediate", "gate_proj", "up_proj", "down_proj", "c_fc"]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value", "in_proj_qkv", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if self.protect_critical and not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _get_hidden_size(self, layer: nn.Module) -> int:
        """Infers hidden feature dimension from transformer layer."""
        if hasattr(self, "model") and hasattr(self.model, "config"):
            for attr in ["hidden_size", "d_model", "dim"]:
                if hasattr(self.model.config, attr):
                    return int(getattr(self.model.config, attr))

        for attr in ["hidden_size", "d_model", "dim", "embed_dim"]:
            if hasattr(layer, attr):
                return int(getattr(layer, attr))

        if hasattr(layer, "config"):
            for attr in ["hidden_size", "d_model", "dim"]:
                if hasattr(layer.config, attr):
                    return int(getattr(layer.config, attr))

        _OUTPUT_PROJ_KEYS = ("o_proj", "out_proj", "output", "dense")
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear) and any(k in name for k in _OUTPUT_PROJ_KEYS):
                return m.out_features

        sizes = [m.in_features for _, m in layer.named_modules() if isinstance(m, nn.Linear)]
        if sizes:
            from collections import Counter
            return Counter(sizes).most_common(1)[0][0]

        for m in layer.modules():
            if isinstance(m, nn.LayerNorm):
                ns = m.normalized_shape
                return ns[0] if isinstance(ns, (tuple, list)) else int(ns)

        raise ValueError(f"Cannot determine hidden_size for {type(layer).__name__}")

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
            ("model.encoder.block", lambda m: getattr(getattr(getattr(m, "model", None), "encoder", None), "block", None)),
            ("model.decoder.block", lambda m: getattr(getattr(getattr(m, "model", None), "decoder", None), "block", None)),
            ("encoder.block", lambda m: getattr(getattr(m, "encoder", None), "block", None)),
            ("decoder.block", lambda m: getattr(getattr(m, "decoder", None), "block", None)),
        ]

        for name, getter in candidates:
            res = getter(self.model)
            if isinstance(res, (nn.ModuleList, nn.Sequential, list)) and len(res) > 0:
                parts = name.split(".")
                parent = self.model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                return parent, parts[-1], res

        # Generic search prioritizing largest ModuleList named layer/h/block
        best_match = None
        max_score = -1
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) >= 2:
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
        parent, attr_name, layers = self._find_layer_container()
        bridges_inserted = 0
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []

        orig_len = 0
        keep_len = 0

        if layers is not None and parent is not None and attr_name is not None:
            orig_len = len(layers)
            keep_len = max(1, int(orig_len * (1.0 - self.sparsity)))
            if self.target_layers is not None:
                keep_len = min(orig_len, max(1, self.target_layers))

            if keep_len < orig_len:
                # Uniformly sampled layer indices
                kept_indices = np.round(np.linspace(0, orig_len - 1, keep_len)).astype(int).tolist()
                seen = set()
                kept_indices = [x for x in kept_indices if not (x in seen or seen.add(x))]

                bridged_module_list = nn.ModuleList()
                for i, layer_idx in enumerate(kept_indices):
                    layer = layers[layer_idx]
                    is_gap = (i < len(kept_indices) - 1 and kept_indices[i + 1] > layer_idx + 1)
                    if is_gap:
                        hidden_size = self._get_hidden_size(layer)
                        p = next(layer.parameters(), None)
                        bridge = CoMeBridge(
                            hidden_size=hidden_size,
                            bridge_rank=self.bridge_rank,
                            alpha_init=self.alpha_init,
                        )
                        if p is not None:
                            bridge = bridge.to(device=p.device, dtype=p.dtype)
                        bridged_module_list.append(CoMeBridgedLayer(layer, bridge=bridge))
                        bridges_inserted += 1
                    else:
                        bridged_module_list.append(layer)

                setattr(parent, attr_name, bridged_module_list)
                self.compressed_layers.append(f"{attr_name} (shed {orig_len - keep_len} layers)")
        else:
            # Fallback SVD compression if layer ModuleList is absent
            seen_weights: set[int] = set()
            for name, module in list(self.model.named_modules()):
                if self._is_target_module(name, module):
                    target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
                    w_id = id(target_mod.weight)
                    if w_id in seen_weights:
                        self.skipped_layers.append(name)
                        continue
                    seen_weights.add(w_id)

                    try:
                        raw_w = target_mod.weight.data
                        cls_name = module.__class__.__name__.lower()
                        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
                        W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
                        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                        r = max(1, int(len(S) * (1.0 - self.sparsity)))
                        from framework.compression_engine.aa_svd_compressor import SVDLinear
                        
                        in_features = W.shape[1]
                        out_features = W.shape[0]
                        svd_layer = SVDLinear(
                            in_features=in_features, 
                            out_features=out_features, 
                            rank=r, 
                            bias=getattr(target_mod, "bias", None) is not None
                        )
                        # SVDLinear expects U (out x r) and Vt (r x in)
                        svd_layer.set_low_rank_factors(U[:, :r] * S[:r], Vt[:r, :])
                        if getattr(target_mod, "bias", None) is not None:
                            svd_layer.bias.data.copy_(target_mod.bias.data)
                        
                        replace_module(self.model, name, svd_layer)
                        self.compressed_layers.append(name)
                    except Exception as e:
                        logger.warning("CoMe fallback SVD failed on %s: %s", name, e)
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False

        # Post-shedding LoRA adapter insertion on retained layers if lora_r > 0
        if self.lora_r > 0:
            seen_weights: set[int] = set()
            for name, module in list(self.model.named_modules()):
                if self._is_target_module(name, module):
                    target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
                    has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")
                    w_id = id(target_mod.weight)
                    if w_id in seen_weights:
                        self.skipped_layers.append(name)
                        continue
                    seen_weights.add(w_id)

                    if has_lora:
                        # Existing LoRALinear: freeze base layer, keep LoRA parameters trainable
                        target_mod.weight.requires_grad = False
                        if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
                            target_mod.bias.requires_grad = False
                        for p_name, p in module.named_parameters():
                            if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                                p.requires_grad = True
                    else:
                        lora_mod = CoMeLoRALinear.from_linear(
                            target_mod,
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

        # Post-compression forward sanity check (fail loud if allow_forward_failure=False)
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.error("CoMe post-compression forward sanity check failed: %s", e)
            if not self.allow_forward_failure:
                raise RuntimeError(f"CoMe post-compression forward check failed: {e}") from e

        self._compressed_params = self._count_params(self.model)
        shed_count = orig_len - keep_len if orig_len > 0 else 0
        reduc_ratio = float(shed_count) / orig_len if orig_len > 0 else self.sparsity
        flops_reduction = f"{reduc_ratio * 100:.1f}%"

        bridge_params_count = sum(
            p.numel() for m in self.model.modules() if isinstance(m, CoMeBridge) for p in m.parameters()
        )
        lora_params_count = sum(
            p.numel() for m in self.model.modules() if isinstance(m, CoMeLoRALinear) for p in [m.lora_A, m.lora_B] if p is not None
        )

        stats = self.get_compression_stats()
        stats.update({
            "original_layer_count": orig_len,
            "kept_layer_count": keep_len,
            "shed_layer_count": shed_count,
            "layer_reduction_ratio": reduc_ratio,
            "estimated_flops_reduction": flops_reduction,
            "bridges_inserted": bridges_inserted,
            "bridge_rank": self.bridge_rank,
            "bridge_parameters": bridge_params_count,
            "lora_parameters": lora_params_count,
            "trainable_parameters": sum(p.numel() for p in self.get_trainable_params()),
            "target_layers": self.target_layers,
            "sparsity": self.sparsity,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_attention_only": self.target_attention_only,
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
        self.model.compression_stats = stats
        return self.model, stats

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns list of all trainable parameters (bridges + LoRA adapters)."""
        trainable = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        return trainable


@COMPRESSOR_REGISTRY.register("come_factory")
@COMPRESSOR_REGISTRY.register("come_compressor_factory")
def come_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for CoMe compressor."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    target_layers = kwargs.pop(
        "target_layers",
        _cfg_get(comp_config, "come_target_layers", _cfg_get(comp_config, "target_layers", None)),
    )
    if target_layers is not None:
        target_layers = int(target_layers)
    sparsity = float(
        kwargs.pop(
            "sparsity",
            _cfg_get(comp_config, "come_sparsity", _cfg_get(comp_config, "sparsity", 0.3)),
        )
    )
    bridge_rank = int(
        kwargs.pop(
            "bridge_rank",
            _cfg_get(comp_config, "come_bridge_rank", _cfg_get(comp_config, "bridge_rank", 64)),
        )
    )
    alpha_init = float(
        kwargs.pop(
            "alpha_init",
            _cfg_get(comp_config, "come_alpha_init", _cfg_get(comp_config, "alpha_init", 0.1)),
        )
    )
    lora_r = int(
        kwargs.pop(
            "lora_r",
            _cfg_get(comp_config, "come_lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 16))),
        )
    )
    lora_alpha = float(
        kwargs.pop(
            "lora_alpha",
            _cfg_get(comp_config, "come_lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0)),
        )
    )
    skip_attention_qkv = bool(
        kwargs.pop(
            "skip_attention_qkv",
            _cfg_get(comp_config, "come_skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", True)),
        )
    )
    protect_critical = bool(
        kwargs.pop(
            "protect_critical",
            _cfg_get(comp_config, "protect_critical", True),
        )
    )
    target_attention_only = bool(
        kwargs.pop(
            "target_attention_only",
            _cfg_get(comp_config, "come_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
        )
    )
    tokenizer = getattr(model, "tokenizer", kwargs.pop("tokenizer", None))
    recalibrate_ln = bool(
        kwargs.pop(
            "recalibrate_ln",
            _cfg_get(comp_config, "come_recalibrate_ln", _cfg_get(comp_config, "recalibrate_ln", True)),
        )
    )
    allow_forward_failure = bool(
        kwargs.pop(
            "allow_forward_failure",
            _cfg_get(comp_config, "allow_forward_failure", False),
        )
    )

    compressor = CoMeCompressor(
        model,
        tokenizer=tokenizer,
        target_layers=target_layers,
        sparsity=sparsity,
        bridge_rank=bridge_rank,
        alpha_init=alpha_init,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        protect_critical=protect_critical,
        target_attention_only=target_attention_only,
        recalibrate_ln=recalibrate_ln,
        allow_forward_failure=allow_forward_failure,
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
