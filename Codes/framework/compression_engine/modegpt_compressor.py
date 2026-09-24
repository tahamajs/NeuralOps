"""
MoDeGPT: Modular Decomposition of Transformer Blocks.
Decomposes pairs of linear projections (e.g. Q & K, Up & Gate) jointly via shared low-rank bases.
Registered as 'modegpt'.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class MoDeGPTLoRALinear(nn.Module):
    is_custom_attention: bool = True
    """
    Linear layer containing MoDeGPT decomposed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
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
        weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> MoDeGPTLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        bias = base_layer.bias.detach() if hasattr(base_layer, "bias") and base_layer.bias is not None else None
        in_features = getattr(base_layer, "in_features", weight.shape[1])
        out_features = getattr(base_layer, "out_features", weight.shape[0])

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

        # Preserve existing LoRA weights if present
        if hasattr(linear, "lora_A") and hasattr(linear, "lora_B") and layer.lora_A is not None and layer.lora_B is not None:
            with torch.no_grad():
                src_A = linear.lora_A.weight.data if hasattr(linear.lora_A, "weight") else linear.lora_A.data
                src_B = linear.lora_B.weight.data if hasattr(linear.lora_B, "weight") else linear.lora_B.data
                if src_A.shape == layer.lora_A.shape and src_B.shape == layer.lora_B.shape:
                    layer.lora_A.copy_(src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype))
                    layer.lora_B.copy_(src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype))

        if layer.lora_A is not None:
            layer.lora_A.requires_grad = True
        if layer.lora_B is not None:
            layer.lora_B.requires_grad = True

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)
        return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin", "self_attn"
}


@COMPRESSOR_REGISTRY.register("modegpt")
@COMPRESSOR_REGISTRY.register("modegpt_compressor")
class MoDeGPTCompressor(BaseCompressor):
    """
    MoDeGPT: Modular Decomposition of Transformer Projection Blocks.
    Decomposes paired linear projections (e.g. Q & K, Gate & Up) jointly into shared low-rank subspaces:
        [W_1; W_2] = U_r S_r V_r^T + E

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.50).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection and support across GPT-2 architectures.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping: avoids corrupting asymmetric attention projections in GQA models.
    - Tied / shared weight protection tracking parameter IDs.
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        method: str = "svd",
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        min_rank: int = 4,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="modegpt", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "modegpt_rank", _cfg_get(cfg, "rank", rank))
            method = _cfg_get(cfg, "modegpt_method", _cfg_get(cfg, "method", method))
            max_error = _cfg_get(cfg, "modegpt_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "modegpt_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.rank = max(1, int(rank))
        self.method = str(method).lower()
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.min_rank = max(1, int(min_rank))
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
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

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only:
            is_attn = any(k in name.lower() for k in _ATTN_NAMES)
            if not is_attn:
                return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _get_weight(self, module: nn.Module) -> Tuple[torch.Tensor, bool]:
        target_layer = getattr(module, "base_layer", module)
        raw_w = target_layer.weight.data
        cls_name = target_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
        W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
        return W, is_conv1d

    def _decompose_joint_pair(
        self,
        name1: str,
        mod1: nn.Module,
        name2: str,
        mod2: nn.Module,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """Jointly decompose two linear layer weights sharing the same input dimension."""
        W1, is_conv1 = self._get_weight(mod1)
        W2, is_conv2 = self._get_weight(mod2)

        if W1.shape[1] != W2.shape[1]:
            return None, None, 0

        # Concatenate along output dimension: (out1 + out2, in)
        W_joint = torch.cat([W1, W2], dim=0)
        max_r = min(W_joint.shape)
        r = max(min(self.min_rank, max_r), min(self.rank, max_r))

        U, S, Vt = torch.linalg.svd(W_joint, full_matrices=False)
        r = min(r, len(S))
        U_r = U[:, :r] * S[:r].unsqueeze(0)
        Vt_r = Vt[:r, :]
        W_approx = torch.nan_to_num(U_r @ Vt_r, nan=0.0)

        out1 = W1.shape[0]
        W1_approx = W_approx[:out1, :]
        W2_approx = W_approx[out1:, :]
        return W1_approx, W2_approx, r

    def _decompose_single(
        self,
        module: nn.Module,
    ) -> Tuple[torch.Tensor, int]:
        """Decompose a single linear or Conv1D layer."""
        W, _ = self._get_weight(module)
        max_r = min(W.shape)
        r = max(min(self.min_rank, max_r), min(self.rank, max_r))

        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        r = min(r, len(S))
        U_r = U[:, :r] * S[:r].unsqueeze(0)
        Vt_r = Vt[:r, :]
        W_approx = torch.nan_to_num(U_r @ Vt_r, nan=0.0)
        return W_approx, r

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Identify paired linear layers in transformer blocks and apply joint modular decomposition."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        seen_weights: Set[int] = set()
        processed_names: Set[str] = set()

        total_orig_params = sum(p.numel() for p in self.model.parameters())
        total_comp_params = 0

        # Step 1: Search for named pairs in submodules
        for parent_name, parent_mod in list(self.model.named_modules()):
            pair_candidates = [
                ("q_lin", "k_lin"),
                ("q_lin", "v_lin"),
                ("q_proj", "k_proj"),
                ("q_proj", "v_proj"),
                ("query", "key"),
                ("W_q", "W_k"),
                ("gate_proj", "up_proj"),
            ]
            for attr1, attr2 in pair_candidates:
                if hasattr(parent_mod, attr1) and hasattr(parent_mod, attr2):
                    mod1 = getattr(parent_mod, attr1)
                    mod2 = getattr(parent_mod, attr2)
                    name1 = f"{parent_name}.{attr1}" if parent_name else attr1
                    name2 = f"{parent_name}.{attr2}" if parent_name else attr2

                    if name1 in processed_names or name2 in processed_names:
                        continue

                    if not (self._is_target_module(name1, mod1) and self._is_target_module(name2, mod2)):
                        continue

                    target_l1 = getattr(mod1, "base_layer", mod1)
                    target_l2 = getattr(mod2, "base_layer", mod2)
                    if not (hasattr(target_l1, "weight") and hasattr(target_l2, "weight")):
                        continue

                    w1_id, w2_id = id(target_l1.weight), id(target_l2.weight)
                    if w1_id in seen_weights or w2_id in seen_weights:
                        self.skipped_layers.extend([name1, name2])
                        continue
                    seen_weights.add(w1_id)
                    seen_weights.add(w2_id)

                    try:
                        W1_approx, W2_approx, r = self._decompose_joint_pair(name1, mod1, name2, mod2)
                        if W1_approx is None or W2_approx is None:
                            continue

                        W1_orig, is_conv1 = self._get_weight(mod1)
                        W2_orig, is_conv2 = self._get_weight(mod2)

                        err1 = (torch.norm(W1_orig - W1_approx) / max(torch.norm(W1_orig).item(), 1e-12)).item()
                        err2 = (torch.norm(W2_orig - W2_approx) / max(torch.norm(W2_orig).item(), 1e-12)).item()

                        self.layer_errors[name1] = float(err1)
                        self.layer_errors[name2] = float(err2)
                        self.layer_ranks[name1] = r
                        self.layer_ranks[name2] = r

                        # Check error on pair
                        if err1 > self.max_error or err2 > self.max_error or not math.isfinite(err1) or not math.isfinite(err2):
                            logger.warning(
                                "MoDeGPT joint pair error (%.4f, %.4f) exceeded max_error %.4f on (%s, %s).",
                                err1, err2, self.max_error, name1, name2,
                            )
                            if self.fallback_to_original:
                                self.fallback_layers.extend([name1, name2])
                                processed_names.add(name1)
                                processed_names.add(name2)
                                for m in [mod1, mod2]:
                                    for p in m.parameters(recurse=False):
                                        p.requires_grad = False
                                        total_comp_params += p.numel()
                                continue

                        w1_rep = W1_approx
                        w2_rep = W2_approx

                        layer1 = MoDeGPTLoRALinear.from_linear(
                            mod1,
                            weight=w1_rep.to(dtype=target_l1.weight.dtype, device=target_l1.weight.device),
                            lora_r=self.lora_r,
                            lora_alpha=self.lora_alpha,
                        )
                        layer2 = MoDeGPTLoRALinear.from_linear(
                            mod2,
                            weight=w2_rep.to(dtype=target_l2.weight.dtype, device=target_l2.weight.device),
                            lora_r=self.lora_r,
                            lora_alpha=self.lora_alpha,
                        )
                        replace_module(self.model, name1, layer1)
                        replace_module(self.model, name2, layer2)
                        self.compressed_layers.extend([name1, name2])
                        processed_names.add(name1)
                        processed_names.add(name2)
                        for l in [layer1, layer2]:
                            for p in l.parameters(recurse=False):
                                total_comp_params += p.numel()
                    except Exception as e:
                        logger.warning("MoDeGPT joint pair decomposition exception on (%s, %s): %s", name1, name2, e)
                        self.fallback_layers.extend([name1, name2])
                        processed_names.add(name1)
                        processed_names.add(name2)
                        for m in [mod1, mod2]:
                            for p in m.parameters(recurse=False):
                                p.requires_grad = False
                                total_comp_params += p.numel()

        # Step 2: Decompose remaining individual layers
        for name, module in list(self.model.named_modules()):
            if name in processed_names:
                continue
            if not self._is_target_module(name, module):
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue

            target_layer = getattr(module, "base_layer", module)
            if not hasattr(target_layer, "weight") or target_layer.weight is None:
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue

            w_id = id(target_layer.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()
                continue
            seen_weights.add(w_id)

            try:
                W_orig, is_conv = self._get_weight(module)
                W_approx, r = self._decompose_single(module)
                err = (torch.norm(W_orig - W_approx) / max(torch.norm(W_orig).item(), 1e-12)).item()
                self.layer_errors[name] = float(err)
                self.layer_ranks[name] = r

                if not math.isfinite(err) or err > self.max_error:
                    logger.warning(
                        "MoDeGPT reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err, name, self.max_error,
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters(recurse=False):
                            p.requires_grad = False
                            total_comp_params += p.numel()
                        continue

                w_rep = W_approx
                new_layer = MoDeGPTLoRALinear.from_linear(
                    module,
                    weight=w_rep.to(dtype=target_layer.weight.dtype, device=target_layer.weight.device),
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
                processed_names.add(name)
                for p in new_layer.parameters(recurse=False):
                    total_comp_params += p.numel()
            except Exception as e:
                logger.warning("MoDeGPT decomposition exception on %s: %s", name, e)
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                for p in module.parameters(recurse=False):
                    p.requires_grad = False
                    total_comp_params += p.numel()

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    self.model(dummy.float())
        except Exception as e:
            logger.warning("MoDeGPT post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        eff_ratio = total_comp_params / max(1, total_orig_params)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "effective_parameter_ratio": eff_ratio,
            "rank": self.rank,
            "method": self.method,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("modegpt_factory")
def modegpt_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """MoDeGPT compressor factory function."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "modegpt_rank", _cfg_get(comp_config, "rank", 64)))
    method = kwargs.pop("method", _cfg_get(comp_config, "modegpt_method", _cfg_get(comp_config, "method", "svd")))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "modegpt_max_error", _cfg_get(comp_config, "max_error", 0.50)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(
        comp_config, "modegpt_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    ))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = MoDeGPTCompressor(
        model,
        rank=rank,
        method=method,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "MoDeGPTLoRALinear",
    "MoDeGPTCompressor",
    "modegpt_factory",
]
