"""
DRONE Compressor: Data-aware Low-Rank SVD with Identity Weighting.

Implements truncated SVD directly on target linear weight matrices (identity weighting)
and replaces target layers with structural SVDLinear modules.
Applies the identical rank allocation and Frobenius reconstruction error fallback threshold as AASVD/FWSVD.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear, get_dynamic_compression_rank
from framework.compression_engine.aa_svd_compressor import SVDLinear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("drone")
@COMPRESSOR_REGISTRY.register("drone_compressor")
class DRONECompressor(BaseCompressor):
    """
    DRONE Compressor.
    Data-aware SVD with identity weighting (standard data-independent SVD baseline)
    using identical rank adaptation, parameter constraints, and fallback error mechanism as AASVD.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        rank: int = 64,
        max_error: float = 0.85,
        fallback_to_original: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        target_modules: Optional[List[str]] = None,
        target_attention_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="drone", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "drone_rank", getattr(cfg, "rank", rank))
            max_error = getattr(cfg, "drone_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "drone_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )
            target_attention_only = getattr(
                cfg, "drone_target_attention_only", getattr(cfg, "target_attention_only", target_attention_only)
            )

        target_attention_only = kwargs.get(
            "target_attention_only", kwargs.get("drone_target_attention_only", target_attention_only)
        )
        self.rank = rank
        self.max_error = max_error
        self.fallback_to_original = fallback_to_original
        self.tokenizer = tokenizer
        self.target_modules = target_modules
        self.target_attention_only = target_attention_only
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, SVDLinear):
            return False
        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
            or "loralinear" in cls_name
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False
        name_lower = name.lower()
        if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score", "pooler"]):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj", "query", "key", "value"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress_linear(
        self,
        layer: nn.Module,
        rank: Optional[int] = None,
    ) -> Tuple[Optional[SVDLinear], float, int]:
        try:
            target_mod = getattr(layer, "base_layer", layer)
            cls_name = layer.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or "conv1d" in target_cls or hasattr(layer, "nf") or hasattr(target_mod, "nf")
            weight_tensor = getattr(target_mod, "weight", getattr(layer, "weight", None))
            if weight_tensor is None:
                return None, float("inf"), 0

            raw_w = weight_tensor.data

            if is_conv1d:
                in_feat, out_feat = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()
            else:
                out_feat, in_feat = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            r = get_dynamic_compression_rank(
                in_feat,
                out_feat,
                model_or_family=self.model,
                default_rank=rank or self.rank,
                comp_config=getattr(self, "comp_config", None),
            )
            max_allowed_rank = max(1, min(in_feat, out_feat, int(0.75 * (in_feat * out_feat) / (in_feat + out_feat))))
            r = min(r, max_allowed_rank)
            W_norm = torch.norm(W, p="fro").item()

            W_dev = W.to(device=layer.weight.device, dtype=torch.float32)
            U, S, Vt = torch.linalg.svd(W_dev, full_matrices=False)

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0))
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :])

            W_recon = torch.matmul(U_r, V_r)
            err = float(torch.norm(W - W_recon, p="fro").item() / max(W_norm, 1e-12))

            dtype = weight_tensor.dtype
            device = weight_tensor.device
            bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))

            svd_layer = SVDLinear(
                in_features=in_feat,
                out_features=out_feat,
                rank=r,
                bias=bias_tensor is not None,
                dtype=dtype,
                device=device,
                lora_r=getattr(self, "lora_r", 0),
                lora_alpha=getattr(self, "lora_alpha", 1.0),
            )
            svd_layer.set_low_rank_factors(
                U_r.to(device=device, dtype=dtype),
                V_r.to(device=device, dtype=dtype),
            )
            if bias_tensor is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(bias_tensor.data)

            svd_layer.reconstruction_error = err
            svd_layer.rank = r
            return svd_layer, err, r
        except Exception as e:
            logger.warning("DRONE factorization failed on linear: %s", e)
            return None, float("inf"), 0

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self.model.eval()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                svd_layer, err, r = self.compress_linear(module, rank=self.rank)
                self.layer_errors[name] = err
                self.layer_ranks[name] = r

                if svd_layer is None or err > self.max_error:
                    if self.fallback_to_original:
                        logger.warning(
                            "DRONE reconstruction error %.4f on %s exceeded threshold %.4f. Skipping compression (fallback).",
                            err,
                            name,
                            self.max_error,
                        )
                        self.fallback_layers.append(name)
                        continue
                    else:
                        if svd_layer is None:
                            self.fallback_layers.append(name)
                            continue

                if hasattr(module, "base_layer"):
                    module.base_layer = svd_layer
                else:
                    replace_module(self.model, name, svd_layer)
                self.compressed_layers.append(name)

        stats = {
            "compressed_layers": len(self.compressed_layers),
            "fallback_layers": len(self.fallback_layers),
            "avg_reconstruction_error": float(
                sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            ) if self.layer_errors else 0.0,
            "layer_ranks": self.layer_ranks,
        }
        self.model.compression_stats = stats
        return self.model, stats
