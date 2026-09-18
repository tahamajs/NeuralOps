"""
Subspace Projection Compressor: Low-Rank Subspace Projection and Factorization.
Registered in COMPRESSOR_REGISTRY as 'subspace_projection'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class SubspaceLinear(nn.Module):
    is_custom_attention: bool = True
    """
    Subspace Projection Factorized Linear Layer with optional LoRA adapter support.
    W ≈ B @ A where:
    - A in R^{rank x in_features}: orthonormal / projection basis
    - B in R^{out_features x rank}: coefficient expansion matrix

    Forward pass:
    y = ((x @ A^T) @ B^T) + bias + (lora_B @ lora_A @ x) * scaling
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        self.A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        if dtype in (torch.float16, torch.bfloat16):
            A_fp32 = torch.empty(self.rank, in_features, dtype=torch.float32)
            nn.init.orthogonal_(A_fp32)
            self.A.data.copy_(A_fp32.to(dtype=dtype, device=device))
        else:
            nn.init.orthogonal_(self.A)
        nn.init.xavier_uniform_(self.B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, in_features), dtype=torch.float32, device=device))
            self.lora_B = nn.Parameter(torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.B, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # x @ A^T: (K, rank)
        x_proj = F.linear(x_flat, self.A)
        # x_proj @ B^T + bias: (K, out_features)
        out = F.linear(x_proj, self.B, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out.to(out.dtype)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full weight matrix W = B @ A."""
        return torch.matmul(self.B, self.A)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int = 64,
        method: str = "svd",
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "SubspaceLinear":
        """
        Project linear or Conv1D layer weight matrix onto top-rank subspace.
        """
        base_mod = getattr(linear, "base_layer", linear)
        raw_w = base_mod.weight.data
        cls_name = base_mod.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf") or (
            raw_w.ndim == 2
            and hasattr(base_mod, "in_features")
            and base_mod.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().contiguous()
            out_f, in_f = W.shape
        else:
            W = raw_w
            out_f, in_f = W.shape

        bias = base_mod.bias.data if getattr(base_mod, "bias", None) is not None else None
        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1) if (in_f * out_f > 16) else max_r
        r = max(1, min(rank, max_r, param_cap))

        has_existing_lora = hasattr(linear, "lora_A") and hasattr(linear, "lora_B")
        effective_lora_r = lora_r
        if has_existing_lora and getattr(linear, "r", 0) > 0:
            effective_lora_r = getattr(linear, "r", lora_r)

        subspace_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            lora_r=effective_lora_r if (use_lora or has_existing_lora) else 0,
            lora_alpha=lora_alpha,
            dtype=W.dtype,
            device=W.device,
        )

        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        A_data = Vt[:r, :]
        B_data = U[:, :r] * S[:r].unsqueeze(0)

        subspace_layer.A.data.copy_(A_data.to(dtype=W.dtype))
        subspace_layer.B.data.copy_(B_data.to(dtype=W.dtype))

        if bias is not None and subspace_layer.bias is not None:
            subspace_layer.bias.data.copy_(bias)

        if has_existing_lora and subspace_layer.lora_r > 0:
            try:
                if hasattr(linear.lora_A, "weight") and subspace_layer.lora_A is not None:
                    subspace_layer.lora_A.data.copy_(linear.lora_A.weight.data)
                if hasattr(linear.lora_B, "weight") and subspace_layer.lora_B is not None:
                    subspace_layer.lora_B.data.copy_(linear.lora_B.weight.data)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        return subspace_layer


class SubspaceProjectionCompressor(BaseCompressor):
    """
    Subspace Projection Compression Engine.
    Compresses neural networks by projecting weight matrices onto their principal
    subspaces, replacing dense linear layers with factorized SubspaceLinear modules.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        method: str = "svd",
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="subspace_projection", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config", None)
        self.rank = rank or _cfg_get(cfg_obj, "subspace_rank", _cfg_get(cfg_obj, "rank", 64))
        self.method = method or _cfg_get(cfg_obj, "subspace_method", "svd")
        self.cfg = cfg_obj
        self.target_layers = target_layers or _cfg_get(cfg_obj, "target_layers", None)
        self.target_attention_only = bool(target_attention_only if target_attention_only is not None else _cfg_get(cfg_obj, "target_attention_only", False))
        self.use_lora = bool(use_lora or _cfg_get(cfg_obj, "use_lora", False) or (lora_r > 0))
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.max_error = float(_cfg_get(cfg_obj, "max_error", max_error))
        self.fallback_to_original = bool(_cfg_get(cfg_obj, "fallback_to_original", fallback_to_original))
        self.compressed_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        if self.target_attention_only:
            attn_keywords = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj', 'query', 'key', 'value']
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        is_linear_like = isinstance(base_mod, nn.Linear) or "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
        if not is_linear_like:
            return False
        if isinstance(base_mod, SubspaceLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is None:
            return True
        return any(t in name for t in self.target_layers)

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers via subspace projection."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        from framework.core.model_utils import replace_module

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                base_mod = getattr(module, "base_layer", module)
                raw_w = base_mod.weight.data
                cls_name = base_mod.__class__.__name__
                is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf") or (
                    raw_w.ndim == 2
                    and hasattr(base_mod, "in_features")
                    and base_mod.in_features == raw_w.shape[0]
                    and raw_w.shape[0] < raw_w.shape[1]
                )
                W_orig = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
                max_rank = min(W_orig.shape[0], W_orig.shape[1])
                r = min(self.rank, max_rank)

                new_layer = SubspaceLinear.from_linear(
                    module,
                    rank=r,
                    method=self.method,
                    use_lora=self.use_lora,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_rec = new_layer.weight.float()
                err = (torch.norm(W_orig - W_rec, p="fro") / (torch.norm(W_orig, p="fro") + 1e-8)).item()

                if self.fallback_to_original and err > self.max_error:
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

                if hasattr(module, "base_layer"):
                    module.base_layer = new_layer
                    if hasattr(module, "lora_A") and module.lora_A is not None:
                        for p in (module.lora_A.parameters() if hasattr(module.lora_A, "parameters") else [module.lora_A]):
                            p.requires_grad = True
                    if hasattr(module, "lora_B") and module.lora_B is not None:
                        for p in (module.lora_B.parameters() if hasattr(module.lora_B, "parameters") else [module.lora_B]):
                            p.requires_grad = True
                else:
                    replace_module(self.model, name, new_layer)

                self.compressed_layers.append({
                    "name": name,
                    "rank": new_layer.rank,
                    "relative_error": float(err),
                })

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "subspace_rank": self.rank,
                "subspace_method": self.method,
                "compressed_layers": self.compressed_layers,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "max_error": self.max_error,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Module) -> SubspaceLinear:
        """Compress a single linear layer into SubspaceLinear."""
        return SubspaceLinear.from_linear(linear, rank=self.rank, method=self.method, use_lora=self.use_lora, lora_r=self.lora_r, lora_alpha=self.lora_alpha)


@COMPRESSOR_REGISTRY.register("subspace_projection")
@COMPRESSOR_REGISTRY.register("subspace_projection_compressor")
def subspace_projection_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for Subspace Projection compression."""
    effective_rank = rank or _cfg_get(comp_config, "subspace_rank", _cfg_get(comp_config, "rank", 64))
    method = _cfg_get(comp_config, "subspace_method", "svd")
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))

    compressor = SubspaceProjectionCompressor(
        model,
        rank=effective_rank,
        method=method,
        cfg=comp_config,
        target_attention_only=target_attention_only,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model

