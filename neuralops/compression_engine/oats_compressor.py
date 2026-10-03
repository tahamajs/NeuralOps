"""
OATS: Outlier-Aware Training & Saliency-Based Sparsification with Low-Rank Matrix Decomposition.
Registered in COMPRESSOR_REGISTRY as 'oats' and 'oats_compressor'.

Mathematical Formulation:
1. Activation Second Moment:
   $$M = \\frac{1}{N} X^\\top X + \\epsilon I \\in \\mathbb{R}^{d_{in} \\times d_{in}}$$
2. Cholesky Factorization:
   $$M = L_M L_M^\\top$$
3. Preconditioned Weight Matrix:
   $$\\widetilde{W} = W L_M$$
4. Alternating Low-Rank + Sparse Decomposition:
   - Low-Rank Outlier Component: $L = \\operatorname{SVD}_r(\\widetilde{W} - S)$
   - Sparse Residual Component: $S = \\Pi_{2:4}(\\widetilde{W} - L)$ or $\\operatorname{Top-}(1-s)(\\widetilde{W} - L)$
5. De-conditioned Weight Matrix:
   $$W_{oats} = (L + S) L_M^{-1}$$
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import (
    apply_24_structured_mask,
    get_calibration_tensors,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class OATSLoRALinear(nn.Module):
    is_custom_attention: bool = True
    """
    Linear layer containing OATS decomposed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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

        # Strictly ensure nn.Linear weight orientation (out_features, in_features) for F.linear.
        # Conv1D stores weight as (in_features, out_features) = (nx, nf).
        if weight.shape == (in_features, out_features) and in_features != out_features:
            weight = weight.t().contiguous()

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
                torch.empty(
                    (self.lora_r, in_features),
                    dtype=torch.float32,
                    device=target_device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (out_features, self.lora_r),
                    dtype=torch.float32,
                    device=target_device,
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        **kwargs: Any,
    ) -> OATSLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        if weight is None and hasattr(base_layer, "weight"):
            weight = base_layer.weight.detach()
        if "rank" in kwargs and lora_r == 0:
            lora_r = kwargs["rank"]
        bias = (
            base_layer.bias.detach()
            if hasattr(base_layer, "bias") and base_layer.bias is not None
            else None
        )
        in_features = getattr(
            base_layer, "in_features", getattr(base_layer, "nx", None)
        )
        out_features = getattr(
            base_layer, "out_features", getattr(base_layer, "nf", None)
        )
        if in_features is None and out_features is None:
            out_features, in_features = weight.shape
        elif in_features is None:
            in_features = (
                weight.shape[1] if weight.shape[0] == out_features else weight.shape[0]
            )
        elif out_features is None:
            out_features = (
                weight.shape[0] if weight.shape[1] == in_features else weight.shape[1]
            )

        # Strictly orient weight as (out_features, in_features) for F.linear
        if weight.shape == (in_features, out_features) and in_features != out_features:
            weight = weight.t().contiguous()

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(
            linear, "lora_alpha", getattr(linear, "alpha", lora_alpha)
        )

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
        if (
            hasattr(linear, "lora_A")
            and hasattr(linear, "lora_B")
            and layer.lora_A is not None
            and layer.lora_B is not None
        ):
            with torch.no_grad():
                src_A = (
                    linear.lora_A.weight.data
                    if hasattr(linear.lora_A, "weight")
                    else linear.lora_A.data
                )
                src_B = (
                    linear.lora_B.weight.data
                    if hasattr(linear.lora_B, "weight")
                    else linear.lora_B.data
                )
                if (
                    src_A.shape == layer.lora_A.shape
                    and src_B.shape == layer.lora_B.shape
                ):
                    layer.lora_A.copy_(
                        src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype)
                    )
                    layer.lora_B.copy_(
                        src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype)
                    )

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
    "attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
    "self_attn",
}


@COMPRESSOR_REGISTRY.register("oats")
@COMPRESSOR_REGISTRY.register("oats_compressor")
class OATSCompressor(BaseCompressor):
    """
    OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.85).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection by class name and nf attribute without unreliable shape heuristics.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping to protect asymmetric attention heads.
    - Tied / shared weight protection tracking parameter IDs.
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        sparsity: float = 0.5,
        rank_ratio: float = 0.2,
        num_iters: int = 4,
        device: str | torch.device | None = None,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        structured_24: bool = True,
        max_error: float = 0.85,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        target_modules: list[str] | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        min_rank: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="oats", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            sparsity = _cfg_get(
                cfg, "oats_sparsity", _cfg_get(cfg, "sparsity", sparsity)
            )
            rank_ratio = _cfg_get(
                cfg, "oats_rank_ratio", _cfg_get(cfg, "rank_ratio", rank_ratio)
            )
            num_iters = _cfg_get(
                cfg, "oats_num_iters", _cfg_get(cfg, "num_iters", num_iters)
            )
            structured_24 = _cfg_get(
                cfg, "oats_structured_24", _cfg_get(cfg, "structured_24", structured_24)
            )
            max_error = _cfg_get(
                cfg, "oats_max_error", _cfg_get(cfg, "max_error", max_error)
            )
            fallback_to_original = _cfg_get(
                cfg,
                "oats_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.rank_ratio = max(0.01, min(float(rank_ratio), 0.9))
        self.num_iters = max(1, int(num_iters))
        self.structured_24 = bool(structured_24)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.min_rank = max(1, int(min_rank))

        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.stats: dict[str, Any] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        """Check if layer is an attention QKV projection in a model with GQA."""
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        ):
            name_lower = name.lower()
            if any(
                p in name_lower
                for p in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                ]
            ):
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

    def _collect_activations(self) -> dict[str, torch.Tensor]:
        """Collect input activations across all compressible linear/conv1d modules."""
        self.model.eval()
        activations: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(inp.cpu())

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("OATS activation collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat(act_list, dim=0)

        return aggregated

    def _oats_decompose(
        self,
        W: torch.Tensor,
        activations: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        """Decomposes weight into outlier low-rank L + sparse S under metric M."""
        d_out, d_in = W.shape
        device = W.device

        # Compute M
        if activations is not None and activations.numel() > 0:
            X = activations.reshape(-1, activations.shape[-1]).float().to(device)
            if X.shape[-1] == d_in:
                M = torch.matmul(X.T, X) / max(X.shape[0], 1)
                diag_mean = torch.diagonal(M).mean().item()
                reg = max(1e-4 * diag_mean, 1e-5)
                M = M + reg * torch.eye(d_in, device=device)
            else:
                M = torch.eye(d_in, device=device)
        else:
            M = torch.eye(d_in, device=device)

        try:
            L_M = torch.linalg.cholesky(M)
            L_M_inv = torch.linalg.inv(L_M)
            if torch.isnan(L_M).any() or torch.isnan(L_M_inv).any():
                raise ValueError("Cholesky produced NaNs")
        except Exception:
            L_M = torch.eye(d_in, device=device)
            L_M_inv = torch.eye(d_in, device=device)

        # Scale weights: W_scaled = W @ L_M
        W_scaled = torch.matmul(W, L_M)
        rank = max(
            min(self.min_rank, min(d_out, d_in)),
            min(int(self.rank_ratio * min(d_out, d_in)), min(d_out, d_in)),
        )

        # Initialize L via truncated SVD
        try:
            U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
            r = min(rank, len(S))
            L = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
        except Exception:
            r = rank
            L = torch.zeros_like(W_scaled)

        # Initialize S
        R = W_scaled - L
        if self.structured_24:
            S_mat, _ = apply_24_structured_mask(R, saliency=torch.abs(R), dim=1)
        else:
            flat = R.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (R.abs() >= thr).float()
            S_mat = R * mask

        # Alternating minimization iterations
        for _ in range(self.num_iters):
            try:
                U, S, Vh = torch.linalg.svd(W_scaled - S_mat, full_matrices=False)
                r = min(rank, len(S))
                L = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
            except Exception as e:
                logger.warning(
                    "OATS SVD alternating iteration failed: %s; keeping previous decomposition",
                    e,
                )
                break

            R = W_scaled - L
            if self.structured_24:
                S_mat, _ = apply_24_structured_mask(R, saliency=torch.abs(R), dim=1)
            else:
                flat = R.abs().reshape(-1)
                k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
                thr = torch.topk(flat, k, largest=True).values[-1]
                mask = (R.abs() >= thr).float()
                S_mat = R * mask

        # De-scale: W_comp = (L + S_mat) @ L_M^-1
        try:
            W_comp = torch.matmul(L + S_mat, L_M_inv)
        except Exception as e:
            logger.warning("OATS de-scaling failed: %s; returning original weight", e)
            return W, rank

        if torch.isnan(W_comp).any() or torch.isinf(W_comp).any():
            logger.warning(
                "OATS decomposition produced NaNs or Infs; returning original weight"
            )
            return W, rank
        return W_comp, r

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Executes OATS decomposition across compressible Linear and Conv1D modules."""
        activations_map = self._collect_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        seen_weights: set[int] = set()
        total_orig_params = 0
        total_comp_params = 0

        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                base_module = getattr(module, "base_layer", module)
                if not hasattr(base_module, "weight") or base_module.weight is None:
                    continue

                w_id = id(base_module.weight)
                if w_id in seen_weights:
                    logger.info("Skipping %s due to shared/tied weight.", name)
                    self.skipped_layers.append(name)
                    continue
                seen_weights.add(w_id)

                for p in module.parameters():
                    total_orig_params += p.numel()

                raw_w = base_module.weight.data
                cls_name = base_module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(base_module, "nf")

                W = (
                    raw_w.t().contiguous().clone().float()
                    if is_conv1d
                    else raw_w.clone().float()
                )
                device = raw_w.device
                dtype = raw_w.dtype

                act = activations_map.get(name, None)
                try:
                    W_comp, r = self._oats_decompose(W, act)
                    w_norm = torch.norm(W, p="fro").item()
                    err = (torch.norm(W - W_comp, p="fro") / max(w_norm, 1e-12)).item()
                    self.layer_errors[name] = float(err)
                    self.layer_ranks[name] = r

                    if not math.isfinite(err) or err > self.max_error:
                        logger.warning(
                            "OATS reconstruction error %.4f on %s exceeded threshold %.4f.",
                            err,
                            name,
                            self.max_error,
                        )
                        if self.fallback_to_original:
                            logger.info(
                                "Skipping OATS compression for %s (fallback to original).",
                                name,
                            )
                            self.fallback_layers.append(name)
                            for p in module.parameters():
                                p.requires_grad = False
                                total_comp_params += p.numel()
                            continue

                    W_target = W_comp
                    compressed_layer = OATSLoRALinear.from_linear(
                        module,
                        weight=W_target.to(dtype=dtype, device=device),
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    replace_module(self.model, name, compressed_layer)
                    self.compressed_layers.append(name)
                    for p in compressed_layer.parameters():
                        total_comp_params += p.numel()
                except Exception as e:
                    logger.warning("OATS decomposition exception in %s: %s", name, e)
                    self.fallback_layers.append(name)
                    self.layer_errors[name] = 1.0
                    for p in module.parameters():
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
            logger.warning("OATS post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        eff_ratio = total_comp_params / max(1, total_orig_params)
        stats.update(
            {
                "decomposed_layers": len(self.compressed_layers),
                "sparsity_target": self.sparsity,
                "rank_ratio": self.rank_ratio,
                "structured_24": self.structured_24,
                "num_iters": self.num_iters,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "effective_parameter_ratio": eff_ratio,
                "forward_check_success": forward_check_success,
                "method": "oats_outlier_decomposition",
            }
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("oats_factory")
@COMPRESSOR_REGISTRY.register("oats_compressor_factory")
def oats_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for OATS compressor."""
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "oats_sparsity", _cfg_get(comp_config, "sparsity", 0.5)),
    )
    rank_ratio = kwargs.pop(
        "rank_ratio",
        _cfg_get(
            comp_config, "oats_rank_ratio", _cfg_get(comp_config, "rank_ratio", 0.2)
        ),
    )
    num_iters = kwargs.pop(
        "num_iters",
        _cfg_get(comp_config, "oats_num_iters", _cfg_get(comp_config, "num_iters", 4)),
    )
    structured_24 = kwargs.pop(
        "structured_24",
        _cfg_get(
            comp_config,
            "oats_structured_24",
            _cfg_get(comp_config, "structured_24", True),
        ),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "oats_max_error", _cfg_get(comp_config, "max_error", 0.85)
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "oats_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    tokenizer = kwargs.pop(
        "tokenizer",
        getattr(model, "tokenizer", None) or _cfg_get(comp_config, "tokenizer", None),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = OATSCompressor(
        model,
        tokenizer=tokenizer,
        sparsity=sparsity,
        rank_ratio=rank_ratio,
        num_iters=num_iters,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
