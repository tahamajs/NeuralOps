#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
OATS: Outlier-Aware Low-Rank and Saliency Pruning Compressor

Separates salient outlier channels from normal weights, applying low-rank SVD to bulk weights while preserving high-magnitude paths.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.common import _cfg_get
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import apply_24_structured_mask, get_calibration_tensors
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class OATSLoRALinear(nn.Module):
    """
    Linear layer containing OATS factorized low-rank base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor | None = None,
        U: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
        S: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features
        self.nx = in_features
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        ref_t = U if U is not None else weight
        target_device = device if device is not None else (ref_t.device if ref_t is not None else None)
        target_dtype = dtype if dtype is not None else (ref_t.dtype if ref_t is not None else torch.float32)

        if U is not None and V is not None:
            self.U = nn.Parameter(
                U.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
            self.V = nn.Parameter(
                V.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
            self.register_parameter("weight_param", None)
            self.rank = U.shape[1]
        elif weight is not None:
            if weight.shape == (in_features, out_features) and in_features != out_features:
                weight = weight.t().contiguous()
            self.weight_param = nn.Parameter(
                weight.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
            self.register_parameter("U", None)
            self.register_parameter("V", None)
            self.rank = min(in_features, out_features)
        else:
            raise ValueError("Either (U, V) or weight must be provided to OATSLoRALinear")

        if S is not None:
            self.register_buffer(
                "S_sparse",
                S.to(device=target_device, dtype=target_dtype),
            )
        else:
            self.register_buffer("S_sparse", None)

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

    @property
    def weight(self) -> torch.Tensor:
        if self.U is not None and self.V is not None:
            base = torch.matmul(self.U.float(), self.V.float()).to(self.U.dtype)
            if getattr(self, "S_sparse", None) is not None:
                base = base + self.S_sparse.to(self.U.dtype)
        else:
            base = self.weight_param
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            return base + (self.scaling * torch.matmul(self.lora_B.float(), self.lora_A.float())).to(base.dtype)
        return base

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor | None = None,
        U: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
        S: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        **kwargs: Any,
    ) -> OATSLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        bias = base_layer.bias.detach() if hasattr(base_layer, "bias") and base_layer.bias is not None else None
        in_features = getattr(base_layer, "in_features", getattr(base_layer, "nx", None))
        out_features = getattr(base_layer, "out_features", getattr(base_layer, "nf", None))
        if in_features is None or out_features is None:
            if U is not None and V is not None:
                out_features = U.shape[0]
                in_features = V.shape[1]
            elif weight is not None:
                out_features = out_features or weight.shape[0]
                in_features = in_features or weight.shape[1]

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

        ref_t = U if U is not None else weight
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            U=U,
            V=V,
            S=S,
            bias=bias,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=ref_t.device if ref_t is not None else None,
            dtype=ref_t.dtype if ref_t is not None else None,
        )

        if (
            hasattr(linear, "lora_A")
            and hasattr(linear, "lora_B")
            and layer.lora_A is not None
            and layer.lora_B is not None
        ):
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
        if self.U is not None and self.V is not None:
            h = F.linear(x, self.V)
            out = F.linear(h, self.U, self.bias)
            if getattr(self, "S_sparse", None) is not None:
                out = out + F.linear(x, self.S_sparse)
        else:
            out = F.linear(x, self.weight_param, self.bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK, check_triton_suitability

            suitable, _ = check_triton_suitability(x, "low_rank")
            if not FORCE_PYTORCH_FALLBACK and suitable and x.is_cuda:
                try:
                    from neuralops.ops.triton.low_rank import fused_down_up_gemm_bias_triton

                    lora_out = fused_down_up_gemm_bias_triton(
                        x, self.lora_A.to(x.dtype), self.lora_B.t().to(x.dtype).contiguous(), bias=None
                    )
                    return out + (self.scaling * lora_out)
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass

            out = out + self.scaling * F.linear(F.linear(x.float(), self.lora_A), self.lora_B).to(out.dtype)
        return out


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
        self._compressed_params: int | None = None
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            sparsity = _cfg_get(cfg, "oats_sparsity", _cfg_get(cfg, "sparsity", sparsity))
            rank_ratio = _cfg_get(cfg, "oats_rank_ratio", _cfg_get(cfg, "rank_ratio", rank_ratio))
            num_iters = _cfg_get(cfg, "oats_num_iters", _cfg_get(cfg, "num_iters", num_iters))
            structured_24 = _cfg_get(cfg, "oats_structured_24", _cfg_get(cfg, "structured_24", structured_24))
            max_error = _cfg_get(cfg, "oats_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg,
                "oats_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
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
        target_rank = kwargs.get("target_rank", kwargs.get("rank", None))
        if cfg is not None and target_rank is None:
            target_rank = _cfg_get(cfg, "oats_rank", _cfg_get(cfg, "rank", _cfg_get(cfg, "target_rank", None)))
        self.target_rank = int(target_rank) if target_rank is not None else None

        self.target_device = (
            torch.device(device) if device is not None else next(model.parameters(), torch.tensor(0.0)).device
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
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
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
        rank: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Decomposes weight into outlier low-rank factors U, V and sparse outliers S under metric M."""
        d_out, d_in = W.shape
        device = W.device

        if activations is not None and activations.numel() > 0:
            X = activations.reshape(-1, activations.shape[-1]).float().to(device)
            if X.shape[-1] == d_in and X.shape[0] >= 1:
                M = torch.matmul(X.T, X) / max(X.shape[0], 1)
                diag_mean = torch.diagonal(M).mean().item()
                reg = max(1e-2 * diag_mean, 1e-3)
                M = M + reg * torch.eye(d_in, device=device)
            else:
                M = torch.eye(d_in, device=device)
        else:
            M = torch.eye(d_in, device=device)

        try:
            L_M = torch.linalg.cholesky(M)
            L_M_inv = torch.linalg.inv(L_M)
            if torch.isnan(L_M).any() or torch.isnan(L_M_inv).any() or torch.isinf(L_M_inv).any():
                raise ValueError("Cholesky produced NaNs or Infs")
        except Exception:
            L_M = torch.eye(d_in, device=device)
            L_M_inv = torch.eye(d_in, device=device)

        W_scaled = torch.matmul(W, L_M)
        target_r = rank if rank is not None else self.target_rank
        if target_r is not None:
            r = min(int(target_r), min(d_out, d_in) - 1)
        else:
            r = max(
                min(self.min_rank, min(d_out, d_in)),
                min(int(self.rank_ratio * min(d_out, d_in)), min(d_out, d_in)),
            )
            r = min(r, min(d_out, d_in) - 1)
        r = max(1, r)

        try:
            U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
            curr_r = min(r, len(S))
            L = torch.matmul(U[:, :curr_r] * S[:curr_r].unsqueeze(0), Vh[:curr_r, :])
        except Exception:
            curr_r = r
            L = torch.zeros_like(W_scaled)

        R = W_scaled - L
        if self.structured_24:
            S_mat, _ = apply_24_structured_mask(R, saliency=torch.abs(R), dim=1)
        else:
            flat = R.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (R.abs() >= thr).float()
            S_mat = R * mask

        for _ in range(self.num_iters):
            try:
                U, S, Vh = torch.linalg.svd(W_scaled - S_mat, full_matrices=False)
                curr_r = min(r, len(S))
                L = torch.matmul(U[:, :curr_r] * S[:curr_r].unsqueeze(0), Vh[:curr_r, :])
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

        try:
            U, S, Vh = torch.linalg.svd(W_scaled - S_mat, full_matrices=False)
            curr_r = min(r, len(S))
            sqrt_S = torch.sqrt(torch.clamp(S[:curr_r], min=1e-12))
            U_factor = U[:, :curr_r] * sqrt_S.unsqueeze(0)
            V_factor = sqrt_S.unsqueeze(1) * torch.matmul(Vh[:curr_r, :], L_M_inv)
            S_sparse = torch.matmul(S_mat, L_M_inv)
        except Exception as e:
            logger.warning("OATS factorization failed: %s; falling back to direct SVD", e)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            curr_r = min(r, len(S))
            sqrt_S = torch.sqrt(torch.clamp(S[:curr_r], min=1e-12))
            U_factor = U[:, :curr_r] * sqrt_S.unsqueeze(0)
            V_factor = sqrt_S.unsqueeze(1) * Vh[:curr_r, :]
            S_sparse = None

        if (
            torch.isnan(U_factor).any()
            or torch.isnan(V_factor).any()
            or torch.isinf(U_factor).any()
            or torch.isinf(V_factor).any()
        ):
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            curr_r = min(r, len(S))
            sqrt_S = torch.sqrt(torch.clamp(S[:curr_r], min=1e-12))
            U_factor = U[:, :curr_r] * sqrt_S.unsqueeze(0)
            V_factor = sqrt_S.unsqueeze(1) * Vh[:curr_r, :]
            S_sparse = None

        return U_factor, V_factor, S_sparse, curr_r

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

                W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
                device = raw_w.device
                dtype = raw_w.dtype

                act = activations_map.get(name, None)
                try:
                    U_factor, V_factor, S_sparse, r = self._oats_decompose(W, act, rank=self.target_rank)
                    compressed_layer = OATSLoRALinear.from_linear(
                        module,
                        U=U_factor.to(dtype=dtype, device=device),
                        V=V_factor.to(dtype=dtype, device=device),
                        S=S_sparse.to(dtype=dtype, device=device) if S_sparse is not None else None,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    w_norm = torch.norm(W, p="fro").item()
                    w_rec = compressed_layer.weight
                    err = (torch.norm(W - w_rec.float(), p="fro") / max(w_norm, 1e-12)).item()
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

        forward_check_success = True
        try:
            device = next(self.model.parameters(), torch.tensor(0.0)).device
            with torch.no_grad():
                cfg = getattr(self.model, "config", None)
                if cfg is not None and getattr(cfg, "vocab_size", None) is not None:
                    dummy = torch.randint(0, min(cfg.vocab_size, 100), (1, 8), device=device)
                    self.model(dummy)
                else:
                    dim = 64
                    for m in self.model.modules():
                        if hasattr(m, "in_features"):
                            dim = m.in_features
                            break
                    first_p = next(self.model.parameters())
                    dummy = torch.randn(1, dim, device=device, dtype=first_p.dtype)
                    self.model(dummy)
        except Exception as e:
            logger.warning("OATS post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
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


def oats_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for OATS compressor."""
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "oats_sparsity", _cfg_get(comp_config, "sparsity", 0.5)),
    )
    rank_ratio = kwargs.pop(
        "rank_ratio",
        _cfg_get(comp_config, "oats_rank_ratio", _cfg_get(comp_config, "rank_ratio", 0.2)),
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
        _cfg_get(comp_config, "oats_max_error", _cfg_get(comp_config, "max_error", 0.85)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "oats_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
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
