"""
COMPOT: Compressed Orthogonal Dictionary Learning Compressor.
Registered in COMPRESSOR_REGISTRY as 'compot'. Production hardened for
multi-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with
reconstruction validation, Conv1D support, rank reduction bounds, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.pruning_utils import get_calibration_tensors, adaptive_max_error
from framework.utils.compat import should_compress_linear


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


logger = logging.getLogger(__name__)


class COMPOTLinear(nn.Module):
    is_custom_attention: bool = True
    """
    Factorized Linear Layer constructed via Orthogonal Dictionary Learning (COMPOT):
    W ≈ B @ A
    - A in R^{rank x in_features}: input-orthogonal projection matrix
    - B in R^{out_features x rank}: output dictionary matrix

    Forward pass:
    y = ((x @ A^T) @ B^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        self.A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        with torch.no_grad():
            A_f32 = torch.empty(self.rank, in_features, dtype=torch.float32)
            nn.init.orthogonal_(A_f32)
            self.A.copy_(A_f32.to(device=device, dtype=dtype))
            nn.init.xavier_uniform_(self.B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.B, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass matching standard Linear and Conv1D."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_a = F.linear(x_flat, self.A)
        out = F.linear(x_a, self.B, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full weight matrix W = B @ A."""
        return torch.matmul(self.B, self.A)

    def __iter__(self):
        """Allow tuple unpacking: compot_layer, err, used_act = COMPOTLinear.from_linear_and_activations(...)."""
        yield self
        yield getattr(self, "reconstruction_error", 0.0)
        yield getattr(self, "used_activations", False)

    def __getitem__(self, idx: int) -> Any:
        return (self, getattr(self, "reconstruction_error", 0.0), getattr(self, "used_activations", False))[idx]

    @classmethod
    def from_linear_and_activations(
        cls,
        layer: nn.Module,
        activations: torch.Tensor | None = None,
        rank: int = 64,
    ) -> tuple[COMPOTLinear, float, bool]:
        """
        Construct COMPOTLinear by learning orthogonal dictionary from activations or weights.
        Returns: (compot_layer, relative_reconstruction_error, used_activations)
        """
        cls_name = layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")
        raw_w = layer.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W_float = raw_w.t().contiguous().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W_float = raw_w.float()      # [out_features, in_features]

        bias = layer.bias.data if layer.bias is not None else None
        r = max(1, min(rank, out_f, in_f))
        device = raw_w.device
        dtype = raw_w.dtype

        compot_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=dtype,
            device=device,
        )

        used_act = False
        A_data = None
        B_data = None

        if activations is not None and activations.size(0) >= 4:
            X = activations.float().to(device)
            if X.shape[-1] == in_f:
                try:
                    # QR decomposition of activations: X = Q @ R
                    Q, R = torch.linalg.qr(X, mode="reduced")
                    D = torch.matmul(W_float, R.t().contiguous())
                    U, S, Vt = torch.linalg.svd(D, full_matrices=False)

                    B_cand = U[:, :r] * S[:r].unsqueeze(0)  # (out_f, r)
                    V_r = Vt[:r, :]                         # (r, in_f)
                    R_inv_T = torch.linalg.pinv(R.t().contiguous())
                    A_cand = torch.matmul(V_r, R_inv_T)

                    if not (
                        torch.isnan(A_cand).any()
                        or torch.isnan(B_cand).any()
                        or torch.isinf(A_cand).any()
                        or torch.isinf(B_cand).any()
                    ):
                        A_data = A_cand
                        B_data = B_cand
                        used_act = True
                except Exception as e:
                    logger.debug("COMPOT QR decomposition exception on layer: %s", e)

        # Fallback to direct SVD on weight matrix
        if A_data is None or B_data is None:
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            B_data = U[:, :r] * sqrt_S.unsqueeze(0)
            A_data = sqrt_S.unsqueeze(1) * Vt[:r, :]
            used_act = False

        compot_layer.A.data.copy_(A_data.to(dtype=dtype, device=device))
        compot_layer.B.data.copy_(B_data.to(dtype=dtype, device=device))

        if bias is not None and compot_layer.bias is not None:
            compot_layer.bias.data.copy_(bias)

        # Compute relative Frobenius reconstruction error
        W_recon = torch.matmul(B_data, A_data)
        norm_orig = torch.norm(W_float, p="fro").item()
        if norm_orig < 1e-12:
            rel_err = 0.0
        else:
            rel_err = float(torch.norm(W_float - W_recon, p="fro").item() / norm_orig)

        compot_layer.reconstruction_error = rel_err
        compot_layer.used_activations = used_act
        return compot_layer


@COMPRESSOR_REGISTRY.register("compot_compressor_class")
class COMPOTCompressor(BaseCompressor):
    """
    COMPOT (Compressed Orthogonal Dictionary Learning) Compressor.
    Learns a low-rank basis from activation QR decompositions and compresses
    linear and Conv1D layers into factorized COMPOTLinear modules.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original layer if error exceeds threshold or decomposition is unstable.
    - SVD fallback if QR decomposition on activations fails.
    - Parameter reduction guarantee: checks r * (in + out) < in * out.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - Comprehensive per-layer stats and error reporting.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int = 64,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        calibration_texts: list[str] | None = None,
        calibration_loader: Any | None = None,
        tokenizer: Any | None = None,
        cfg: Any | None = None,
        target_attention_only: bool = False,
        target_layers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="compot", **kwargs)
        conf = cfg or kwargs.get("comp_config")
        if conf is not None:
            rank = getattr(conf, "compot_rank", getattr(conf, "rank", rank))
            max_error = getattr(conf, "compot_max_error", getattr(conf, "max_error", max_error))
            fallback_to_original = getattr(
                conf, "compot_fallback_to_original", getattr(conf, "fallback_to_original", fallback_to_original)
            )

        self.rank = int(rank)
        if max_error is None or max_error == 0.20:
            max_error = adaptive_max_error(self.rank, 768, 768, "distilbert")

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.calibration_texts = calibration_texts
        self.calibration_loader = calibration_loader
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.target_layers = target_layers
        self.target_attention_only = target_attention_only

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.used_activations: dict[str, bool] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(module, COMPOTLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_attention_only and not any(k in name.lower() for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "c_attn", "c_proj"]):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def _collect_activations(self) -> dict[str, torch.Tensor]:
        """Collect input activations across target modules."""
        layer_data: dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    inp = input[0].detach().float()
                    cls_name = module.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                    in_f = module.weight.shape[0] if is_conv1d else module.weight.shape[1]
                    flat = inp.reshape(-1, in_f)
                    if name not in layer_data:
                        layer_data[name] = flat.cpu()
                    else:
                        layer_data[name] = torch.cat([layer_data[name], flat.cpu()], dim=0)
            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters(), torch.tensor(0.0)).device
        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=device,
        )

        self.model.eval()
        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("COMPOT activation collection forward pass exception: %s", e)
            finally:
                for h in hooks:
                    h.remove()

        return layer_data

    def compress(self, model: nn.Module | None = None) -> tuple[nn.Module, dict[str, Any]]:
        """Compress linear and Conv1D layers via COMPOT orthogonal dictionary learning."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        activations = self._collect_activations()
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.used_activations = {}

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("COMPOT: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            target_mod = getattr(module, 'base_layer', module)
            raw_w = target_mod.weight.data
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            # Guarantee parameter reduction: r * (in_f + out_f) < in_f * out_f
            max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
            max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))
            r = max(1, min(self.rank, max_r))

            act = activations.get(name, None)
            compot_layer, err, used_act = COMPOTLinear.from_linear_and_activations(target_mod, activations=act, rank=r)
            self.layer_errors[name] = err
            self.layer_ranks[name] = r
            self.used_activations[name] = used_act

            # Validate dimensions
            if compot_layer.in_features != in_f or compot_layer.out_features != out_f:
                logger.warning("Dimension mismatch on %s: (%d, %d) vs (%d, %d)", name, compot_layer.out_features, compot_layer.in_features, out_f, in_f)
                for p in module.parameters(): p.requires_grad = False
                self.fallback_layers.append(name)
                continue

            if err > self.max_error:
                logger.warning(
                    "COMPOT reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping COMPOT compression for %s (fallback to original).", name)
                    for p in module.parameters(): p.requires_grad = False
                    self.fallback_layers.append(name)
                    continue

            if hasattr(module, 'base_layer'):
                if hasattr(module, 'lora_A'):
                    compot_layer.lora_A = module.lora_A
                if hasattr(module, 'lora_B'):
                    compot_layer.lora_B = module.lora_B
                module.base_layer = compot_layer
                compot_layer = module
            replace_module(self.model, name, compot_layer)
            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "used_activations": self.used_activations,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "compot_rank": self.rank,
            }
        )
        logger.info(
            "Compressor compot finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("compot")
@COMPRESSOR_REGISTRY.register("compot_compressor")
@COMPRESSOR_REGISTRY.register("compot_factory")
@COMPRESSOR_REGISTRY.register("compot_compressor_factory")
def compot_compressor_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Factory function for COMPOT compression."""
    effective_rank = rank or getattr(comp_config, "compot_rank", None) or getattr(comp_config, "rank", 64)
    cal_texts = getattr(comp_config, "compot_calibration_texts", None)
    tokenizer = getattr(comp_config, "tokenizer", None) or getattr(model, "tokenizer", None)
    max_error = getattr(comp_config, "compot_max_error", getattr(comp_config, "max_error", 0.20))
    fallback = getattr(comp_config, "compot_fallback_to_original", getattr(comp_config, "fallback_to_original", True))

    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    compressor = COMPOTCompressor(
        model,
        rank=effective_rank,
        max_error=max_error,
        fallback_to_original=fallback,
        calibration_texts=cal_texts,
        tokenizer=tokenizer,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = ["COMPOTLinear", "COMPOTCompressor", "compot_compressor_factory"]
