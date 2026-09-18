"""
AdaSVD: Adaptive SVD with per-layer importance allocation and iterative ALS refinement.
Registered as 'adasvd'. Production hardened with reconstruction validation, fallback,
Conv1D support, rank reduction bounds, and comprehensive per-layer stats.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    try:
        from transformers.modeling_utils import Conv1D
    except ImportError:
        Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("adasvd")
class AdaSVDCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Adaptive SVD Compressor (AdaSVD).

    Assigns SVD ranks dynamically across layers proportional to spectral importance
    (nuclear norm ||W||_* = sum(sigma_i)), initializes factorized matrices with
    symmetric square-root singular value distribution (A = U * sqrt(Sigma), B = sqrt(Sigma) * V^T),
    refines factors via Alternating Least Squares (ALS), and replaces dense layers
    with factorized SVDLinear modules.

    Hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - ALS check vs initial SVD: reverts to initial SVD factors if ALS worsens error.
    - Safe fallback to original layer if error exceeds max_error or factorization fails.
    - Preserves LoRALinear wrappers: replaces inner base_layer with SVDLinear, retaining trainable LoRA deltas.
    - Enforced rank safeguard: r <= max_param_r and r >= min_rank (or 15% of min dimension).
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition to [out_features, in_features].
    - GPT-2 c_attn support: max_param_r=432, rank floor >= 288.
    - Target attention only option: skips FFN/MLP projections for safety.
    - Full FP32 computation for numerical stability, casting back to original dtype at module insertion.
    - Per-layer logging and detailed compression statistics.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        target_param_ratio: float = 0.7,
        num_iters: int = 5,
        max_error: float = 0.40,
        fallback_to_original: bool = True,
        target_attention_only: bool = True,
        min_rank: int = 16,
        target_modules: Optional[List[str]] = None,
        comp_config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adasvd", **kwargs)
        cfg = comp_config or kwargs.get("comp_config", kwargs.get("cfg", getattr(model, "config", None)))

        self.rank = _cfg_get(cfg, "adasvd_rank", _cfg_get(cfg, "rank", rank))
        self.target_param_ratio = float(
            _cfg_get(cfg, "adasvd_target_ratio", _cfg_get(cfg, "target_param_ratio", target_param_ratio))
        )
        self.num_iters = int(_cfg_get(cfg, "adasvd_iters", _cfg_get(cfg, "num_iters", num_iters)))
        self.max_error = float(_cfg_get(cfg, "adasvd_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.fallback_to_original = bool(
            _cfg_get(cfg, "adasvd_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.target_attention_only = bool(
            _cfg_get(cfg, "adasvd_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.min_rank = int(_cfg_get(cfg, "adasvd_min_rank", _cfg_get(cfg, "min_rank", min_rank)))
        self.target_modules = target_modules
        self.comp_config = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.layer_iters: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, SVDLinear):
            return False
        # Do not target inner submodules of LoRA or wrappers
        if any(name.endswith(sfx) for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        # Extract weight and ensure it is strictly a 2D matrix (filters out 3D/4D convs like nn.Conv1d)
        weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return False

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
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
            # Skip FFN / MLP / feed-forward layers
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            # RoBERTa layer.X.output.dense is FFN, while attention.output.dense is attention
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            # Explicitly match attention projections including GPT-2 c_attn and c_proj, and Qwen linear_attn
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj", "query", "key", "value",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    @staticmethod
    def _compute_nuclear_norm(W: torch.Tensor) -> float:
        """Compute nuclear norm ||W||_* = sum(sigma_i) in FP32."""
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            return float(s.sum().item())

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
        name: str = "",
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """
        Compress a single nn.Linear, Conv1D, or LoRALinear layer using symmetric SVD and ALS refinement.
        Returns: (svd_layer, relative_reconstruction_error, actual_als_iters)
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = layer.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()
        is_conv1d = (
            (Conv1D is not None and (isinstance(layer, Conv1D) or isinstance(target_mod, Conv1D)))
            or hasattr(layer, "nf")
            or hasattr(target_mod, "nf")
        )

        weight_tensor = getattr(target_mod, "weight", getattr(layer, "weight", None))
        if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
            return None, float("inf"), 0

        raw_w = weight_tensor.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()  # [out_features, in_features]

        # Enforce parameter reduction bound: r * (in_f + out_f) <= 0.75 * in_f * out_f
        max_param_r = max(1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f))))

        # Minimum rank floor: min_rank or 15% of min dimension
        r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))

        # For GPT-2 c_attn (768->2304): max_param_r is 432, rank floor is at least 288
        if "c_attn" in name.lower() or (in_f >= 768 and out_f >= 2000):
            r_floor = max(r_floor, 288)
            max_param_r = min(max_param_r, 432)

        r = max(r_floor, min(rank, max_param_r))
        if r > max_param_r:
            r = max_param_r

        device = weight_tensor.device
        dtype = weight_tensor.dtype

        # Compute truncated SVD in FP32
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("AdaSVD SVD computation failed on %s: %s", name, e)
            return None, 1.0, 0

        # Symmetric square-root singular value initialization
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A_0 = U[:, :r] * sqrt_S.unsqueeze(0)       # [out_features, r]
        B_0 = sqrt_S.unsqueeze(1) * Vt[:r, :]      # [r, in_features]

        W_norm = torch.norm(W, p="fro").item()
        if W_norm < 1e-12:
            init_err = 0.0
        else:
            recon_0 = torch.matmul(A_0, B_0)
            init_err = float(torch.norm(W - recon_0, p="fro").item() / W_norm)

        A = A_0.clone()
        B = B_0.clone()
        actual_iters = 0

        # Alternating Least Squares (ALS) refinement
        eps_eye = 1e-6 * torch.eye(r, device=device, dtype=torch.float32)
        for it in range(self.num_iters):
            BBt = B @ B.t().contiguous() + eps_eye
            try:
                A_next = W @ B.t().contiguous() @ torch.linalg.inv(BBt)
                AtA = A_next.t().contiguous() @ A_next + eps_eye
                B_next = torch.linalg.inv(AtA) @ A_next.t().contiguous() @ W
            except Exception as e:
                logger.debug("AdaSVD ALS inversion failed at iter %d on %s: %s", it, name, e)
                break

            if (
                torch.isnan(A_next).any()
                or torch.isnan(B_next).any()
                or torch.isinf(A_next).any()
                or torch.isinf(B_next).any()
            ):
                logger.debug("AdaSVD NaN/Inf during ALS iter %d on %s. Halting ALS.", it, name)
                break

            A, B = A_next, B_next
            actual_iters += 1

        # Check ALS factors validity and error vs initial SVD
        if (
            torch.isnan(A).any()
            or torch.isnan(B).any()
            or torch.isinf(A).any()
            or torch.isinf(B).any()
        ):
            A, B = A_0, B_0
            final_err = init_err
        else:
            recon = torch.matmul(A, B)
            if W_norm < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - recon, p="fro").item() / W_norm)

            if final_err > init_err:
                logger.debug(
                    "AdaSVD: ALS error (%.4f) > initial SVD error (%.4f) on %s. Reverting to initial SVD.",
                    final_err,
                    init_err,
                    name,
                )
                A, B = A_0, B_0
                final_err = init_err

        # Instantiate structural SVDLinear
        bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))
        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias_tensor is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        if bias_tensor is not None and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(bias_tensor.data)

        return svd_layer, final_err, actual_iters

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Executes adaptive rank assignment and ALS refinement across all target linear/conv1d layers.
        """
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_iters = {}

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaSVD: No compressible layers found.")
            return self.model, self.get_compression_stats()

        # Step 1: Compute spectral nuclear norms in FP32
        layer_importances = {}
        total_importance = 0.0
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            cls_name = module.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
            if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
                continue
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            raw_w = weight_tensor.data
            W_2d = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            imp = self._compute_nuclear_norm(W_2d)
            layer_importances[name] = max(imp, 1e-8)
            total_importance += layer_importances[name]

        # Step 2: Assign ranks per layer and compress
        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            cls_name = module.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            weight_tensor = getattr(target_mod, "weight", getattr(module, "weight", None))
            if weight_tensor is None or not hasattr(weight_tensor, "dim") or weight_tensor.dim() != 2:
                continue
            is_conv1d = (
                (Conv1D is not None and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D)))
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            raw_w = weight_tensor.data
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            max_param_r = max(1, min(in_f, out_f, int(0.75 * (in_f * out_f) / (in_f + out_f))))
            r_floor = max(self.min_rank, int(0.15 * min(in_f, out_f)))
            if "c_attn" in name.lower() or (in_f >= 768 and out_f >= 2000):
                r_floor = max(r_floor, 288)
                max_param_r = min(max_param_r, 432)

            if self.rank is not None:
                frac = layer_importances.get(name, 1.0) / (total_importance / max(1, len(target_layers)))
                allocated_rank = int(round(self.rank * frac))
            else:
                frac = layer_importances.get(name, 1.0) / max(total_importance, 1e-8)
                allocated_rank = int(round(min(in_f, out_f) * self.target_param_ratio * frac * len(target_layers)))

            allocated_rank = max(r_floor, min(allocated_rank, max_param_r))
            if allocated_rank > max_param_r:
                allocated_rank = max_param_r

            svd_layer, err, iters = self.compress_linear(module, rank=allocated_rank, name=name)
            self.layer_errors[name] = err
            self.layer_ranks[name] = allocated_rank
            self.layer_iters[name] = iters

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "AdaSVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping AdaSVD compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            if hasattr(module, "base_layer"):
                module.base_layer = svd_layer
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    for p in (module.lora_A.parameters() if hasattr(module.lora_A, "parameters") else [module.lora_A]):
                        p.requires_grad = True
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    for p in (module.lora_B.parameters() if hasattr(module.lora_B, "parameters") else [module.lora_B]):
                        p.requires_grad = True
            else:
                replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)

        # Explicitly freeze fallback layers so recovery fine-tuning does not touch them
        for name, mod in self.model.named_modules():
            if name in self.fallback_layers:
                target = getattr(mod, "base_layer", mod)
                for p in target.parameters():
                    p.requires_grad = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "layer_iters": self.layer_iters,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats["avg_reconstruction_error"] = avg_err
        logger.info(
            "AdaSVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


def adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """AdaSVD factory function with robust config handling."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "adasvd_rank", _cfg_get(comp_config, "rank", None)))
    target_param_ratio = kwargs.pop(
        "target_param_ratio", _cfg_get(comp_config, "adasvd_target_ratio", _cfg_get(comp_config, "target_param_ratio", 0.7))
    )
    num_iters = kwargs.pop("num_iters", _cfg_get(comp_config, "adasvd_iters", _cfg_get(comp_config, "num_iters", 5)))
    max_error = kwargs.pop(
        "max_error", _cfg_get(comp_config, "adasvd_max_error", _cfg_get(comp_config, "max_error", 0.40))
    )
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "adasvd_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "adasvd_target_attention_only", _cfg_get(comp_config, "target_attention_only", True))
    )
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "adasvd_min_rank", _cfg_get(comp_config, "min_rank", 16)))
    compressor = AdaSVDCompressor(
        model,
        rank=rank,
        target_param_ratio=target_param_ratio,
        num_iters=num_iters,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        min_rank=min_rank,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


adasvd_compressor_factory = adasvd_factory

__all__ = ["AdaSVDCompressor", "adasvd_factory", "adasvd_compressor_factory"]
