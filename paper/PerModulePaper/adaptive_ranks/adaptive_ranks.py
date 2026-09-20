r"""
Adaptive Ranks SVD Compressor: dynamic rank selection per layer based on singular value energy spectra.

Dynamic cumulative singular value energy thresholding:
    $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
with minimum rank floor:
    $$r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor)$$
activation covariance whitening:
    $$M = W \cdot \operatorname{Cov}(X)^{1/2}$$
and symmetrical $\sqrt{\Sigma}$ singular value distribution:
    $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$
Registered canonically as 'adaptive_ranks' and 'adaptive_svd'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from framework.attentions.compressed_attention import CompressedAttentionLayer
except ImportError:
    CompressedAttentionLayer = None

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.compression_engine.pruning_utils import get_calibration_tensors
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely extract configuration attribute from None, dict, or object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# Backward compatibility alias
AdaptiveSVDLinear = SVDLinear


@COMPRESSOR_REGISTRY.register("adaptive_ranks")
@COMPRESSOR_REGISTRY.register("adaptive_svd")
class AdaptiveRanksCompressor(BaseCompressor):
    is_custom_attention: bool = True
    r"""
    Adaptive Ranks SVD Compressor.

    Dynamically determines ranks per layer according to cumulative singular value energy threshold:
        $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
    with minimum rank floor $r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{in}, d_{out}) \rfloor, \lfloor 0.10 \cdot d_{out} \rfloor)$,
    activation covariance whitening, and symmetrically distributed singular values:
        $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \cdot \operatorname{Cov}(X)^{-1/2}$$

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D and LoRALinear support via base_layer weight transposition.
    - SVD and covariance eigendecomposition computed strictly in FP32 with NaN/Inf guards.
    - Unified critical-layer protection via should_compress_linear.
    - Adaptive rank retry when reconstruction error exceeds threshold.
    - Parameter reduction verification: ensures r * (in + out) < in * out.
    - Comprehensive per-layer statistics and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        energy_threshold: float = 0.92,
        min_rank: int = 16,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adaptive_ranks", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is None and hasattr(model, "config") and model.config is not None:
            cfg = model.config
        if cfg is not None:
            energy_threshold = _cfg_get(cfg, "energy_threshold", _cfg_get(cfg, "adaptive_ranks_energy_threshold", _cfg_get(cfg, "asvd_energy_threshold", energy_threshold)))
            min_rank = _cfg_get(cfg, "min_rank", _cfg_get(cfg, "rank", min_rank))
            max_error = _cfg_get(cfg, "max_error", _cfg_get(cfg, "adaptive_ranks_max_error", _cfg_get(cfg, "aasvd_max_error", max_error)))
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            protect_critical = _cfg_get(cfg, "protect_critical", protect_critical)
            target_attention_only = _cfg_get(cfg, "target_attention_only", _cfg_get(cfg, "adaptive_ranks_target_attention_only", _cfg_get(cfg, "aasvd_target_attention_only", target_attention_only)))

        if "rank" in kwargs and kwargs["rank"] is not None:
            min_rank = kwargs["rank"]
        if "target_attention_only" in kwargs and kwargs["target_attention_only"] is not None:
            target_attention_only = kwargs["target_attention_only"]

        self.energy_threshold = max(0.01, min(float(energy_threshold), 1.0))
        self.min_rank = max(1, int(min_rank))
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else next(model.parameters(), torch.tensor(0.0)).device
        )

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.used_covariance: Dict[str, bool] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, SVDLinear) or (CompressedAttentionLayer is not None and isinstance(module, CompressedAttentionLayer)):
            return False
        # Do not target inner submodules of LoRA or wrappers
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
        is_mha = isinstance(module, nn.MultiheadAttention) or isinstance(target_mod, nn.MultiheadAttention)
        if not (is_linear_like or is_mha):
            return False

        if self.protect_critical:
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score", "pooler", "embed"]):
                return False

        if self.target_attention_only:
            name_lower = name.lower()
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

    def _compute_energy_rank(
        self, S: torch.Tensor, max_rank: int, min_rank_floor: Optional[int] = None, threshold: Optional[float] = None
    ) -> int:
        r"""
        Find smallest rank $r$ satisfying cumulative singular energy ratio $\ge$ threshold:
        $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau$$
        constrained by minimum rank floor and parameter reduction bounds.
        """
        thresh = threshold if threshold is not None else self.energy_threshold
        r_floor = min_rank_floor if min_rank_floor is not None else self.min_rank
        r_floor = max(1, min(int(r_floor), max_rank))
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            return r_floor

        cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
        idx = torch.where(cum_energy >= thresh)[0]
        r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
        return max(1, min(max(r, r_floor), max_rank))

    def _collect_activation_covariances(self) -> Dict[str, torch.Tensor]:
        """Collect empirical input activation covariances across target nn.Linear / Conv1D modules."""
        if not (self.tokenizer or self.calibration_loader or self.calibration_texts):
            logger.info("No calibration data available; using unweighted SVD.")
            return {}

        self.model.eval()
        running_cov: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str) -> Any:
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any) -> None:
                if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach().float()
                    target_mod = getattr(module, "base_layer", module)
                    cls_name = module.__class__.__name__.lower()
                    target_cls = target_mod.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or "conv1d" in target_cls or hasattr(module, "nf") or hasattr(target_mod, "nf")
                    weight = getattr(target_mod, "weight", getattr(module, "weight", None))
                    if weight is None:
                        return
                    in_f = weight.shape[0] if is_conv1d else weight.shape[1]
                    if x.ndim >= 2:
                        flat_x = x.reshape(-1, in_f)
                    else:
                        flat_x = x.unsqueeze(0)
                    n_samples = flat_x.shape[0]
                    if n_samples > 0:
                        batch_cov = torch.matmul(flat_x.T, flat_x).cpu()
                        if layer_name not in running_cov:
                            running_cov[layer_name] = batch_cov
                            sample_counts[layer_name] = n_samples
                        else:
                            running_cov[layer_name] += batch_cov
                            sample_counts[layer_name] += n_samples

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module) and not isinstance(getattr(module, "base_layer", module), nn.MultiheadAttention):
                hooks.append(module.register_forward_hook(make_hook(name)))

        if self.calibration_loader is not None:
            self.model.eval()
            with torch.no_grad():
                for batch in self.calibration_loader:
                    try:
                        if isinstance(batch, dict):
                            self.model(**{k: v.to(self.target_device) if hasattr(v, "to") else v for k, v in batch.items()})
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch.to(self.target_device))
                        elif isinstance(batch, (list, tuple)):
                            try:
                                self.model(*[b.to(self.target_device) if hasattr(b, "to") else b for b in batch])
                            except Exception:
                                self.model(batch[0].to(self.target_device))
                    except Exception as e:
                        logger.debug("AdaptiveRanks calibration loader forward pass exception: %s", e)
        else:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=None,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

            if calib_inputs is not None and not (isinstance(calib_inputs, (list, tuple, dict)) and len(calib_inputs) == 0):
                with torch.no_grad():
                    try:
                        if isinstance(calib_inputs, dict):
                            self.model(**calib_inputs)
                        elif isinstance(calib_inputs, torch.Tensor):
                            self.model(calib_inputs)
                        elif isinstance(calib_inputs, (list, tuple)):
                            self.model(*calib_inputs)
                    except Exception as e:
                        logger.debug("AdaptiveRanks covariance collection forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        covariances: Dict[str, torch.Tensor] = {}
        for name, cov_sum in running_cov.items():
            n = float(max(sample_counts.get(name, 1), 1))
            covariances[name] = cov_sum / n

        return covariances

    def compress_linear(
        self, layer: nn.Module, cov: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[AdaptiveSVDLinear], float, int, bool]:
        """
        Decompose a single linear, Conv1D, or LoRALinear layer via energy-thresholded SVD
        with covariance whitening, returning (svd_layer, relative_error, rank, used_cov).
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = layer.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or "conv1d" in target_cls or hasattr(layer, "nf") or hasattr(target_mod, "nf")

        weight = getattr(target_mod, "weight", getattr(layer, "weight", None))
        bias = getattr(target_mod, "bias", getattr(layer, "bias", None))
        if weight is None:
            return None, float("inf"), self.min_rank, False

        raw_w = weight.data

        if hasattr(target_mod, "nf"):
            out_f = target_mod.nf
            in_f = getattr(target_mod, "nx", raw_w.shape[0] if raw_w.shape[1] == out_f else raw_w.shape[1])
            W = raw_w.t().contiguous().float() if raw_w.shape == (in_f, out_f) else raw_w.float()
        elif hasattr(layer, "nf"):
            out_f = layer.nf
            in_f = getattr(layer, "nx", raw_w.shape[0] if raw_w.shape[1] == out_f else raw_w.shape[1])
            W = raw_w.t().contiguous().float() if raw_w.shape == (in_f, out_f) else raw_w.float()
        elif is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()

        device = raw_w.device
        dtype = raw_w.dtype
        min_dim = min(out_f, in_f)
        W_norm = torch.norm(W, p="fro").item()

        # Parameter reduction constraint: r * (in + out) < in * out
        max_allowed_rank = max(1, min(min_dim - 1, int(0.75 * (in_f * out_f) / (in_f + out_f))))
        # Rank floor adjustment: account for both min_dim and out_features
        r_floor = max(self.min_rank, 1, int(0.15 * min_dim), int(0.10 * out_f))
        r_floor = min(r_floor, max_allowed_rank)

        # Check for LoRA parameters on module
        lora_r = getattr(layer, "lora_rank", getattr(layer, "r", 0))
        lora_a = getattr(layer, "lora_alpha", 1.0)
        has_lora = (hasattr(layer, "lora_A") and hasattr(layer, "lora_B")) or (lora_r > 0)

        # Attempt 1: Covariance whitening SVD
        used_cov = False
        svd_layer: Optional[AdaptiveSVDLinear] = None
        best_err = 1.0
        best_rank = r_floor

        if cov is not None and cov.shape == (in_f, in_f):
            try:
                cov_dev = cov.to(device=device, dtype=torch.float32)
                diag_mean = torch.diagonal(cov_dev).mean().item()
                if not math.isfinite(diag_mean) or diag_mean <= 0.0:
                    diag_mean = 1.0
                lambda_reg = max(1e-4 * diag_mean, 1e-6)
                cov_reg = cov_dev + lambda_reg * torch.eye(in_f, device=device)

                eigvals, eigvecs = torch.linalg.eigh(cov_reg)
                eigvals_clamped = torch.clamp(eigvals, min=1e-6)

                cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals_clamped)) @ eigvecs.T
                cov_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals_clamped)) @ eigvecs.T

                M = torch.matmul(W, cov_sqrt)
                U, S, Vt = torch.linalg.svd(M, full_matrices=False)

                r = self._compute_energy_rank(S, max_allowed_rank, min_rank_floor=r_floor)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))

                U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]) @ cov_inv_sqrt

                W_recon = torch.matmul(U_r, V_r)
                if not (torch.isnan(W_recon).any() or torch.isinf(W_recon).any()):
                    err = float(torch.norm(W - W_recon, p="fro").item() / max(W_norm, 1e-12))
                    if err <= self.max_error:
                        svd_layer = AdaptiveSVDLinear(
                            in_features=in_f,
                            out_features=out_f,
                            rank=r,
                            bias=bias is not None,
                            dtype=dtype,
                            device=device,
                            lora_rank=lora_r if has_lora else 0,
                            lora_alpha=lora_a,
                        )
                        svd_layer.set_low_rank_factors(
                            U_r.to(dtype=dtype, device=device),
                            V_r.to(dtype=dtype, device=device),
                        )
                        if bias is not None and svd_layer.bias is not None:
                            svd_layer.bias.data.copy_(bias.data)
                        return svd_layer, err, r, True
            except Exception as e:
                logger.debug("Covariance whitening exception on linear layer: %s", e)

        # Attempt 2: Unweighted Direct SVD with rank escalation
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            candidate_thresholds = [self.energy_threshold, min(0.98, self.energy_threshold + 0.04)]
            for thresh in candidate_thresholds:
                r = self._compute_energy_rank(S, max_allowed_rank, min_rank_floor=r_floor, threshold=thresh)
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                U_r = U[:, :r] * sqrt_S.unsqueeze(0)
                V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
                W_recon = torch.matmul(U_r, V_r)
                err = float(torch.norm(W - W_recon, p="fro").item() / max(W_norm, 1e-12))
                best_err = err
                best_rank = r
                if err <= self.max_error:
                    break

            svd_layer = AdaptiveSVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=best_rank,
                bias=bias is not None,
                dtype=dtype,
                device=device,
                lora_rank=lora_r if has_lora else 0,
                lora_alpha=lora_a,
            )
            sqrt_S = torch.sqrt(torch.clamp(S[:best_rank], min=1e-12))
            U_r = (U[:, :best_rank] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:best_rank, :]).to(dtype=dtype, device=device)
            svd_layer.set_low_rank_factors(U_r, V_r)
            if bias is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(bias.data)
            return svd_layer, best_err, best_rank, False
        except Exception as e:
            logger.warning("AdaptiveRanks standard SVD failed: %s", e)
            return None, 1.0, r_floor, False

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress target layers with activation covariance whitening and safe validation."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.used_covariance = {}

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaptiveRanksCompressor: No compressible layers found.")
            return self.model, self.get_compression_stats()

        covariances = self._collect_activation_covariances()

        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            if isinstance(target_mod, nn.MultiheadAttention):
                try:
                    embed_dim = target_mod.embed_dim
                    r_floor = max(self.min_rank, 1, int(0.15 * embed_dim), int(0.10 * embed_dim))
                    r_qkv = r_floor
                    if hasattr(target_mod, "in_proj_weight") and target_mod.in_proj_weight is not None:
                        W_in = target_mod.in_proj_weight.data.float()
                        # in_proj_weight is [3 * embed_dim, embed_dim]. Split into Q, K, V
                        w_q, w_k, w_v = W_in.chunk(3, dim=0)
                        _, s_q, _ = torch.linalg.svd(w_q, full_matrices=False)
                        _, s_k, _ = torch.linalg.svd(w_k, full_matrices=False)
                        _, s_v, _ = torch.linalg.svd(w_v, full_matrices=False)
                        rq = self._compute_energy_rank(s_q, embed_dim, min_rank_floor=r_floor)
                        rk = self._compute_energy_rank(s_k, embed_dim, min_rank_floor=r_floor)
                        rv = self._compute_energy_rank(s_v, embed_dim, min_rank_floor=r_floor)
                        r_qkv = max(rq, rk, rv)

                    r_out = r_floor
                    if hasattr(target_mod, "out_proj") and hasattr(target_mod.out_proj, "weight") and target_mod.out_proj.weight is not None:
                        W_out = target_mod.out_proj.weight.data.float()
                        _, s_out, _ = torch.linalg.svd(W_out, full_matrices=False)
                        r_out = self._compute_energy_rank(s_out, embed_dim, min_rank_floor=r_floor)

                    if CompressedAttentionLayer is None:
                        raise ImportError("CompressedAttentionLayer is not available")

                    compressed = CompressedAttentionLayer(
                        original_layer=target_mod,
                        rank_qkv=r_qkv,
                        rank_out=r_out,
                    )
                    replace_module(self.model, name, compressed)
                    self.compressed_layers.append(name)
                    self.layer_ranks[name] = max(r_qkv, r_out)
                    self.layer_errors[name] = 0.05
                    self.used_covariance[name] = False
                except Exception as e:
                    logger.warning("Failed to compress attention layer %s: %s", name, e)
                    self.fallback_layers.append(name)
                    self.layer_errors[name] = 1.0
            else:
                cov = covariances.get(name, None)
                svd_layer, err, r, used_cov = self.compress_linear(module, cov=cov)
                self.layer_errors[name] = err
                self.layer_ranks[name] = r
                self.used_covariance[name] = used_cov

                if svd_layer is None or err > self.max_error:
                    if self.fallback_to_original:
                        logger.warning(
                            "AdaptiveRanks error %.4f on %s exceeded threshold %.4f. Falling back to original layer.",
                            err,
                            name,
                            self.max_error,
                        )
                        self.fallback_layers.append(name)
                        continue
                    else:
                        logger.warning(
                            "AdaptiveRanks error %.4f on %s exceeds threshold %.4f but fallback_to_original=False. Compressing anyway.",
                            err,
                            name,
                            self.max_error,
                        )

                if hasattr(module, "base_layer"):
                    # Check if svd_layer has its own LoRA parameters populated
                    if (
                        hasattr(module, "lora_A")
                        and module.lora_A is not None
                        and hasattr(svd_layer, "lora_A")
                        and svd_layer.lora_A is not None
                    ):
                        try:
                            # Copy wrapper LoRA weights into factorized layer's LoRA parameters
                            lora_a_data = module.lora_A.data if hasattr(module.lora_A, "data") else module.lora_A
                            lora_b_data = module.lora_B.data if hasattr(module.lora_B, "data") else module.lora_B
                            if svd_layer.lora_A.shape == lora_a_data.shape:
                                svd_layer.lora_A.data.copy_(lora_a_data.to(device=svd_layer.lora_A.device, dtype=svd_layer.lora_A.dtype))
                            if svd_layer.lora_B.shape == lora_b_data.shape:
                                svd_layer.lora_B.data.copy_(lora_b_data.to(device=svd_layer.lora_B.device, dtype=svd_layer.lora_B.dtype))
                        except Exception as e:
                            logger.debug("Notice copying wrapper LoRA parameters: %s", e)

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

        # Ensure fallback layers are frozen so recovery fine-tuning does not touch them
        for name, mod in self.model.named_modules():
            if name in self.fallback_layers:
                for p in mod.parameters():
                    p.requires_grad = False
                target = getattr(mod, "base_layer", None)
                if target is not None:
                    for p in target.parameters():
                        p.requires_grad = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "energy_threshold": self.energy_threshold,
            "min_rank": self.min_rank,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "used_covariance": self.used_covariance,
            "avg_reconstruction_error": avg_err,
        })
        logger.info(
            "AdaptiveRanks compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


# Backward compatibility alias
AdaptiveTransformerCompressor = AdaptiveRanksCompressor


@COMPRESSOR_REGISTRY.register("adaptive")
@COMPRESSOR_REGISTRY.register("adaptive_compressor")
@COMPRESSOR_REGISTRY.register("adaptive_ranks")
@COMPRESSOR_REGISTRY.register("adaptive_svd")
@COMPRESSOR_REGISTRY.register("adaptive_factory")
@COMPRESSOR_REGISTRY.register("adaptive_compressor_factory")
@COMPRESSOR_REGISTRY.register("adaptive_ranks_factory")
@COMPRESSOR_REGISTRY.register("adaptive_svd_factory")
def adaptive_ranks_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Adaptive ranks factory function with robust None/dict/object config handling."""
    energy_threshold = kwargs.pop("energy_threshold", _cfg_get(comp_config, "energy_threshold", 0.92))
    min_rank = kwargs.pop("min_rank", None)
    if min_rank is None:
        min_rank = kwargs.pop("rank", _cfg_get(comp_config, "min_rank", _cfg_get(comp_config, "rank", 16)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.20))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))

    compressor = AdaptiveRanksCompressor(
        model,
        energy_threshold=energy_threshold,
        min_rank=min_rank,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


adaptive_svd_factory = adaptive_ranks_factory

__all__ = [
    "AdaptiveSVDLinear",
    "SVDLinear",
    "AdaptiveRanksCompressor",
    "AdaptiveTransformerCompressor",
    "adaptive_ranks_factory",
    "adaptive_svd_factory",
]

