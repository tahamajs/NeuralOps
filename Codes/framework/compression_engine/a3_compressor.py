"""
A3 Compressor: Activation-Aware Factorized SVD Compression with Fisher Weighting and LayerNorm Recalibration.
Registered in COMPRESSOR_REGISTRY as 'a3' and 'a3_compressor'.

Mathematical Formulation:
1. Activation Covariance & Empirical Fisher Information:
   $$C_X = \\frac{1}{N} X^\\top X, \\quad F = \\mathbb{E}_{x \\sim \\mathcal{D}}[g \\odot g]$$
   Proxy when gradient is omitted: activation second moment (energy) $F \\approx \\frac{1}{N} \\sum x^2$.
2. Activation / Fisher Weighted SVD:
   $$W_{\\text{weighted}} = W \\operatorname{diag}(\\sqrt{F + \\epsilon})$$
   $$W_{\\text{weighted}} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
   Unweighted low-rank factors:
   $$U_r = U_{:, :r} \\sqrt{\\Sigma_r}, \\quad V_r = \\sqrt{\\Sigma_r} V_{:r, :}^\\top \\operatorname{diag}(1 / \\sqrt{F + \\epsilon})$$
3. Reconstruction Validation:
   $$\\text{rel\\_error} = \\frac{\\|W - U_r V_r\\|_F}{\\|W\\|_F + 10^{-8}} \\le \\tau$$
   If relative error exceeds threshold $\\tau$ (default 0.20), the rank is increased adaptively or compression skipped.
4. Structural Module Replacement:
   $$y = (x V_r^\\top) U_r^\\top + b$$
5. LayerNorm Recalibration:
   $$\\gamma \\leftarrow \\gamma \\cdot \\frac{\\sigma_{\\text{target}}}{\\sigma_{\\text{current}}}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear, get_dynamic_compression_rank
from framework.compression_engine.pruning_utils import (
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


class A3FactorizedLinear(nn.Module):
    """
    True Low-Rank Factorized Linear Layer:
    y = (x @ V.T) @ U.T + b
    Achieves physical parameter reduction from d_in * d_out to r * (d_in + d_out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        # Factorized parameters: V (rank, in_features) and U (out_features, rank)
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for param in (self.V, self.U):
                if param.dtype in (torch.float16, torch.bfloat16):
                    tmp = torch.empty_like(param, dtype=torch.float32, device="cpu")
                    nn.init.orthogonal_(tmp)
                    param.copy_(tmp.to(device=param.device, dtype=param.dtype))
                else:
                    nn.init.orthogonal_(param)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.U, self.V)

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        """Sets the low-rank factor parameters."""
        with torch.no_grad():
            self.U.data.copy_(U_factor.to(device=self.U.device, dtype=self.U.dtype))
            self.V.data.copy_(V_factor.to(device=self.V.device, dtype=self.V.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two-stage linear transformation: (..., in_features) -> (..., rank) -> (..., out_features)
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int,
        U_factor: Optional[torch.Tensor] = None,
        V_factor: Optional[torch.Tensor] = None,
    ) -> A3FactorizedLinear:
        in_f = getattr(linear, "in_features", None)
        out_f = getattr(linear, "out_features", None)
        if in_f is None or out_f is None:
            w = getattr(linear, "weight", None)
            if w is not None and w.ndim == 2:
                # For Conv1D, weight is often (in_features, out_features)
                in_f, out_f = w.shape[0], w.shape[1]
            else:
                raise ValueError(f"Cannot determine in/out features for {type(linear)}")

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            bias=getattr(linear, "bias", None) is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        if U_factor is not None and V_factor is not None:
            layer.U.data.copy_(U_factor.to(device=layer.U.device, dtype=layer.U.dtype))
            layer.V.data.copy_(V_factor.to(device=layer.V.device, dtype=layer.V.dtype))
        if getattr(linear, "bias", None) is not None:
            layer.bias.data.copy_(linear.bias.data.to(device=layer.bias.device, dtype=layer.bias.dtype))
        return layer


class A3Compressor(BaseCompressor):
    """
    A3 Compressor: Activation-Aware SVD with Fisher weighting, LayerNorm recalibration,
    and relative reconstruction error validation.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        use_fisher: bool = True,
        recalibrate_ln: bool = True,
        target_attention_only: bool = False,
        max_error: float = 0.20,
        max_retries: int = 3,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="a3", **kwargs)
        self.rank = max(1, int(rank))
        self.use_fisher = use_fisher
        self.recalibrate_ln = recalibrate_ln
        self.target_attention_only = target_attention_only
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}

    def _collect_activations_and_fisher(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Collects activation covariance and empirical Fisher proxy (activation energy)
        via calibration forward pass.
        """
        if not self.use_fisher and not self.recalibrate_ln:
            return {}, {}

        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        fisher_info: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(x.cpu())
            return hook

        for name, module in self.model.named_modules():
            # Hook linear layers and Conv1D modules
            cls_name = module.__class__.__name__
            if isinstance(module, nn.Linear) or cls_name == "Conv1D":
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
                logger.debug("A3 calibration forward pass notice: %s", e)

        for h in hooks:
            h.remove()

        cov_map: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                try:
                    act_concat = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0).float()
                    N, D = act_concat.shape
                    cov = torch.matmul(act_concat.T, act_concat) / max(N, 1)
                    cov_map[name] = cov
                    # Note: When full backward loss is not passed, activation energy (second moment)
                    # serves as the standard empirical Fisher diagonal proxy: F ≈ E[x^2].
                    fisher_info[name] = torch.mean(act_concat.pow(2), dim=0)
                except Exception as e:
                    logger.debug("Failed to compute covariance/Fisher for %s: %s", name, e)

        return cov_map, fisher_info

    def compress_linear(
        self,
        module: nn.Module,
        name: str = "",
        cov: Optional[torch.Tensor] = None,
        fisher: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[A3FactorizedLinear], float, int]:
        """
        Compresses an nn.Linear or Conv1D module into A3FactorizedLinear using
        activation/Fisher-weighted SVD with reconstruction error validation.

        Returns:
            Tuple of (factorized_layer_or_None, rel_frobenius_error, final_rank)
        """
        cls_name = module.__class__.__name__
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name.lower()
            or (hasattr(module, "weight") and hasattr(module.weight, "ndim") and module.weight.ndim == 2 and not isinstance(module, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)))
        )
        if not is_linear_like:
            logger.warning("Unsupported layer type %s for A3 compression; skipping %s", cls_name, name)
            return None, 0.0, 0

        # Extract weights and orient as [out_features, in_features]
        device = module.weight.device
        dtype = module.weight.dtype
        raw_w = module.weight.data.float()

        if "conv1d" in cls_name.lower() or (raw_w.ndim == 2 and raw_w.shape[0] < raw_w.shape[1] and getattr(module, "in_features", None) == raw_w.shape[0]):
            W = raw_w.t()
            in_f = W.shape[1]
            out_f = W.shape[0]
        else:
            W = raw_w
            out_f, in_f = W.shape[0], W.shape[1]

        # Determine initial compression rank safely
        r = get_dynamic_compression_rank(
            in_f,
            out_f,
            model_or_family=self.model,
            default_rank=self.rank,
            comp_config=getattr(self, "config", None),
        )
        r = max(1, min(int(r), in_f, out_f))

        # 1. Compute diagonal weighting factor s (input dimension)
        scale = None
        if self.use_fisher and fisher is not None:
            f = fisher.float()
            if f.ndim == 1 and f.shape[0] == in_f:
                scale = torch.sqrt(f.clamp(min=1e-8) + 1e-6)
        elif cov is not None:
            diag_c = torch.diag(cov).float()
            if diag_c.ndim == 1 and diag_c.shape[0] == in_f:
                scale = torch.sqrt(diag_c.clamp(min=1e-8) + 1e-6)

        if scale is not None:
            # Normalize scale to unit RMS to prevent extreme condition numbers
            rms = torch.norm(scale) / math.sqrt(in_f) + 1e-8
            scale = (scale / rms).clamp(min=1e-3, max=1e3).to(W.device)
            W_w = W * scale.unsqueeze(0)
        else:
            logger.debug("No calibration data available for %s; falling back to unweighted SVD.", name)
            W_w = W

        # 2. SVD and progressive rank validation
        current_r = r
        best_layer_factors = None
        best_err = float("inf")
        best_r = current_r

        for attempt in range(self.max_retries + 1):
            try:
                U, S, Vt = torch.linalg.svd(W_w, full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(W_w.cpu(), full_matrices=False)
                U, S, Vt = U.to(W.device), S.to(W.device), Vt.to(W.device)

            sqrt_S = torch.sqrt(torch.clamp(S[:current_r], min=1e-12))
            U_r = U[:, :current_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:current_r, :]

            # Unweight factors to preserve original W space
            if scale is not None:
                V_r = V_r / scale.unsqueeze(0)

            # Compute relative Frobenius reconstruction error
            W_rec = torch.matmul(U_r, V_r)
            norm_orig = torch.linalg.norm(W).item()
            rel_err = (torch.linalg.norm(W - W_rec).item()) / (norm_orig + 1e-8)

            if rel_err < best_err:
                best_err = rel_err
                best_layer_factors = (U_r, V_r)
                best_r = current_r

            # If error is acceptable or rank has reached matrix rank, finish
            if rel_err <= self.max_error or current_r >= min(in_f, out_f):
                break

            # Incrementally increase rank for next retry towards min(in_f, out_f)
            remaining = max(1, self.max_retries - attempt)
            step = max(8, (min(in_f, out_f) - current_r) // remaining)
            current_r = min(current_r + step, min(in_f, out_f))

        # 3. Validation threshold check
        if best_err > self.max_error:
            logger.warning(
                "A3 compression for %s exceeded max_error (%0.3f > %0.3f) at rank %d.",
                name,
                best_err,
                self.max_error,
                best_r,
            )
            if getattr(self, "fallback_to_original", False):
                return None, best_err, best_r

        # 4. Construct factorized layer
        U_final, V_final = best_layer_factors
        factorized_layer = A3FactorizedLinear(
            in_features=in_f,
            out_features=out_f,
            rank=best_r,
            bias=module.bias is not None,
            dtype=dtype,
            device=device,
        )
        factorized_layer.U.data.copy_(U_final.to(dtype=dtype, device=device))
        factorized_layer.V.data.copy_(V_final.to(dtype=dtype, device=device))
        if module.bias is not None:
            factorized_layer.bias.data.copy_(module.bias.data.to(dtype=dtype, device=device))

        return factorized_layer, best_err, best_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Executes A3 factorized compression with activation-aware weighting,
        reconstruction validation, and LayerNorm recalibration.
        """
        cov_map, fisher_map = self._collect_activations_and_fisher()
        factorized_count = 0
        skipped_count = 0
        layer_stats: Dict[str, Any] = {}
        total_err = 0.0

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_linear_like = (
                isinstance(module, nn.Linear)
                or "conv1d" in cls_name.lower()
                or (hasattr(module, "weight") and hasattr(module.weight, "ndim") and module.weight.ndim == 2 and not isinstance(module, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)))
            )
            if not is_linear_like:
                continue

            # Never compress classification heads or pre-classifiers
            if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler"]):
                continue
            if not should_compress_linear(self.model, name):
                continue
            if self.target_attention_only and not any(
                k in name.lower()
                for k in [
                    "attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj",
                    "o_proj", "q_lin", "k_lin", "v_lin", "out_lin", "query", "key", "value", "c_attn", "c_proj"
                ]
            ):
                continue

            cov = cov_map.get(name, None)
            fisher = fisher_map.get(name, None)

            factorized_layer, rel_err, chosen_rank = self.compress_linear(
                module, name=name, cov=cov, fisher=fisher
            )

            if factorized_layer is not None:
                replace_module(self.model, name, factorized_layer)
                factorized_count += 1
                total_err += rel_err
                layer_stats[name] = {
                    "rank": chosen_rank,
                    "rel_error": rel_err,
                    "weighted": bool(fisher is not None or cov is not None),
                }
            else:
                skipped_count += 1

        # LayerNorm Recalibration - only execute if calibration data & tokenizer are present
        if self.recalibrate_ln:
            has_calib_data = (
                self.calibration_loader is not None
                or self.calibration_texts is not None
                or self.tokenizer is not None
            )
            if has_calib_data:
                try:
                    recalibrate_layernorms(
                        self.model,
                        tokenizer=self.tokenizer,
                        calibration_loader=self.calibration_loader,
                        calibration_texts=self.calibration_texts,
                        device=self.target_device,
                    )
                except Exception as e:
                    logger.warning("LayerNorm recalibration notice: %s", e)
            else:
                logger.info("LayerNorm recalibration skipped: no calibration inputs provided.")

        self._compressed_params = self._count_params(self.model)
        avg_err = (total_err / factorized_count) if factorized_count > 0 else 0.0

        stats = self.get_compression_stats()
        stats.update({
            "factorized_layers": factorized_count,
            "skipped_layers": skipped_count,
            "rank": self.rank,
            "use_fisher": self.use_fisher,
            "recalibrate_ln": self.recalibrate_ln,
            "avg_reconstruction_error": avg_err,
            "layer_stats": layer_stats,
            "method": "a3_activation_aware_factorized_svd",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("a3")
@COMPRESSOR_REGISTRY.register("a3_compressor")
def a3_compressor_factory(
    model: nn.Module,
    comp_config: Any = None,
    rank: int = 64,
    use_fisher: bool = True,
    recalibrate_ln: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """A3 compressor factory function."""
    # Robust configuration extraction
    if comp_config is not None:
        if isinstance(comp_config, dict):
            r = comp_config.get("a3_rank", comp_config.get("rank", rank))
            u_fish = comp_config.get("a3_use_fisher", comp_config.get("use_fisher", use_fisher))
            recal_ln = comp_config.get("a3_recalibrate_ln", comp_config.get("recalibrate_ln", recalibrate_ln))
            target_attn = comp_config.get("a3_attention_only", comp_config.get("target_attention_only", False))
            max_err = comp_config.get("max_error", kwargs.get("max_error", 0.20))
        else:
            r = getattr(comp_config, "a3_rank", getattr(comp_config, "rank", rank))
            u_fish = getattr(comp_config, "a3_use_fisher", getattr(comp_config, "use_fisher", use_fisher))
            recal_ln = getattr(comp_config, "a3_recalibrate_ln", getattr(comp_config, "recalibrate_ln", recalibrate_ln))
            target_attn = getattr(comp_config, "a3_attention_only", getattr(comp_config, "target_attention_only", False))
            max_err = getattr(comp_config, "max_error", kwargs.get("max_error", 0.20))
    else:
        r = kwargs.get("rank", rank)
        u_fish = kwargs.get("use_fisher", use_fisher)
        recal_ln = kwargs.get("recalibrate_ln", recalibrate_ln)
        target_attn = kwargs.get("target_attention_only", False)
        max_err = kwargs.get("max_error", 0.20)

    # Validate and clamp rank
    r = max(1, int(r))

    compressor = A3Compressor(
        model,
        rank=r,
        use_fisher=u_fish,
        recalibrate_ln=recal_ln,
        target_attention_only=target_attn,
        max_error=max_err,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["A3FactorizedLinear", "A3Compressor", "a3_compressor_factory"]
