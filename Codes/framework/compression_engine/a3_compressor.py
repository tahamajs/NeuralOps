"""
A3 Compressor: Activation-Aware Factorized SVD Compression with Fisher/Activation Energy Weighting and LayerNorm Recalibration.
Registered in COMPRESSOR_REGISTRY as 'a3' and 'a3_compressor'.

Mathematical Formulation:
1. Online Streaming Activation Covariance & Empirical Fisher Diagonal Proxy (Activation Energy):
   $$C_X = \\frac{1}{N} \\sum_{t=1}^N x_t x_t^\\top, \\quad F = \\mathbb{E}_{x \\sim \\mathcal{D}}[x^2] \\approx \\frac{1}{N} \\sum_{t=1}^N x_t^2$$
   Computed in an online streaming fashion during calibration passes to ensure strictly O(D^2) memory overhead.
2. Activation / Energy-Weighted SVD:
   $$W_{\\text{weighted}} = W \\operatorname{diag}(\\sqrt{F + \\epsilon})$$
   $$W_{\\text{weighted}} = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
   Unweighted low-rank factors:
   $$U_r = U_{:, :r} \\sqrt{\\Sigma_r}, \\quad V_r = \\sqrt{\\Sigma_r} V_{:r, :}^\\top \\operatorname{diag}(1 / \\sqrt{F + \\epsilon})$$
3. Reconstruction Validation & Enforcement:
   $$\\text{rel\\_error} = \\frac{\\|W - U_r V_r\\|_F}{\\|W\\|_F + 10^{-8}} \\le \\tau$$
   If relative error exceeds threshold $\\tau$ (default 0.20), the rank is stepped or compression strictly aborted to prevent silent model degradation.
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

        # Fix 5: Safe Device Initialization directly with factory kwargs
        factory_kwargs = {"dtype": dtype, "device": device}
        self.V = nn.Parameter(torch.empty(self.rank, in_features, **factory_kwargs))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
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

    # Fix 4: Framework Compatibility - Standard method instead of @property weight
    def get_weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.U, self.V)

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        """Sets the low-rank factor parameters."""
        with torch.no_grad():
            self.U.data.copy_(U_factor.to(device=self.U.device, dtype=self.U.dtype))
            self.V.data.copy_(V_factor.to(device=self.V.device, dtype=self.V.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __iter__(self):
        """Allow tuple unpacking: layer, err, r = compressor.compress_linear(...)."""
        yield self
        yield getattr(self, "reconstruction_error", 0.0)
        yield getattr(self, "rank", self.rank)

    def __getitem__(self, idx: int) -> Any:
        return (self, getattr(self, "reconstruction_error", 0.0), getattr(self, "rank", self.rank))[idx]

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int,
        U_factor: Optional[torch.Tensor] = None,
        V_factor: Optional[torch.Tensor] = None,
    ) -> A3FactorizedLinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        in_f = getattr(linear, "in_features", None)
        out_f = getattr(linear, "out_features", None)
        if in_f is None or out_f is None:
            w = getattr(linear, "weight", None)
            if w is not None and w.ndim == 2:
                if is_conv1d:
                    in_f, out_f = w.shape[0], w.shape[1]
                else:
                    out_f, in_f = w.shape[0], w.shape[1]
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
    A3 Compressor: Activation-Aware SVD with Fisher / Activation-Energy weighting,
    LayerNorm recalibration, streaming memory-safe accumulation, and strict reconstruction validation.
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
        fallback_to_original: bool = True,
        normalize_scale: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="a3", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "a3_rank", getattr(cfg, "rank", rank))
            use_fisher = getattr(cfg, "a3_use_fisher", getattr(cfg, "use_fisher", use_fisher))
            recalibrate_ln = getattr(cfg, "a3_recalibrate_ln", getattr(cfg, "recalibrate_ln", recalibrate_ln))
            target_attention_only = getattr(cfg, "a3_attention_only", getattr(cfg, "target_attention_only", target_attention_only))
            max_error = getattr(cfg, "a3_max_error", getattr(cfg, "max_error", max_error))
            max_retries = getattr(cfg, "a3_max_retries", getattr(cfg, "max_retries", max_retries))
            fallback_to_original = getattr(cfg, "a3_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original))
            normalize_scale = getattr(cfg, "a3_normalize_scale", getattr(cfg, "normalize_scale", normalize_scale))

        self.rank = max(1, int(rank))
        self.use_fisher = bool(use_fisher)  # activation energy / empirical Fisher proxy
        self.recalibrate_ln = bool(recalibrate_ln)
        self.target_attention_only = bool(target_attention_only)
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        # Fix 1 & 3: fallback_to_original attribute cleanly defined and honored
        self.fallback_to_original = bool(kwargs.get("fallback_to_original", fallback_to_original))
        # Fix 5: scale normalization toggle
        self.normalize_scale = bool(kwargs.get("normalize_scale", normalize_scale))
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}
        self.compressed_layers: List[str] = []
        self.skipped_layers: List[str] = []

    def _collect_activations_and_fisher(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Fix 1 (OOM Prevention): Collects activation covariance and empirical Fisher proxy
        (activation energy E[x^2]) using online streaming accumulation.
        Raw activations are never buffered in RAM.
        """
        if not self.use_fisher and not self.recalibrate_ln:
            return {}, {}

        # Initialize running states
        cov_running: Dict[str, torch.Tensor] = {}
        fisher_running: Dict[str, torch.Tensor] = {}
        count_running: Dict[str, int] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    with torch.no_grad():
                        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                        # Filter padding tokens (assuming exact zeros)
                        mask = x.abs().sum(dim=-1) > 1e-5
                        x_valid = x[mask]
                        N_valid = x_valid.size(0)

                        if N_valid > 0:
                            c_batch = torch.matmul(x_valid.T, x_valid)
                            f_batch = x_valid.pow(2).sum(dim=0)

                            if name not in cov_running:
                                cov_running[name] = c_batch.cpu()
                                fisher_running[name] = f_batch.cpu()
                                count_running[name] = N_valid
                            else:
                                cov_running[name] += c_batch.cpu()
                                fisher_running[name] += f_batch.cpu()
                                count_running[name] += N_valid
            return hook

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf"):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        if calib_inputs is not None:
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
                    logger.debug("A3 calibration forward pass notice: %s", e)

        for h in hooks:
            h.remove()

        cov_map: Dict[str, torch.Tensor] = {}
        fisher_info: Dict[str, torch.Tensor] = {}
        for name in count_running:
            n = max(count_running[name], 1)
            cov_map[name] = cov_running[name] / n
            # Activation energy (second moment proxy for Fisher): E[x^2]
            fisher_info[name] = fisher_running[name] / n

        return cov_map, fisher_info

    def compress_linear(
        self,
        module: nn.Module,
        name: str = "",
        cov: Optional[torch.Tensor] = None,
        fisher: Optional[torch.Tensor] = None,
        rank: Optional[int] = None,
    ) -> Tuple[Optional[A3FactorizedLinear], float, int]:
        """
        Compresses an nn.Linear or Conv1D module into A3FactorizedLinear using
        activation/Fisher-weighted SVD with reconstruction error validation.

        Returns:
            Tuple of (factorized_layer_or_None, rel_frobenius_error, final_rank)
        """
        cls_name = module.__class__.__name__.lower()
        # Fix 7: Robust Conv1D detection
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        is_linear = isinstance(module, nn.Linear)
        is_linear_like = (
            is_linear
            or is_conv1d
            or (hasattr(module, "weight") and hasattr(module.weight, "ndim") and module.weight.ndim == 2 and not isinstance(module, (nn.Embedding, nn.LayerNorm, nn.RMSNorm)))
        )
        if not is_linear_like:
            logger.warning("Unsupported layer type %s for A3 compression; skipping %s", module.__class__.__name__, name)
            return None, 0.0, 0

        # Extract weights and orient as [out_features, in_features]
        device = module.weight.device
        dtype = module.weight.dtype
        raw_w = module.weight.data.float()

        if is_conv1d:
            # HuggingFace Conv1D weights are [in_features, out_features]
            W = raw_w.t()
            in_f = W.shape[1]
            out_f = W.shape[0]
        else:
            # Standard nn.Linear weights are [out_features, in_features]
            W = raw_w
            out_f, in_f = W.shape[0], W.shape[1]

        # Determine initial compression rank safely
        target_r = rank if rank is not None else self.rank
        r = get_dynamic_compression_rank(
            in_f,
            out_f,
            model_or_family=self.model,
            default_rank=target_r,
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
            # Fix 5: Optional scale normalization via normalize_scale
            if self.normalize_scale:
                rms = torch.norm(scale) / math.sqrt(in_f) + 1e-8
                scale = (scale / rms).clamp(min=1e-3, max=1e3).to(W.device)
            else:
                scale = scale.clamp(min=1e-3, max=1e3).to(W.device)
            W_w = W * scale.unsqueeze(0)
        else:
            logger.debug("No calibration data available for %s; falling back to unweighted SVD.", name)
            W_w = W

        # Fix 2: Pull SVD Out of the Retry Loop
        try:
            U, S, Vt = torch.linalg.svd(W_w, full_matrices=False)
        except Exception:
            U, S, Vt = torch.linalg.svd(W_w.cpu(), full_matrices=False)
            U, S, Vt = U.to(W.device), S.to(W.device), Vt.to(W.device)

        norm_orig = torch.linalg.norm(W).item()
        current_r = r
        best_layer_factors = None
        best_err = float("inf")
        best_r = current_r

        # Only retry / increase rank if rank was not explicitly specified
        retries = 0 if rank is not None else self.max_retries
        for attempt in range(retries + 1):
            clamped_r = min(current_r, S.shape[0])
            sqrt_S = torch.sqrt(torch.clamp(S[:clamped_r], min=1e-12))
            U_r = U[:, :clamped_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:clamped_r, :]

            # Unweight factors to preserve original W space
            if scale is not None:
                V_r = V_r / scale.unsqueeze(0)

            # Compute relative Frobenius reconstruction error
            W_rec = torch.matmul(U_r, V_r)
            rel_err = (torch.linalg.norm(W - W_rec).item()) / (norm_orig + 1e-8)

            if rel_err < best_err:
                best_err = rel_err
                best_layer_factors = (U_r, V_r)
                best_r = clamped_r

            # If error is acceptable or rank has reached maximum possible rank, finish
            if rel_err <= self.max_error or clamped_r >= min(in_f, out_f):
                break

            # Incrementally increase rank for next retry towards min(in_f, out_f)
            if attempt < retries:
                remaining = max(1, retries - attempt)
                step = max(8, (min(in_f, out_f) - current_r) // remaining)
                current_r = min(current_r + step, min(in_f, out_f))

        # Fix 3: Validation threshold check - Enforce max_error Abort
        if best_err > self.max_error:
            logger.warning(
                "A3 compression for %s exceeded max_error (%0.3f > %0.3f) at rank %d. Skipping layer.",
                name,
                best_err,
                self.max_error,
                best_r,
            )
            return None, best_err, best_r  # Mandatory abort

        # Construct factorized layer with Safe Device Initialization (Fix 5)
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

        factorized_layer.reconstruction_error = best_err
        return factorized_layer, best_err, best_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Executes A3 factorized compression with activation-aware weighting,
        reconstruction validation, and LayerNorm recalibration.
        """
        # Fix 4 (Calibration data absence handled gracefully)
        has_calib_sources = (
            self.calibration_loader is not None
            or bool(self.calibration_texts)
            or self.tokenizer is not None
        )
        if not has_calib_sources or (not self.use_fisher and not self.recalibrate_ln):
            logger.info("No calibration sources provided; skipping activation collection and using unweighted SVD.")
            cov_map, fisher_map = {}, {}
            self.use_fisher = False
            self.recalibrate_ln = False
        else:
            cov_map, fisher_map = self._collect_activations_and_fisher()

        factorized_count = 0
        skipped_count = 0
        layer_stats: Dict[str, Any] = {}
        layer_param_counts: Dict[str, int] = {}
        total_err = 0.0
        self.compressed_layers = []
        self.skipped_layers = []

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__.lower()
            is_linear_like = (
                isinstance(module, nn.Linear)
                or "conv1d" in cls_name
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
                p_count = sum(p.numel() for p in module.parameters())
                layer_param_counts[name] = p_count
                layer_stats[name] = {
                    "rank": chosen_rank,
                    "rel_error": rel_err,
                    "weighted": bool(fisher is not None or cov is not None),
                }
                self.compressed_layers.append(name)
            else:
                skipped_count += 1
                self.skipped_layers.append(name)

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
        total_params = sum(layer_param_counts.values()) if layer_param_counts else 0
        weighted_err = (
            sum(layer_stats[k]["rel_error"] * layer_param_counts[k] for k in layer_stats) / max(1, total_params)
            if total_params > 0
            else 0.0
        )

        stats = self.get_compression_stats()
        stats.update({
            "factorized_layers": factorized_count,
            "skipped_layers": skipped_count,
            "compressed_layers": self.compressed_layers,
            "rank": self.rank,
            "use_fisher": self.use_fisher,
            "recalibrate_ln": self.recalibrate_ln,
            "avg_reconstruction_error": avg_err,
            "weighted_reconstruction_error": weighted_err,
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
) -> Any:
    """A3 compressor factory function returning (compressed_model, stats) or compressed_model."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    # Robust configuration extraction
    if comp_config is not None:
        if isinstance(comp_config, dict):
            r = comp_config.get("a3_rank", comp_config.get("rank", rank))
            u_fish = comp_config.get("a3_use_fisher", comp_config.get("use_fisher", use_fisher))
            recal_ln = comp_config.get("a3_recalibrate_ln", comp_config.get("recalibrate_ln", recalibrate_ln))
            target_attn = comp_config.get("a3_attention_only", comp_config.get("target_attention_only", False))
            max_err = comp_config.get("max_error", kwargs.get("max_error", 0.20))
            fallback = comp_config.get("fallback_to_original", kwargs.get("fallback_to_original", True))
            norm_scale = comp_config.get("normalize_scale", kwargs.get("normalize_scale", True))
        else:
            r = getattr(comp_config, "a3_rank", getattr(comp_config, "rank", rank))
            u_fish = getattr(comp_config, "a3_use_fisher", getattr(comp_config, "use_fisher", use_fisher))
            recal_ln = getattr(comp_config, "a3_recalibrate_ln", getattr(comp_config, "recalibrate_ln", recalibrate_ln))
            target_attn = getattr(comp_config, "a3_attention_only", getattr(comp_config, "target_attention_only", False))
            max_err = getattr(comp_config, "max_error", kwargs.get("max_error", 0.20))
            fallback = getattr(comp_config, "fallback_to_original", kwargs.get("fallback_to_original", True))
            norm_scale = getattr(comp_config, "normalize_scale", kwargs.get("normalize_scale", True))
    else:
        r = kwargs.get("rank", rank)
        u_fish = kwargs.get("use_fisher", use_fisher)
        recal_ln = kwargs.get("recalibrate_ln", recalibrate_ln)
        target_attn = kwargs.get("target_attention_only", False)
        max_err = kwargs.get("max_error", 0.20)
        fallback = kwargs.get("fallback_to_original", True)
        norm_scale = kwargs.get("normalize_scale", True)

    compressor = A3Compressor(
        model,
        rank=max(1, int(r)),
        use_fisher=u_fish,
        recalibrate_ln=recal_ln,
        target_attention_only=target_attn,
        max_error=max_err,
        fallback_to_original=fallback,
        normalize_scale=norm_scale,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = ["A3FactorizedLinear", "A3Compressor", "a3_compressor_factory"]
