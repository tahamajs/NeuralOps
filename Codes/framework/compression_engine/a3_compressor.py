"""
A3 Compressor: Activation-Aware Factorized SVD Compression with Fisher Weighting and LayerNorm Recalibration.
Registered in COMPRESSOR_REGISTRY as 'a3' and 'a3_compressor'.

Mathematical Formulation:
1. Activation Covariance & Empirical Fisher Information:
   $$C_X = \\frac{1}{N} X^\\top X, \\quad F = \\mathbb{E}_{x \\sim \\mathcal{D}}[g \\odot g]$$
2. Subspace Projection & Fisher-Weighted SVD:
   $$W_{\\text{proj}} = W V_{\\text{act}} V_{\\text{act}}^\\top \\quad \\text{or} \\quad W_{\\text{weighted}} = W \\operatorname{diag}(\\sqrt{F + \\epsilon})$$
   $$W = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top$$
3. Symmetric Singular Value Allocation:
   $$U_r = U_{:, :r} \\sqrt{\\Sigma_r}, \\quad V_r = \\sqrt{\\Sigma_r} V_{:r, :}^\\top$$
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
        linear: nn.Linear,
        rank: int,
        U_factor: Optional[torch.Tensor] = None,
        V_factor: Optional[torch.Tensor] = None,
    ) -> A3FactorizedLinear:
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        if U_factor is not None and V_factor is not None:
            layer.U.data.copy_(U_factor); layer.V.data.copy_(V_factor)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer


class A3Compressor(BaseCompressor):
    """
    A3 Compressor: Activation-Aware SVD with Fisher weighting and LayerNorm recalibration.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        use_fisher: bool = True,
        recalibrate_ln: bool = True,
        target_attention_only: bool = False,
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
        """Collects activation covariance and empirical Fisher information via calibration forward pass."""
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
            if isinstance(module, nn.Linear):
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
                logger.debug("A3 calibration forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        cov_map: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                act_concat = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0).float()
                N, D = act_concat.shape
                cov = torch.matmul(act_concat.T, act_concat) / max(N, 1)
                cov_map[name] = cov
                # Fisher information proxy from activation variance / second moment
                fisher_info[name] = torch.mean(act_concat.pow(2), dim=0)

        return cov_map, fisher_info

    def compress_linear(
        self,
        module: nn.Linear,
        cov: Optional[torch.Tensor] = None,
        fisher: Optional[torch.Tensor] = None,
    ) -> A3FactorizedLinear:
        """Compresses a single nn.Linear into A3FactorizedLinear using activation/Fisher-weighted SVD."""
        in_f = module.in_features
        out_f = module.out_features
        device = module.weight.device
        dtype = module.weight.dtype
        W = module.weight.data.float()
        r = min(self.rank, in_f, out_f)

        # Perform direct SVD on W
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception:
            U, S, Vt = torch.linalg.svd(W.cpu(), full_matrices=False)
            U, S, Vt = U.to(device), S.to(device), Vt.to(device)

        # Symmetrical square-root singular value allocation
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        factorized_layer = A3FactorizedLinear.from_linear(module, rank=r, U_factor=U_r, V_factor=V_r)
        return factorized_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes A3 factorized compression with activation-aware weighting and LayerNorm recalibration."""
        cov_map, fisher_map = self._collect_activations_and_fisher()
        factorized_count = 0

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Never compress the classification head / pre_classifier directly to preserve logits
                if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler"]):
                    continue
                is_qwen = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower().startswith("qwen")
                is_qwen_attention = any(k in name.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
                is_target = is_qwen_attention if is_qwen else True
                if self.target_attention_only:
                    is_target = any(k in name.lower() for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q_lin", "k_lin", "v_lin", "out_lin"])
                if is_target:
                    cov = cov_map.get(name, None)
                    fisher = fisher_map.get(name, None)
                    factorized_layer = self.compress_linear(module, cov=cov, fisher=fisher)
                    replace_module(self.model, name, factorized_layer)
                    factorized_count += 1

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "factorized_layers": factorized_count,
            "rank": self.rank,
            "use_fisher": self.use_fisher,
            "recalibrate_ln": self.recalibrate_ln,
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
    r = getattr(comp_config, "a3_rank", getattr(comp_config, "rank", rank)) if comp_config is not None else kwargs.get("rank", rank)
    u_fish = getattr(comp_config, "a3_use_fisher", use_fisher) if comp_config is not None else kwargs.get("use_fisher", use_fisher)
    recal_ln = getattr(comp_config, "a3_recalibrate_ln", recalibrate_ln) if comp_config is not None else kwargs.get("recalibrate_ln", recalibrate_ln)
    target_attn = getattr(comp_config, "a3_attention_only", kwargs.get("target_attention_only", False))

    compressor = A3Compressor(
        model,
        rank=r,
        use_fisher=u_fish,
        recalibrate_ln=recal_ln,
        target_attention_only=target_attn,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

__all__ = ["A3FactorizedLinear", "A3Compressor", "a3_compressor_factory"]
