"""
SVD Factorized Linear Attention & Kernelized Linear Attention Modules with Exponential Decay Gating.

Provides three distinct operational modes:
1. `ProjectionFactorizedAttention` (Exact Softmax):
   Factorizes projection matrices $W_Q, W_K, W_V, W_O$ into $U \\Sigma V^\\top$ while maintaining
   exact $\\mathcal{O}(N^2)$ softmax attention scores.
2. `LinearKernelAttention` (Standard $\\mathcal{O}(N)$ Non-Decayed Linear Complexity):
   Computes attention using non-negative kernel feature maps $\\phi(x) = \\text{elu}(x) + 1$:
   $$\\text{Attention}(Q, K, V) = \\frac{\\phi(Q) (\\phi(K)^\\top V)}{\\phi(Q) \\sum_j \\phi(k_j)^\\top + \\epsilon}$$
3. `DecayGatedLinearAttention` (State-of-the-Art $\\mathcal{O}(N)$ Data-Dependent Exponential Decay Recurrence):
   Prevents associative memory saturation and catastrophic retrieval degradation ("Needle-in-a-Haystack")
   on ultra-long contexts (>32k tokens) via data-dependent exponential gating:
   $$\\lambda_t = \\exp(-\\text{softplus}(\\text{decay\\_gate}(x_t))) \\in (0, 1)$$
   $$S_t = \\lambda_t S_{t-1} + \\phi(K_t)^\\top V_t \\in \\mathbb{R}^{d_k \\times d_v}$$
   $$Z_t = \\lambda_t Z_{t-1} + \\phi(K_t)^\\top \\in \\mathbb{R}^{d_k \\times 1}$$
   $$Y_t = \\frac{\\phi(Q_t) S_t}{\\phi(Q_t) Z_t + \\epsilon} \\in \\mathbb{R}^{1 \\times d_v}$$

Includes Stiefel manifold orthogonality regularization, frozen-base LoRA residual adapters,
fused kernel acceleration, and single-step autoregressive state recurrence caching.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear
    HAS_FUSED_KERNEL = True
except ImportError:
    HAS_FUSED_KERNEL = False

logger = logging.getLogger(__name__)


def compute_exponential_decay(
    decay_logits: torch.Tensor, activation: str = "softplus"
) -> torch.Tensor:
    r"""
    RESEARCH FIX: Computes strictly bounded, numerically stable retention factor $\lambda_t \in (0, 1)$.

    Mathematical Formulations:
    - "softplus" (default, provably stable):
      $$\lambda_t = \exp(-\text{softplus}(\text{decay\_logits})) \in (0, 1)$$
      Guarantees $\lambda_t \le 1.0$ unconditionally since $\text{softplus}(u) > 0$,
      preventing recurrent state explosion while preserving fine-grained decay gradients.
    - "sigmoid":
      $$\lambda_t = \sigma(\text{decay\_logits}) \in (0, 1)$$
    """
    if activation == "softplus":
        # RESEARCH FIX: softplus is strictly non-negative; negating guarantees exp result <= 1.0.
        # Clamping to [1e-5, 1.0] prevents float underflow to 0.0 while preserving decay dynamics.
        decay = torch.exp(-F.softplus(decay_logits))
        return torch.clamp(decay, min=1e-5, max=1.0)
    elif activation == "sigmoid":
        decay = torch.sigmoid(decay_logits)
        return torch.clamp(decay, min=1e-5, max=1.0 - 1e-5)
    else:
        raise ValueError(
            f"Unsupported decay activation '{activation}'. Must be 'softplus' or 'sigmoid'."
        )


class SVDLinear(nn.Module):
    r"""
    Low-Rank Factorized Linear Layer executing as:
    $$y = (x V^\top) U^\top + b$$
    
    Optionally supports freezing the SVD base and training a LoRA residual adapter:
    $$y = (x V_{\text{frozen}}^\top) U_{\text{frozen}}^\top + \frac{\alpha}{r_{\text{lora}}} (x B_{\text{lora}}^\top) A_{\text{lora}}^\top + b$$

    Also supports accelerated fused SVD kernel dispatch with optional in-SRAM RMSNorm.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        freeze_svd: bool = False,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 16.0,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.freeze_svd = freeze_svd
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_fused_kernel = use_fused_kernel and HAS_FUSED_KERNEL
        self.use_rms_norm = use_rms_norm
        self.eps = eps

        # SVD Base Parameters
        self.V = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))

        if use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(rank, dtype=dtype, device=device))
        else:
            self.register_parameter("norm_weight", None)

        if freeze_svd:
            self.U.requires_grad = False
            self.V.requires_grad = False

        # Optional LoRA Residual Adapter to prevent basis drift
        if lora_rank is not None and lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(out_features, lora_rank, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.empty(lora_rank, in_features, dtype=dtype, device=device))
            self.lora_scaling = lora_alpha / lora_rank
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_scaling = 1.0

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W = torch.matmul(self.U, self.V)
        if self.lora_A is not None and self.lora_B is not None:
            W = W + torch.matmul(self.lora_A, self.lora_B) * self.lora_scaling
        return W

    def reset_parameters(self) -> None:
        # orthogonal_ uses QR decomposition which is not supported for FP16/BF16
        # on CPU. Upcast to float32 for init, then cast back.
        for param in (self.U, self.V):
            orig_dtype = param.data.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                param.data = param.data.float()
                nn.init.orthogonal_(param)
                param.data = param.data.to(orig_dtype)
            else:
                nn.init.orthogonal_(param)
        if self.norm_weight is not None:
            nn.init.ones_(self.norm_weight)
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.zeros_(self.lora_A)
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RESEARCH FIX: Dispatches fused kernel when accelerated mode is active
        if self.use_fused_kernel and HAS_FUSED_KERNEL:
            has_lora = self.lora_A is not None and self.lora_B is not None
            base_bias = None if has_lora else self.bias
            out = fused_svd_linear(
                x=x,
                V=self.V,
                U=self.U,
                norm_weight=self.norm_weight if self.use_rms_norm else None,
                bias=base_bias,
                eps=self.eps,
                use_rms_norm=self.use_rms_norm,
            )
            if has_lora:
                lora_latent = F.linear(x, self.lora_B)
                lora_out = F.linear(lora_latent, self.lora_A) * self.lora_scaling
                out = out + lora_out
                if self.bias is not None:
                    out = out + self.bias
            return out

        # Sequential standard projection
        latent = F.linear(x, self.V)  # [..., rank]
        if self.use_rms_norm:
            if latent.dtype in (torch.float16, torch.bfloat16):
                rms = torch.rsqrt(torch.mean(latent.float() ** 2, dim=-1, keepdim=True) + self.eps).to(latent.dtype)
            else:
                rms = torch.rsqrt(torch.mean(latent ** 2, dim=-1, keepdim=True) + self.eps)
            latent = latent * rms
            if self.norm_weight is not None:
                latent = latent * self.norm_weight
        out = F.linear(latent, self.U)  # [..., out_features]

        # Add LoRA adapter if active
        if self.lora_A is not None and self.lora_B is not None:
            lora_latent = F.linear(x, self.lora_B)
            lora_out = F.linear(lora_latent, self.lora_A) * self.lora_scaling
            out = out + lora_out

        if self.bias is not None:
            out = out + self.bias
        return out

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        with torch.no_grad():
            r = min(self.rank, U_factor.size(1), V_factor.size(0))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U_factor[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :].copy_(V_factor[:r, :].to(device=self.V.device, dtype=self.V.dtype))

    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        FIX: Computes Stiefel manifold penalty ||U^T U - I||_F^2 + ||V V^T - I||_F^2
        to prevent singular subspace collapse during fine-tuning.
        """
        if self.freeze_svd:
            return torch.tensor(0.0, device=self.U.device)
        u_gram = torch.matmul(self.U.T, self.U)
        v_gram = torch.matmul(self.V, self.V.T)
        eye_r = torch.eye(self.rank, device=self.U.device)
        loss_u = torch.norm(u_gram - eye_r, p="fro") ** 2
        loss_v = torch.norm(v_gram - eye_r, p="fro") ** 2
        return (loss_u + loss_v) / (self.rank * self.rank)


@ATTENTION_REGISTRY.register("svd_linear_attn")
@ATTENTION_REGISTRY.register("svd_linear")
@ATTENTION_REGISTRY.register("svd_linear_attention")
@ATTENTION_REGISTRY.register("svd_attention")
# alias
# alias
# alias
# alias
class SVDLinearAttention(BaseAttention):
    r"""
    Multi-Head Attention with SVD-Factorized Projections, Kernel Linear Attention,
    and Data-Dependent Exponential Decay Gating Recurrence.
    
    Supports:
    - Mode "exact": Low-rank projections with exact $\mathcal{O}(N^2)$ softmax attention.
    - Mode "linear" (with `use_decay_gating=False`): Low-rank projections with standard non-decayed
      $\mathcal{O}(N)$ non-negative kernel linear attention:
      $$\text{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \sum_j \phi(k_j)^\top + \epsilon}$$
    - Mode "linear" or "decay" (with `use_decay_gating=True`): Low-rank projections with data-dependent
      exponential decay gated linear attention preventing associative state saturation on sequences > 32k:
      $$\lambda_t = \exp(-\text{softplus}(\text{decay\_gate}(x_t)))$$
      $$S_t = \lambda_t S_{t-1} + \phi(K_t)^\top V_t$$
      $$Z_t = \lambda_t Z_{t-1} + \phi(K_t)^\top$$
      $$Y_t = \frac{\phi(Q_t) S_t}{\phi(Q_t) Z_t + \epsilon}$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        mode: str = "exact",
        freeze_svd: bool = False,
        lora_rank: Optional[int] = None,
        pretrained_attn: Optional[Any] = None,
        use_decay_gating: bool = False,
        decay_activation: str = "softplus",
        decay_channel_wise: bool = False,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank=rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 4)
        self.hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = self.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = rank or max(16, self.hidden_size // 4)
        self.mode: str = mode.lower()
        self.dropout_rate: float = getattr(
            config, "attention_probs_dropout_prob", getattr(config, "dropout", 0.1)
        )
        self.eps: float = getattr(config, "linear_attention_eps", 1e-6)

        # RESEARCH FIX: Configuration-driven decay gating activation and backward compatibility flags
        self.use_decay_gating: bool = (
            use_decay_gating
            or getattr(config, "use_decay_gating", False)
            or (self.mode in ["decay_linear", "linear_decay", "decay", "gated_linear"])
        )
        self.decay_activation: str = getattr(
            config, "decay_activation", decay_activation
        )
        self.decay_channel_wise: bool = getattr(
            config, "decay_channel_wise", decay_channel_wise
        )
        self.use_fused_kernel: bool = (
            use_fused_kernel or getattr(config, "use_fused_kernel", False)
        )
        self.use_rms_norm: bool = (
            use_rms_norm or getattr(config, "use_rms_norm", False)
        )

        # Factorized projections for Q, K, V, and Output
        # RESEARCH FIX: Integrated FusedSVDLinear accelerated projection mode
        self.q_proj = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=lora_rank,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.k_proj = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=lora_rank,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.v_proj = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=lora_rank,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.out_proj = SVDLinear(
            self.all_head_size,
            self.hidden_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=lora_rank,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )

        # RESEARCH FIX: Data-dependent exponential decay gating projection.
        # Projects token representations into per-head (or per-channel) decay logits.
        # Initialized with multi-scale geometric decay horizons to maintain both short- and long-range memory.
        gate_out_dim = self.all_head_size if self.decay_channel_wise else self.num_heads
        self.decay_gate = nn.Linear(self.hidden_size, gate_out_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_decay_gate()

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def gate_proj(self) -> nn.Linear:
        """Alias for decay_gate to support legacy naming."""
        return self.decay_gate

    def _init_decay_gate(self) -> None:
        """
        RESEARCH FIX: Initialize decay gate weights to zero and bias across geometric decay spectrum.
        Weight = 0 ensures stable cold-start baseline retention schedule.
        Bias is initialized such that heads span retention rates lambda in [0.85, 0.999],
        enabling simultaneous local fine-grained attention and >32k long-term retention.
        """
        nn.init.zeros_(self.decay_gate.weight)
        with torch.no_grad():
            if not self.decay_channel_wise:
                # Geometric spread of decay rates: log-spaced between lambda=0.85 and lambda=0.999
                # For softplus: lambda = exp(-softplus(b)) => softplus(b) = -ln(lambda)
                # If b < -2, softplus(b) ~ exp(b) => b = ln(-ln(lambda))
                lambdas = torch.linspace(0.85, 0.999, self.num_heads)
                target_gamma = -torch.log(lambdas)
                # Inverse softplus: b = log(exp(gamma) - 1)
                bias_init = torch.log(torch.expm1(torch.clamp(target_gamma, min=1e-5)))
                self.decay_gate.bias.copy_(bias_init)
            else:
                lambdas = torch.linspace(0.85, 0.999, self.num_heads)
                target_gamma = -torch.log(lambdas)
                bias_per_head = torch.log(torch.expm1(torch.clamp(target_gamma, min=1e-5)))
                bias_expanded = bias_per_head.repeat_interleave(self.head_dim)
                self.decay_gate.bias.copy_(bias_expanded)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        """
        FIX: Copies and symmetrically factorizes pretrained attention weights into SVD projections.
        """
        from framework.core.attention_utils import (
            FUSED_QKV_REGEX,
            PROJ_REGEX_PATTERNS,
            _find_submodule_by_regex,
            _slice_fused_qkv_weights,
        )

        # 1. Check for unified fused QKV layer
        fused_mod = None
        for child_name, child_mod in pretrained_attn.named_children():
            if FUSED_QKV_REGEX.match(child_name) and hasattr(child_mod, "weight"):
                fused_mod = child_mod
                break

        if fused_mod is not None:
            sliced = _slice_fused_qkv_weights(
                fused_mod.weight.data,
                fused_mod.bias.data if hasattr(fused_mod, "bias") and fused_mod.bias is not None else None,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)]:
                w, b = sliced[proj_key]
                self._factorize_and_set(target, w, b)

            # Output projection
            target_o = _find_submodule_by_regex(pretrained_attn, PROJ_REGEX_PATTERNS["o"])
            if target_o is not None and hasattr(target_o, "weight"):
                self._factorize_and_set(
                    self.out_proj,
                    target_o.weight.data,
                    target_o.bias.data if hasattr(target_o, "bias") and target_o.bias is not None else None,
                )
        else:
            # 2. Decoupled Q, K, V, O projections
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                sub = _find_submodule_by_regex(pretrained_attn, PROJ_REGEX_PATTERNS[proj_key])
                if sub is not None and hasattr(sub, "weight"):
                    w = sub.weight.data
                    b = sub.bias.data if hasattr(sub, "bias") and sub.bias is not None else None
                    self._factorize_and_set(target, w, b)

    def _factorize_and_set(
        self, target_linear: SVDLinear, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        """Symmetrically factorizes weight matrix W -> U @ diag(sqrt(S)) and diag(sqrt(S)) @ V^T."""
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(weight.detach().cpu().float(), full_matrices=False)
            r = min(target_linear.rank, U.size(1), Vt.size(0))
            # FIX: Symmetrical Square-Root Singular Energy Allocation: U_r = U @ sqrt(S), V_r = sqrt(S) @ V^T
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-5))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=weight.device, dtype=weight.dtype)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=weight.device, dtype=weight.dtype)
            target_linear.set_low_rank_factors(U_r, V_r)
            if bias is not None and target_linear.bias is not None:
                target_linear.bias.data.copy_(bias.to(target_linear.bias.device, target_linear.bias.dtype))

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, H * D] -> [B, H, N, D]
        b, n, _ = x.size()
        return x.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _forward_decay_gated_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        RESEARCH FIX: Data-Dependent Exponential Decay Recurrent Linear Attention Kernel.

        Solves associative memory saturation ("Needle-in-a-Haystack" retrieval cliff) on >32k contexts.
        Computes sequential associative state recurrence:
        $$S_t = \lambda_t S_{t-1} + \phi(K_t)^\top V_t \in \mathbb{R}^{B \times H \times D_k \times D_v}$$
        $$Z_t = \lambda_t Z_{t-1} + \phi(K_t)^\top \in \mathbb{R}^{B \times H \times D_k \times 1}$$
        $$Y_t = \frac{\phi(Q_t) S_t}{\phi(Q_t) Z_t + \epsilon} \in \mathbb{R}^{B \times H \times 1 \times D_v}$$
        
        Args:
            q: Projected query activations $[B, H, N, D_k]$.
            k: Projected key activations $[B, H, N, D_k]$.
            v: Projected value activations $[B, H, N, D_v]$.
            hidden_states: Raw hidden token activations $[B, N, D_{\text{model}}]$ for data-dependent gating.
            mask: Optional attention mask $[B, N]$ or $[B, 1, 1, N]$ or $[B, H, N, 1]$.
            past_state: Optional prior recurrent state $(S_{t-1}, Z_{t-1})$ for streaming generation.

        Returns:
            Tuple of contextualized representation $[B, H, N, D_v]$ and final recurrent state $(S_N, Z_N)$.
        """
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        # 1. Feature Map: Non-negative kernel mapping guarantees bounded non-negative affinity
        phi_q = F.elu(q) + 1.0  # [B, H, N, D_k]
        phi_k = F.elu(k) + 1.0  # [B, H, N, D_k]

        # 2. Masking application
        if mask is not None:
            # RESEARCH FIX: Universal mask conversion for 2D [B, N], 3D [B, 1, N], and 4D [B, 1, 1, N] tensors
            if mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                mask_binary = (mask > -10000.0).to(phi_k.dtype) if (mask < 0).any() else mask.to(phi_k.dtype)
            else:
                mask_binary = mask.to(phi_k.dtype)

            if mask_binary.dim() == 2:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 3:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 4:
                if mask_binary.shape[-2] == 1 and mask_binary.shape[-1] == N:
                    mask_expanded = mask_binary.permute(0, 1, 3, 2)  # [B, 1, 1, N] -> [B, 1, N, 1]
                elif mask_binary.shape[-2] == N and mask_binary.shape[-1] == 1:
                    mask_expanded = mask_binary  # [B, 1, N, 1]
                elif mask_binary.shape[-2] == N and mask_binary.shape[-1] == N:
                    mask_expanded = mask_binary.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # [B, 1, N, 1]
                else:
                    mask_expanded = mask_binary.view(B, 1, N, 1)
            else:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            phi_k = phi_k * mask_expanded

        # 3. Compute Data-Dependent Decay Logits & Lambda Factors
        # decay_gate(hidden_states) -> [B, N, H] or [B, N, H * D]
        decay_logits = self.decay_gate(hidden_states)
        if self.decay_channel_wise:
            # [B, N, H * D] -> [B, N, H, D] -> [B, H, N, D]
            decay_logits = (
                decay_logits.view(B, N, H, D_k).permute(0, 2, 1, 3)
            )
            # lambda_decay: [B, H, N, D_k, 1]
            lambda_decay = compute_exponential_decay(
                decay_logits, activation=self.decay_activation
            ).unsqueeze(-1)
        else:
            # [B, N, H] -> [B, H, N, 1, 1]
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            # lambda_decay: [B, H, N, 1, 1]
            lambda_decay = compute_exponential_decay(
                decay_logits, activation=self.decay_activation
            )

        # 4. Initialize Recurrent Associative Memory State S and Normalizer State Z
        if past_state is not None:
            S_state, Z_state = past_state
            S = S_state.to(device=q.device, dtype=q.dtype)
            Z = Z_state.to(device=q.device, dtype=q.dtype)
        else:
            S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=q.dtype)
            Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=q.dtype)

        # 5. Associative Memory Recurrence Loop
        outputs = []
        for t in range(N):
            qt = phi_q[:, :, t : t + 1, :]  # [B, H, 1, D_k]
            kt = phi_k[:, :, t : t + 1, :]  # [B, H, 1, D_k]
            vt = v[:, :, t : t + 1, :]      # [B, H, 1, D_v]
            lam_t = lambda_decay[:, :, t]   # [B, H, 1, 1] or [B, H, D_k, 1]

            # RESEARCH FIX: Recurrent Associative State Update with Exponential Decay Retention
            # S_t = lambda_t * S_{t-1} + phi(K_t)^T @ V_t
            kt_t = kt.transpose(-1, -2)  # [B, H, D_k, 1]
            kv_t = torch.matmul(kt_t, vt)  # [B, H, D_k, D_v]
            S = lam_t * S + kv_t  # [B, H, D_k, D_v]

            # RESEARCH FIX: Recurrent Normalizer Denominator State Update
            # Z_t = lambda_t * Z_{t-1} + phi(K_t)^T
            Z = lam_t * Z + kt_t  # [B, H, D_k, 1]

            # RESEARCH FIX: Query retrieval & denominator normalization
            # num_t = phi(Q_t) @ S_t  [B, H, 1, D_v]
            # den_t = phi(Q_t) @ Z_t + eps  [B, H, 1, 1]
            num_t = torch.matmul(qt, S)  # [B, H, 1, D_v]
            den_t = torch.matmul(qt, Z) + self.eps  # [B, H, 1, 1]
            y_t = num_t / den_t  # [B, H, 1, D_v]
            outputs.append(y_t)

        context = torch.cat(outputs, dim=2)  # [B, H, N, D_v]
        return context, (S, Z)

    def _forward_linear_kernel(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        FIX: Standard True O(N) Non-Decayed Linear Attention Kernel via ELU+1 feature map:
        Associative scan computation: (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ sum(phi(K))^T + eps)
        Eliminates the quadratic O(N^2) memory footprint and sequence length wall.
        """
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        # Feature maps: phi(x) = elu(x) + 1.0 (guarantees non-negative affinity)
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        if mask is not None:
            if mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                mask_binary = (mask > -10000.0).to(phi_k.dtype) if (mask < 0).any() else mask.to(phi_k.dtype)
            else:
                mask_binary = mask.to(phi_k.dtype)

            if mask_binary.dim() == 2:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 3:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 4:
                if mask_binary.shape[-2] == 1 and mask_binary.shape[-1] == N:
                    mask_expanded = mask_binary.permute(0, 1, 3, 2)
                elif mask_binary.shape[-2] == N and mask_binary.shape[-1] == 1:
                    mask_expanded = mask_binary
                else:
                    mask_expanded = mask_binary.view(B, 1, N, 1)
            else:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            phi_k = phi_k * mask_expanded

        # Associative state matrix: KV = sum_j (phi(k_j)^T @ v_j) -> shape [B, H, D_k, D_v]
        kv_state = torch.matmul(phi_k.transpose(-1, -2), v)  # [B, H, D_k, D_v]

        # Numerator: phi(q) @ KV -> shape [B, H, N, D_v]
        numerator = torch.matmul(phi_q, kv_state)

        # Denominator: phi(q) @ sum_j phi(k_j)^T -> shape [B, H, N, 1]
        k_sum = phi_k.sum(dim=-2, keepdim=True)  # [B, H, 1, D_k]
        denominator = torch.matmul(phi_q, k_sum.transpose(-1, -2)) + self.eps

        context = numerator / denominator
        return context

    def _forward_exact_softmax(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact Softmax Attention with Low-Rank Projections."""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        if mask is not None:
            scores = apply_attention_mask(scores, mask)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        return context, probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        r"""
        Execute forward pass across configured mode:
        1. "exact" -> Exact $\mathcal{O}(N^2)$ Softmax Attention.
        2. "linear" with `use_decay_gating=False` -> Standard non-decayed $\mathcal{O}(N)$ Linear Kernel Attention.
        3. "linear" or "decay" with `use_decay_gating=True` -> Data-Dependent Exponential Decay Recurrent Attention.
        """
        query = kwargs.get("query", None)
        key = kwargs.get("key", None)
        mask = kwargs.get("mask", None)
        return_state = kwargs.get("return_state", False)
        past_state = kwargs.get("past_state", None)

        if hidden_states is None:
            hidden_states = query if query is not None else key
        if attention_mask is None:
            attention_mask = mask

        # 1. Low-rank factorized projections for Q, K, V
        q = self._transpose_for_scores(self.q_proj(hidden_states))
        k = self._transpose_for_scores(self.k_proj(hidden_states))
        v = self._transpose_for_scores(self.v_proj(hidden_states))

        updated_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # 2. Attention computation based on mode and decay gating configuration
        if self.mode == "exact":
            # Exact Softmax Mode
            context, attn_probs = self._forward_exact_softmax(q, k, v, attention_mask)
        elif self.use_decay_gating or self.mode in ["decay_linear", "linear_decay", "decay", "gated_linear"]:
            # RESEARCH FIX: Data-Dependent Exponential Decay Gated Linear Recurrence
            context, updated_state = self._forward_decay_gated_linear_kernel(
                q=q,
                k=k,
                v=v,
                hidden_states=hidden_states,
                mask=attention_mask,
                past_state=past_state,
            )
            attn_probs = None
        else:
            # Standard Non-Decayed Linear Kernel Attention Mode
            context = self._forward_linear_kernel(q, k, v, attention_mask)
            attn_probs = None

        # 3. Head merge and factorized output projection
        b, h, n, d = context.size()
        context = context.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        output = self.out_proj(context)

        attn_out = attn_probs if output_attentions else None

        if return_state and updated_state is not None:
            return output, attn_out, updated_state

        return output, attn_out

    def step_recurrent(
        self,
        x_t: torch.Tensor,
        past_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        RESEARCH FIX: Single-step $O(1)$ recurrent token generation for continuous streaming inference.
        
        Args:
            x_t: Single-token hidden activation $[B, 1, D_{\text{model}}]$ or $[B, D_{\text{model}}]$.
            past_state: Tuple of previous associative memory $(S_{t-1}, Z_{t-1})$.

        Returns:
            Tuple of output token activation $[B, 1, D_{\text{model}}]$ and updated state $(S_t, Z_t)$.
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)  # [B, 1, D]

        q = self._transpose_for_scores(self.q_proj(x_t))  # [B, H, 1, D_k]
        k = self._transpose_for_scores(self.k_proj(x_t))  # [B, H, 1, D_k]
        v = self._transpose_for_scores(self.v_proj(x_t))  # [B, H, 1, D_v]

        phi_q = F.elu(q) + 1.0  # [B, H, 1, D_k]
        phi_k = F.elu(k) + 1.0  # [B, H, 1, D_k]

        B, H, _, D_k = q.shape
        D_v = v.shape[-1]

        decay_logits = self.decay_gate(x_t)
        if self.decay_channel_wise:
            decay_logits = decay_logits.view(B, 1, H, D_k).permute(0, 2, 1, 3)
            lam_t = compute_exponential_decay(
                decay_logits, activation=self.decay_activation
            ).unsqueeze(-1)  # [B, H, 1, D_k, 1]
            lam_t = lam_t.squeeze(2)  # [B, H, D_k, 1]
        else:
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            lam_t = compute_exponential_decay(
                decay_logits, activation=self.decay_activation
            ).squeeze(2)  # [B, H, 1, 1]

        if past_state is not None:
            S_prev, Z_prev = past_state
        else:
            S_prev = torch.zeros(B, H, D_k, D_v, device=x_t.device, dtype=x_t.dtype)
            Z_prev = torch.zeros(B, H, D_k, 1, device=x_t.device, dtype=x_t.dtype)

        kt_t = phi_k.transpose(-1, -2)  # [B, H, D_k, 1]
        kv_t = torch.matmul(kt_t, v)     # [B, H, D_k, D_v]

        S_t = lam_t * S_prev + kv_t       # [B, H, D_k, D_v]
        Z_t = lam_t * Z_prev + kt_t       # [B, H, D_k, 1]

        num_t = torch.matmul(phi_q, S_t)  # [B, H, 1, D_v]
        den_t = torch.matmul(phi_q, Z_t) + self.eps  # [B, H, 1, 1]
        y_t = num_t / den_t  # [B, H, 1, D_v]

        context = y_t.permute(0, 2, 1, 3).contiguous().view(B, 1, H * D_v)
        out_t = self.out_proj(context)

        return out_t, (S_t, Z_t)

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Aggregate Stiefel penalty across all projection submodules."""
        base_loss = (
            self.q_proj.get_orthogonality_loss()
            + self.k_proj.get_orthogonality_loss()
            + self.v_proj.get_orthogonality_loss()
            + self.out_proj.get_orthogonality_loss()
        )
        if isinstance(self.decay_gate, SVDLinear):
            base_loss = base_loss + self.decay_gate.get_orthogonality_loss()
        return base_loss
