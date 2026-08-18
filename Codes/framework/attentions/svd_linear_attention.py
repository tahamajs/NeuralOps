"""
SVD Factorized Linear Attention & Kernelized Linear Attention Modules.

Provides two distinct operational modes:
1. `ProjectionFactorizedAttention` (Exact Softmax):
   Factorizes projection matrices $W_Q, W_K, W_V, W_O$ into $U \\Sigma V^\\top$ while maintaining
   exact softmax attention scores.
2. `LinearKernelAttention` (True $\\mathcal{O}(N)$ Linear Complexity):
   Computes attention using non-negative kernel feature maps $\\phi(x) = \\text{elu}(x) + 1$:
   $$\\text{Attention}(Q, K, V) = \\frac{\\phi(Q) (\\phi(K)^\\top V)}{\\phi(Q) \\sum_j \\phi(k_j)^\\top}$$
   Eliminating quadratic sequence-length memory scaling.

Includes Stiefel manifold orthogonality regularization and frozen-base LoRA residual adapters.
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

logger = logging.getLogger(__name__)


class SVDLinear(nn.Module):
    r"""
    Low-Rank Factorized Linear Layer executing as:
    $$y = (x V^\\top) U^\\top + b$$
    
    Optionally supports freezing the SVD base and training a LoRA residual adapter:
    $$y = (x V_{\\text{frozen}}^\\top) U_{\\text{frozen}}^\\top + \\frac{\\alpha}{r_{\\text{lora}}} (x B_{\\text{lora}}^\\top) A_{\\text{lora}}^\\top + b$$
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

        # SVD Base Parameters
        self.V = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))

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

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.zeros_(self.lora_A)
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base SVD projection
        latent = F.linear(x, self.V)  # [..., rank]
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
            self.U.copy_(U_factor)
            self.V.copy_(V_factor)

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
@ATTENTION_REGISTRY.register("projection_factorized_attn")
class SVDLinearAttention(BaseAttention):
    r"""
    Multi-Head Attention with SVD-Factorized Projections and Optional True Linear-Complexity Kernel.
    
    Supports:
    - Mode "exact": Low-rank projections with exact $\mathcal{O}(N^2)$ softmax attention.
    - Mode "linear": Low-rank projections with true $\mathcal{O}(N)$ non-negative kernel attention:
      $$\text{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \sum_j \phi(k_j)^\top}$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        mode: str = "exact",
        freeze_svd: bool = False,
        lora_rank: Optional[int] = None,
        pretrained_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank=rank)
        self.num_heads = getattr(config, "num_attention_heads", 12)
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.head_dim = self.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank or max(16, self.hidden_size // 4)
        self.mode = mode.lower()
        self.dropout_rate = getattr(config, "attention_probs_dropout_prob", getattr(config, "dropout", 0.1))

        # Factorized projections for Q, K, V, and Output
        self.q_proj = SVDLinear(self.hidden_size, self.all_head_size, rank=self.rank, freeze_svd=freeze_svd, lora_rank=lora_rank)
        self.k_proj = SVDLinear(self.hidden_size, self.all_head_size, rank=self.rank, freeze_svd=freeze_svd, lora_rank=lora_rank)
        self.v_proj = SVDLinear(self.hidden_size, self.all_head_size, rank=self.rank, freeze_svd=freeze_svd, lora_rank=lora_rank)
        self.out_proj = SVDLinear(self.all_head_size, self.hidden_size, rank=self.rank, freeze_svd=freeze_svd, lora_rank=lora_rank)

        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

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
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=weight.device, dtype=weight.dtype)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=weight.device, dtype=weight.dtype)
            target_linear.set_low_rank_factors(U_r, V_r)
            if bias is not None and target_linear.bias is not None:
                target_linear.bias.data.copy_(bias.to(target_linear.bias.device, target_linear.bias.dtype))

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, H * D] -> [B, H, N, D]
        b, n, _ = x.size()
        return x.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _forward_linear_kernel(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        FIX: True O(N) Linear Attention Kernel via ELU+1 feature map:
        Associative scan computation: (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ sum(phi(K))^T)
        Eliminates the quadratic O(N^2) memory footprint and sequence length wall.
        """
        # Feature maps: phi(x) = elu(x) + 1.0 (guarantees non-negative affinity)
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) if mask.dim() == 3 else mask
            phi_k = phi_k * mask_expanded.to(phi_k.dtype)

        # Associative state matrix: KV = sum_j (phi(k_j)^T @ v_j) -> shape [B, H, D, D]
        kv_state = torch.matmul(phi_k.transpose(-1, -2), v)  # [B, H, D_k, D_v]

        # Numerator: phi(q) @ KV -> shape [B, H, N, D]
        numerator = torch.matmul(phi_q, kv_state)

        # Denominator: phi(q) @ sum_j phi(k_j)^T -> shape [B, H, N, 1]
        k_sum = phi_k.sum(dim=-2, keepdim=True)  # [B, H, 1, D]
        denominator = torch.matmul(phi_q, k_sum.transpose(-1, -2)) + 1e-6

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
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Low-rank factorized projections
        q = self._transpose_for_scores(self.q_proj(hidden_states))
        k = self._transpose_for_scores(self.k_proj(hidden_states))
        v = self._transpose_for_scores(self.v_proj(hidden_states))

        # 2. Attention computation based on mode
        if self.mode == "linear":
            context = self._forward_linear_kernel(q, k, v, attention_mask)
            attn_probs = None
        else:
            context, attn_probs = self._forward_exact_softmax(q, k, v, attention_mask)

        # 3. Head merge and factorized output projection
        b, h, n, d = context.size()
        context = context.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        output = self.out_proj(context)

        return output, (attn_probs if output_attentions else None)

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Aggregate Stiefel penalty across all projection submodules."""
        return (
            self.q_proj.get_orthogonality_loss()
            + self.k_proj.get_orthogonality_loss()
            + self.v_proj.get_orthogonality_loss()
            + self.out_proj.get_orthogonality_loss()
        )
