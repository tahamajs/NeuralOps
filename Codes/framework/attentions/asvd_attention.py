"""
ASVD (Activation‑aware Singular Value Decomposition) Attention.
Paper: https://arxiv.org/abs/2307.07382
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY


class ASVDLinear(nn.Module):
    r"""Low-rank factorized linear layer with explicit singular value scaling: $y = ((x V^\top) \odot \Sigma) U^\top + b$."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        factory_kwargs = {"dtype": dtype, "device": device}
        self.U = nn.Parameter(torch.empty(out_features, rank, **factory_kwargs))
        self.S = nn.Parameter(torch.empty(rank, **factory_kwargs))
        self.V = nn.Parameter(torch.empty(rank, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.U.dtype in (torch.float16, torch.bfloat16):
                u_fp32 = torch.empty_like(self.U, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(u_fp32)
                self.U.copy_(u_fp32.to(dtype=self.U.dtype, device=self.U.device))
            else:
                nn.init.orthogonal_(self.U)

            nn.init.ones_(self.S)

            if self.V.dtype in (torch.float16, torch.bfloat16):
                v_fp32 = torch.empty_like(self.V, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(v_fp32)
                self.V.copy_(v_fp32.to(dtype=self.V.dtype, device=self.V.device))
            else:
                nn.init.orthogonal_(self.V)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor, S: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            r = min(self.rank, U.size(1), V.size(0))
            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.zero_()
            self.U.data[:, :r].copy_(U[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :].copy_(V[:r, :].to(device=self.V.device, dtype=self.V.dtype))
            if S is not None:
                self.S.data[:r].copy_(S[:r].to(device=self.S.device, dtype=self.S.dtype))
            else:
                self.S.data[:r].fill_(1.0)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.U * self.S.unsqueeze(0), self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, in_features)
        # 1. x @ V.T -> (*, rank)
        Vx = F.linear(x, self.V)
        # 2. scale by S
        SVx = Vx * self.S
        # 3. SVx @ U.T -> (*, out_features)
        out = F.linear(SVx, self.U)
        if self.bias is not None:
            out = out + self.bias
        return out


@ATTENTION_REGISTRY.register("asvd")
@ATTENTION_REGISTRY.register("asvd_attention")
class ASVDAttention(nn.Module):
    r"""ASVD Attention with Factored Low-Rank Matrix Decomposition.

    Computational Complexity:
        - Factorized Projections: $\mathcal{O}(B \cdot N \cdot r \cdot (d + d_{\text{all}}))$
        - Attention Scoring: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = config.hidden_size
        self.rank: int = getattr(
            config,
            "asvd_rank",
            rank if rank is not None else getattr(config, "rank", 32),
        )

        self.query: ASVDLinear = ASVDLinear(self.hidden_size, self.all_head_size, self.rank, bias=True)
        self.key: ASVDLinear = ASVDLinear(self.hidden_size, self.all_head_size, self.rank, bias=True)
        self.value: ASVDLinear = ASVDLinear(self.hidden_size, self.all_head_size, self.rank, bias=True)
        self.out_proj: ASVDLinear = ASVDLinear(self.all_head_size, self.hidden_size, self.rank, bias=True)
        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _decompose_and_set(self, target_linear: ASVDLinear, src_weight: torch.Tensor, src_bias: Optional[torch.Tensor]) -> None:
        with torch.no_grad():
            W = src_weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            target_linear.U.data.zero_()
            target_linear.V.data.zero_()
            target_linear.S.data.zero_()
            target_linear.U.data[:, :r] = U[:, :r].to(device=target_linear.U.device, dtype=target_linear.U.dtype)
            target_linear.S.data[:r] = S[:r].to(device=target_linear.S.device, dtype=target_linear.S.dtype)
            target_linear.V.data[:r, :] = Vt[:r, :].to(device=target_linear.V.device, dtype=target_linear.V.dtype)
            if src_bias is not None and target_linear.bias is not None:
                target_linear.bias.data.copy_(src_bias.to(device=target_linear.bias.device, dtype=target_linear.bias.dtype))

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                self._decompose_and_set(target, w, b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for factorized ASVD attention."""
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        k: torch.Tensor = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        v: torch.Tensor = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        scale: float = d**-0.5
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = apply_attention_mask(scores, attention_mask)

        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context: torch.Tensor = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out: torch.Tensor = self.out_proj(context)

        return out, attn if output_attentions else None
