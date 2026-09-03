"""
SVD Projection Attention – Each projection (Q, K, V, Out) is replaced by
a low‑rank SVDLinear with explicit weight matrix reconstruction: W = U @ V.
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


class SVDProjectionLinear(nn.Module):
    r"""Low-Rank Factorized Projection Layer with Dynamic Matrix Reconstruction.

    Stores orthogonal singular subspaces $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and
    $V \in \mathbb{R}^{r \times d_{\text{in}}}$, reconstructing the dense linear projection:
    $$W = U V \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$
    $$y = x W^\top + b = x (U V)^\top + b$$

    Computational Complexity:
        - Memory Footprint: $\mathcal{O}(r(d_{\text{in}} + d_{\text{out}}))$
        - Forward Time: $\mathcal{O}(B \cdot N \cdot d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}} \cdot r \cdot d_{\text{in}})$

    Tensor Dimensions:
        - Input $x$: $[*, d_{\text{in}}]$
        - Output $y$: $[*, d_{\text{out}}]$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.U: nn.Parameter = nn.Parameter(torch.empty(out_features, rank))
        self.V: nn.Parameter = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with Kaiming uniform distribution."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input using sequentially factorized low-rank projections.

        Args:
            x (torch.Tensor): Input activations of shape `[*, in_features]`.

        Returns:
            torch.Tensor: Output activations of shape `[*, out_features]`.
        """
        Vx = F.linear(x, self.V)
        return F.linear(Vx, self.U, self.bias), getattr(self, '_attn_weights', None)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        """Set precomputed U and V factors."""
        with torch.no_grad():
            r = min(self.rank, U.size(1), V.size(0))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :].copy_(V[:r, :].to(device=self.V.device, dtype=self.V.dtype))


@ATTENTION_REGISTRY.register("svd_proj_attn")
@ATTENTION_REGISTRY.register("svd_projection_attention")
@ATTENTION_REGISTRY.register("svd_projection")
class SVDProjectionAttention(nn.Module):
    r"""Multi-Head Attention using Factorized SVD Linear Projections.

    Constructs low-rank projection operators from singular subspaces with zero-shot
    initialization from pretrained checkpoints:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(X V_q^\top U_q^\top)(X V_k^\top U_k^\top)^\top}{\sqrt{d_k}} + M\right) (X V_v^\top U_v^\top)$$

    Computational Complexity:
        - Parameter Compression: $4 \cdot r(d + d_{\text{all}})$ vs $4 \cdot d \cdot d_{\text{all}}$
        - Attention: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$
    """

    def __init__(
        self,
        model_config: Any,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        actual_rank = getattr(
            model_config,
            "svd_rank",
            rank if rank is not None else getattr(model_config, "rank", 32),
        )
        self.num_heads: int = getattr(model_config, "num_attention_heads", 4)
        hidden_size: int = getattr(model_config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = hidden_size
        self.rank: int = actual_rank

        self.q_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
        )
        self.k_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
        )
        self.v_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
        )
        self.out_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.all_head_size,
            self.hidden_size,
            self.rank,
            bias=True,
        )

        dropout_p = getattr(
            model_config,
            "dropout",
            getattr(
                model_config,
                "hidden_dropout_prob",
                getattr(model_config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _decompose_and_set(
        self, target: SVDProjectionLinear, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            target.U.data.zero_()
            target.V.data.zero_()
            target.U.data[:, :r] = (U[:, :r] * sqrt_s.unsqueeze(0)).to(device=target.U.device, dtype=target.U.dtype)
            target.V.data[:r, :] = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(device=target.V.device, dtype=target.V.dtype)
            if target.bias is not None and bias is not None:
                target.bias.data.copy_(bias.to(device=target.bias.device, dtype=target.bias.dtype))

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
            ("o", self.out_proj),
        ]:
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SVD projection attention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Binary or additive mask.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output features and optional attention weights.
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q: torch.Tensor = (
            self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        )
        K: torch.Tensor = (
            self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        )
        V: torch.Tensor = (
            self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        )

        scale = D**-0.5
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask.to(scores.dtype)) * -10000.0

        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context: torch.Tensor = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output: torch.Tensor = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("svd_proj_attn_inject")
def inject_svd_proj_attn(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
    **kwargs) -> SVDProjectionAttention:
    """Inject SVD Projection Attention into target model layer."""
    rank: int = getattr(comp_config, "svd_rank", 32)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn: SVDProjectionAttention = SVDProjectionAttention(
        model_config, rank, dropout, old_attn
    )
    layer.attention = new_attn
    return new_attn
