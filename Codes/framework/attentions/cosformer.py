r"""
Cosformer: Rethinking Softmax in Attention with Linear Complexity.
Paper: https://arxiv.org/abs/2202.08791 (ICLR 2022)
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("cosformer")
@ATTENTION_REGISTRY.register("cosformer_attention")
class CosformerAttention(BaseAttention):
    r"""
    Cosformer Attention Layer.

    Mathematical Formulation:
        1. Non-negative kernel mapping:
           $$\phi(x) = \operatorname{ReLU}(x) \quad \text{or} \quad \operatorname{ELU}(x) + 1$$
        
        2. Cosine-based positional reweighting:
           $$R(i, j) = \cos\left(\frac{\pi(i - j)}{2 M}\right) = \cos\left(\frac{\pi i}{2 M}\right)\cos\left(\frac{\pi j}{2 M}\right) + \sin\left(\frac{\pi i}{2 M}\right)\sin\left(\frac{\pi j}{2 M}\right)$$
           where $M$ is maximum sequence length. For $|i - j| \le M$, $\frac{\pi |i - j|}{2 M} \in [0, \pi/2]$, ensuring non-negativity.
        
        3. Kernel projection:
           $$Q^{\cos}_i = \phi(Q_i) \cos\left(\frac{\pi i}{2M}\right), \quad Q^{\sin}_i = \phi(Q_i) \sin\left(\frac{\pi i}{2M}\right)$$
           $$K^{\cos}_j = \phi(K_j) \cos\left(\frac{\pi j}{2M}\right), \quad K^{\sin}_j = \phi(K_j) \sin\left(\frac{\pi j}{2M}\right)$$
        
        4. Linear Attention Computation:
           $$\operatorname{Attn}(Q, K, V)_i = \frac{Q^{\cos}_i (K^{\cos \top} V) + Q^{\sin}_i (K^{\sin \top} V)}{Q^{\cos}_i \sum_j K^{\cos}_j + Q^{\sin}_i \sum_j K^{\sin}_j + \epsilon}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot d)$ linear complexity.
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot d + d^2))$ vs quadratic $\mathcal{O}(B \cdot H \cdot N^2)$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        max_seq_len: int = 512,
        act_fun: str = "relu",
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.max_seq_len: int = getattr(config, "max_position_embeddings", max_seq_len)
        self.act_fun: str = act_fun
        self.eps: float = eps

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Universal adapter weight copying."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except Exception:
                pass

    def _get_cos_sin_weights(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Generate positional cosine and sine factors: [1, 1, seq_len, 1]."""
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        m = max(self.max_seq_len, seq_len)
        angles = (math.pi / (2.0 * m)) * pos
        cos_weight = torch.cos(angles).view(1, 1, seq_len, 1)
        sin_weight = torch.sin(angles).view(1, 1, seq_len, 1)
        return cos_weight, sin_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        K = self.k_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, d).transpose(1, 2)

        # 1. Non-negative activation
        if self.act_fun == "relu":
            Q = F.relu(Q)
            K = F.relu(K)
        elif self.act_fun == "elu":
            Q = F.elu(Q) + 1.0
            K = F.elu(K) + 1.0

        # 2. Token valid mask
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # [B, N]
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)   # [B, 1, N, 1]
        K = K * mask_expanded

        # 3. Cosine/Sine position modulation
        cos_w, sin_w = self._get_cos_sin_weights(N, device, dtype)
        Q_cos = Q * cos_w
        Q_sin = Q * sin_w
        K_cos = K * cos_w
        K_sin = K * sin_w

        # 4. Linear attention associative accumulation
        # KV matrices: [B, H, d, d]
        KV_cos = torch.matmul(K_cos.transpose(-2, -1), V)
        KV_sin = torch.matmul(K_sin.transpose(-2, -1), V)

        # Denominator accumulators: [B, H, d, 1]
        K_cos_sum = K_cos.sum(dim=-2, keepdim=True).transpose(-2, -1)
        K_sin_sum = K_sin.sum(dim=-2, keepdim=True).transpose(-2, -1)

        # Numerator: [B, H, N, d]
        num = torch.matmul(Q_cos, KV_cos) + torch.matmul(Q_sin, KV_sin)

        # Denominator: [B, H, N, 1]
        den = torch.matmul(Q_cos, K_cos_sum) + torch.matmul(Q_sin, K_sin_sum) + self.eps

        context = num / den
        context = self.dropout(context)

        if head_mask is not None:
            context = context * head_mask

        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, None


@ATTENTION_REGISTRY.register("cosformer_inject")
def inject_cosformer(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> CosformerAttention:
    max_seq_len = getattr(comp_config, "max_seq_len", 512)
    new_attn = CosformerAttention(
        config=model_config,
        max_seq_len=max_seq_len,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
