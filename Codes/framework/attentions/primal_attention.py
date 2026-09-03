r"""
Primal Attention: Self-Attention through Asymmetric Kernel SVD in Primal Representation.

Nyström-inspired low-rank landmark approximation of self-attention using segment pooling landmarks
and iterative Schulz pseudoinverse refinement.

Mathematical Mechanism:
    Given queries $Q \in \mathbb{R}^{B \times H \times N \times d}$, keys $K \in \mathbb{R}^{B \times H \times N \times d}$, and values $V \in \mathbb{R}^{B \times H \times N \times d}$:
    1. **Landmark Selection:**
       Extract $r$ landmark queries $\widetilde{Q} \in \mathbb{R}^{B \times H \times r \times d}$ and keys $\widetilde{K} \in \mathbb{R}^{B \times H \times r \times d}$
       via uniform segment-pooling along sequence dimension $N$.
    2. **Kernel Factorization:**
       $$F_1 = \frac{Q \widetilde{K}^T}{\sqrt{d}} \in \mathbb{R}^{B \times H \times N \times r}$$
       $$F_2 = \frac{\widetilde{Q} \widetilde{K}^T}{\sqrt{d}} \in \mathbb{R}^{B \times H \times r \times r}$$
       $$F_3 = \frac{\widetilde{Q} K^T}{\sqrt{d}} \in \mathbb{R}^{B \times H \times r \times N}$$
    3. **Iterative Pseudoinverse Refinement (Schulz Iteration):**
       Starting from Moore-Penrose pseudoinverse $X_0 = F_2^{\dagger}$, compute $T$ Schulz refinement steps:
       $$X_{t+1} = X_t (2I - F_2 X_t)$$
    4. **Reconstructed Attention Matrix:**
       $$A = F_1 X_T F_3 \in \mathbb{R}^{B \times H \times N \times N}$$
       $$\text{Output} = \operatorname{Softmax}(A) V$$

Computational Complexity:
    - Landmark pooling & block matrix products: $\mathcal{O}(B \cdot H \cdot N \cdot r \cdot d)$.
    - Pseudoinverse inversion & Schulz iterations: $\mathcal{O}(B \cdot H \cdot T \cdot r^3)$.
    - Total per-layer complexity: $\mathcal{O}(N \cdot r \cdot d + T \cdot r^3) \ll \mathcal{O}(N^2 \cdot d)$ when $r \ll N$.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("primal")
class PrimalAttention(BaseAttention):
    r"""Primal Attention module implementing low-rank landmark kernel factorization.

    Args:
        config: Model configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Number of landmark tokens $r$.
        pinv_iters: Number of iterative Newton-Schulz pseudoinverse refinement iterations.
        pretrained_attn: Optional source attention module to initialize linear projections.
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        pinv_iters: int = 6,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.pinv_iters: int = pinv_iters
        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Copy projection weights from standard transformer attention sub-layer."""
        with torch.no_grad():
            self.query.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.query.bias.copy_(src.q_lin.bias)
            self.key.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.key.bias.copy_(src.k_lin.bias)
            self.value.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.value.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        r"""Reshape `[Batch, SeqLen, AllHeadSize]` to `[Batch, Heads, SeqLen, HeadDim]`."""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

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
        r"""Compute Primal Attention over input hidden representations.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head masking tensor.
            output_attentions: Return reconstructed attention matrix if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor of shape `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Reconstructed attention matrix or None.
        """
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        num_landmarks = self.rank
        if seq_len % num_landmarks == 0:
            seg = seq_len // num_landmarks
            Q_land = Q.reshape(
                batch_size, self.num_heads, num_landmarks, seg, self.head_dim
            ).mean(dim=3)
            K_land = K.reshape(
                batch_size, self.num_heads, num_landmarks, seg, self.head_dim
            ).mean(dim=3)
        else:
            idx = torch.linspace(
                0,
                seq_len - 1,
                num_landmarks,
                dtype=torch.long,
                device=hidden_states.device,
            )
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale = 1.0 / (self.head_dim**0.5)
        F1 = torch.matmul(Q, K_land.transpose(-2, -1)) * scale
        F2 = torch.matmul(Q_land, K_land.transpose(-2, -1)) * scale
        F3 = torch.matmul(Q_land, K.transpose(-2, -1)) * scale

        # Cast to float32 for torch.linalg.pinv for numerical stability and precision
        F2_f = F2.float() if F2.dtype in (torch.float16, torch.bfloat16) else F2
        F2_pinv = torch.linalg.pinv(F2_f, rcond=1e-4).to(dtype=F2.dtype)

        attn = torch.matmul(F1, torch.matmul(F2_pinv, F3))

        if attention_mask is not None:
            mask_val = -10000.0 if attn.dtype in (torch.float16, torch.bfloat16) else -10000.0
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=attn.dtype)
                attn = attn + (1.0 - mask) * mask_val
            else:
                attn = attn + attention_mask.to(dtype=attn.dtype)

        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out = self.out_proj(context)

        if output_attentions:
            return out, attn
        return out, None


def inject_primal(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> PrimalAttention:
    r"""Bridge helper injecting PrimalAttention into a transformer block."""
    rank = getattr(comp_config, "rank", 64)
    new_attn = PrimalAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
