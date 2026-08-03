r"""
Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.

Compresses transformer multi-head self-attention projection matrices into low-rank
latent subspaces initialized through truncated Singular Value Decomposition (SVD).

Mathematical Mechanism:
    For projection weights $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$, each matrix is factorized into rank-$r$ factors ($r \ll D$):
    1. **SVD-Initialized Factorization:**
       $$W \approx U_{:, :r} \Sigma_{1:r} V_{:r, :}^T \implies W^{\text{down}} = V_{:r, :}^T \in \mathbb{R}^{r \times D}, \quad W^{\text{up}} = U_{:, :r} \Sigma_{1:r} \in \mathbb{R}^{D \times r}$$
    2. **Latent Representation Computation:**
       $$Z_Q = X (W_Q^{\text{down}})^T, \quad Z_K = X (W_K^{\text{down}})^T, \quad Z_V = X (W_V^{\text{down}})^T \in \mathbb{R}^{B \times N \times r}$$
       $$Q = Z_Q (W_Q^{\text{up}})^T, \quad K = Z_K (W_K^{\text{up}})^T, \quad V = Z_V (W_V^{\text{up}})^T \in \mathbb{R}^{B \times N \times D}$$
    3. **Multi-Head Scaled Dot-Product Attention:**
       $$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) \in \mathbb{R}^{B \times H \times N \times N}$$
       $$O = (A V) W_O \in \mathbb{R}^{B \times N \times D}$$

Computational Complexity:
    - Projection FLOPs per token: $\mathcal{O}(D \cdot r)$ compared to $\mathcal{O}(D^2)$ standard dense projections.
    - Attention matrix evaluation: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot D \cdot r + B \cdot H \cdot N^2 \cdot d)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("soa_module")
class SOAAttention(nn.Module):
    r"""SOA Attention layer compressing Q/K/V linear projections into rank-$r$ bottlenecks.

    Args:
        model_config: Transformer configuration specifying `hidden_size` and `num_attention_heads`.
        k_rank: Latent bottleneck compression rank ($r$).
        dropout: Attention dropout probability.
        pretrained_attn: Source attention module for zero-shot SVD weight initialization.
    """

    def __init__(
        self,
        model_config: Any,
        k_rank: int,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot SVD initialization.")

        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.hidden_size: int = model_config.hidden_size
        self.k_rank: int = k_rank

        self.q_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.k_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.v_down = nn.Linear(self.hidden_size, k_rank, bias=False)

        self.q_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.k_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.v_up = nn.Linear(k_rank, self.hidden_size, bias=False)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        self._init_from_pretrained(pretrained_attn)

    def _init_from_pretrained(self, pretrained_attn: Any) -> None:
        r"""Initialize down- and up-projections via truncated SVD decomposition."""
        with torch.no_grad():
            for orig, down, up in [
                (pretrained_attn.q_lin, self.q_down, self.q_up),
                (pretrained_attn.k_lin, self.k_down, self.k_up),
                (pretrained_attn.v_lin, self.v_down, self.v_up),
            ]:
                W = orig.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.k_rank, U.size(1), Vt.size(0))
                down.weight.data[:r, :] = Vt[:r, :].to(down.weight.dtype)
                up.weight.data[:, :r] = (U[:, :r] * S[:r]).to(up.weight.dtype)

            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight)
            if pretrained_attn.out_lin.bias is not None:
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward pass computing bottlenecked multi-head attention.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask tensor `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        B, N, _ = hidden_states.shape

        z_q = self.q_down(hidden_states)
        z_k = self.k_down(hidden_states)
        z_v = self.v_down(hidden_states)

        q = self.q_up(z_q)
        k = self.k_up(z_k)
        v = self.v_up(z_v)

        H = self.num_heads
        D = self.head_dim
        q = q.view(B, N, H, D).transpose(1, 2)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        scale = D**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = (
            attention_mask
            if attention_mask is not None
            else kwargs.get("attn_mask", None)
        )
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -1e9
            elif mask.dim() == 3:
                scores = scores + mask.unsqueeze(1)
            else:
                scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        output = self.out_proj(context)

        if output_attentions:
            return output, attn_weights
        return output, None


@ATTENTION_REGISTRY.register("soa_attn")
def inject_soa_attn(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SOAAttention:
    r"""Bridge helper injecting SOAAttention into a target transformer layer."""
    k_rank = getattr(comp_config, "soa_rank", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = SOAAttention(model_config, k_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn
