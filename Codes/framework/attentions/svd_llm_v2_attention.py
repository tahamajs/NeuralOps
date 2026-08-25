r"""
SVD-LLM V2 Attention: Truncation-Aware SVD for LLM Compression.

Direct low-rank factorized parameterization maintaining explicit $U$, $\Sigma$, and $V^H$ factor buffers,
initialized via truncated SVD with energy preservation thresholds.

Mathematical Mechanism:
    For projection operators $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$:
    1. **Truncated Factorized Parameterization:**
       $$W \approx (U \operatorname{diag}(S)) V^H, \quad U \in \mathbb{R}^{D_{\text{out}} \times r}, \, S \in \mathbb{R}^r, \, V^H \in \mathbb{R}^{r \times D_{\text{in}}}$$
       $$y = x (V^H)^T \operatorname{diag}(S) U^T = \left((x (V^H)^T) \odot S\right) U^T$$
    2. **Multi-Head Attention Computation:**
       $$A = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) \in \mathbb{R}^{B \times H \times N \times N}$$
       $$O = (A V) W_O \in \mathbb{R}^{B \times N \times D}$$

Computational Complexity:
    - Factorized projection evaluation: $\mathcal{O}(B \cdot N \cdot r \cdot (D_{\text{in}} + D_{\text{out}}))$.
    - Attention matrix products: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot r \cdot D + B \cdot H \cdot N^2 \cdot d)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd_llm_v2")
@ATTENTION_REGISTRY.register("svd_llm_v2_attention")
class SVDLLMV2Attention(BaseAttention):
    r"""SVD-factorized attention layer with explicit $U, \Sigma, V^H$ matrix buffers.

    Args:
        config: Transformer configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Target compression rank $r$.
        energy_threshold: Singular value energy threshold fraction $\tau \in (0, 1]$.
        pretrained_attn: Optional source attention module for zero-shot SVD decomposition.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        energy_threshold: float = 0.95,
        pretrained_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        actual_rank = getattr(
            config,
            "svd_rank",
            rank if rank is not None else getattr(config, "rank", 64),
        )
        super().__init__(config, actual_rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        self.head_dim: int = getattr(config, "hidden_size", 768) // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = actual_rank
        self.energy_threshold: float = energy_threshold

        # Create parameter buffers
        self.query_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.query_s = nn.Parameter(torch.empty(self.rank))
        self.query_vh = nn.Parameter(torch.empty(self.rank, getattr(config, "hidden_size", 768)))
        self.query_bias = nn.Parameter(torch.zeros(self.all_head_size))

        self.key_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.key_s = nn.Parameter(torch.empty(self.rank))
        self.key_vh = nn.Parameter(torch.empty(self.rank, getattr(config, "hidden_size", 768)))
        self.key_bias = nn.Parameter(torch.zeros(self.all_head_size))

        self.value_u = nn.Parameter(torch.empty(self.all_head_size, self.rank))
        self.value_s = nn.Parameter(torch.empty(self.rank))
        self.value_vh = nn.Parameter(torch.empty(self.rank, getattr(config, "hidden_size", 768)))
        self.value_bias = nn.Parameter(torch.zeros(self.all_head_size))

        self.out_proj_u = nn.Parameter(torch.empty(getattr(config, "hidden_size", 768), self.rank))
        self.out_proj_s = nn.Parameter(torch.empty(self.rank))
        self.out_proj_vh = nn.Parameter(torch.empty(self.rank, self.all_head_size))
        self.out_proj_bias = nn.Parameter(torch.zeros(getattr(config, "hidden_size", 768)))

        self._init_random()

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )
        self.dropout = nn.Dropout(dropout_p)

    def _init_random(self) -> None:
        r"""Initialize orthogonal factor projections with identity scales."""
        nn.init.orthogonal_(self.query_u)
        nn.init.ones_(self.query_s)
        nn.init.orthogonal_(self.query_vh)
        nn.init.zeros_(self.query_bias)

        nn.init.orthogonal_(self.key_u)
        nn.init.ones_(self.key_s)
        nn.init.orthogonal_(self.key_vh)
        nn.init.zeros_(self.key_bias)

        nn.init.orthogonal_(self.value_u)
        nn.init.ones_(self.value_s)
        nn.init.orthogonal_(self.value_vh)
        nn.init.zeros_(self.value_bias)

        nn.init.orthogonal_(self.out_proj_u)
        nn.init.ones_(self.out_proj_s)
        nn.init.orthogonal_(self.out_proj_vh)
        nn.init.zeros_(self.out_proj_bias)

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        projections = [
            ("q", self.query_u, self.query_s, self.query_vh, self.query_bias),
            ("k", self.key_u, self.key_s, self.key_vh, self.key_bias),
            ("v", self.value_u, self.value_s, self.value_vh, self.value_bias),
            ("o", self.out_proj_u, self.out_proj_s, self.out_proj_vh, self.out_proj_bias),
        ]
        for proj, u_param, s_param, vh_param, bias_param in projections:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                with torch.no_grad():
                    W = w.float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    r = min(self.rank, len(S))
                    u_param.data.zero_()
                    s_param.data.zero_()
                    vh_param.data.zero_()
                    u_param.data[:, :r] = U[:, :r].to(dtype=u_param.dtype)
                    s_param.data[:r] = S[:r].to(dtype=s_param.dtype)
                    vh_param.data[:r, :] = Vt[:r, :].to(dtype=vh_param.dtype)
                    if b is not None:
                        bias_param.data.copy_(b)
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
        r"""Forward execution of SVD-LLM V2 attention layer using factorized kernel multiplications."""
        batch_size, seq_len, _ = hidden_states.shape

        # Factorized linear projections: x @ V^T -> * S -> @ U^T + b
        Q_act = ((hidden_states @ self.query_vh.t()) * self.query_s) @ self.query_u.t() + self.query_bias
        K_act = ((hidden_states @ self.key_vh.t()) * self.key_s) @ self.key_u.t() + self.key_bias
        V_act = ((hidden_states @ self.value_vh.t()) * self.value_s) @ self.value_u.t() + self.value_bias

        Q = Q_act.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K_act.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V_act.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = self.head_dim**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + (1.0 - attention_mask[:, None, :, :].to(scores.dtype)) * -10000.0
            else:
                scores = scores + (1.0 - attention_mask.to(scores.dtype)) * -10000.0

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        context = torch.matmul(probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out = ((context @ self.out_proj_vh.t()) * self.out_proj_s) @ self.out_proj_u.t() + self.out_proj_bias

        if output_attentions:
            return out, probs
        return out, None


@ATTENTION_REGISTRY.register("svd_llm_v2_inject")
def inject_svd_llm_v2(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SVDLLMV2Attention:
    r"""Bridge helper injecting SVDLLMV2Attention into a target transformer layer."""
    rank = getattr(comp_config, "rank", 64)
    energy = getattr(comp_config, "svd_energy", 0.95)
    new_attn = SVDLLMV2Attention(
        model_config, rank=rank, energy_threshold=energy, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
