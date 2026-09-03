"""Multi-Head Latent Attention (MLA) – DeepSeek‑V2.

Paper: https://arxiv.org/abs/2405.04434
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mla")
class MLAAttention(nn.Module):
    r"""Multi-Head Latent Attention (MLA) Mechanism.

    Compresses the Key and Value representations into a low-rank latent vector $\mathbf{c}_{KV} \in \mathbb{R}^{d_c}$
    prior to head projection, drastically reducing Key-Value activation caching requirements:
    $$\mathbf{c}_{KV} = W_{DKV} X \in \mathbb{R}^{B \times N \times d_c}$$
    $$K = W_{UK} \mathbf{c}_{KV} \in \mathbb{R}^{B \times N \times (H \cdot d_k)}, \quad V = W_{UV} \mathbf{c}_{KV} \in \mathbb{R}^{B \times N \times (H \cdot d_v)}$$
    $$\text{Attention}_{\text{MLA}}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot N \cdot d \cdot d_c + B \cdot H \cdot N^2 \cdot d_k)$
        - Cache Storage: $\mathcal{O}(B \cdot N \cdot d_c)$ vs $\mathcal{O}(2 \cdot B \cdot N \cdot H \cdot d_k)$ where $d_c \ll 2 H d_k$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
    """

    def __init__(self, config: Any, rank: int = 128) -> None:
        super().__init__()
        self.num_heads: int = getattr(
            config, "num_attention_heads", getattr(config, "n_heads", getattr(config, "n_head", getattr(config, "num_heads", 12)))
        )
        self.head_dim: int = getattr(config, "head_dim", config.hidden_size // max(1, self.num_heads))
        self.rank: int = getattr(config, "mla_rank", rank)

        self.q_proj: nn.Linear = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.kv_down: nn.Linear = nn.Linear(
            config.hidden_size, self.rank, bias=False
        )
        self.k_up: nn.Linear = nn.Linear(
            self.rank, config.hidden_size, bias=False
        )
        self.v_up: nn.Linear = nn.Linear(
            self.rank, config.hidden_size, bias=False
        )

        self.out_proj: nn.Linear = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.dropout: nn.Dropout = nn.Dropout(getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)))

    def copy_weights(self, old_attn: Any) -> None:
        """Initialize MLA projections from old attention weights via SVD."""
        from framework.core.adapters import UniversalAdapter
        with torch.no_grad():
            # 1. Q projection
            q_res = UniversalAdapter.get_weight_bias(old_attn, "q")
            if q_res is not None:
                w_q, _ = q_res
                if w_q.shape == self.q_proj.weight.shape:
                    self.q_proj.weight.data.copy_(w_q.to(self.q_proj.weight.device, self.q_proj.weight.dtype))
                elif w_q.t().shape == self.q_proj.weight.shape:
                    self.q_proj.weight.data.copy_(w_q.t().to(self.q_proj.weight.device, self.q_proj.weight.dtype))

            # 2. Output projection
            o_res = UniversalAdapter.get_weight_bias(old_attn, "o")
            if o_res is not None:
                w_o, b_o = o_res
                if w_o.shape == self.out_proj.weight.shape:
                    self.out_proj.weight.data.copy_(w_o.to(self.out_proj.weight.device, self.out_proj.weight.dtype))
                elif w_o.t().shape == self.out_proj.weight.shape:
                    self.out_proj.weight.data.copy_(w_o.t().to(self.out_proj.weight.device, self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(b_o.to(self.out_proj.bias.device, self.out_proj.bias.dtype))

            # 3. Joint K and V low-rank decomposition via stacked SVD
            k_res = UniversalAdapter.get_weight_bias(old_attn, "k")
            v_res = UniversalAdapter.get_weight_bias(old_attn, "v")
            if k_res is not None and v_res is not None:
                w_k, _ = k_res
                w_v, _ = v_res
                try:
                    w_k_f = w_k.float()
                    w_v_f = w_v.float()
                    if w_k_f.shape[1] == self.hidden_size and w_v_f.shape[1] == self.hidden_size:
                        w_kv = torch.cat([w_k_f, w_v_f], dim=0)
                        u_kv, s_kv, vh_kv = torch.linalg.svd(w_kv, full_matrices=False)
                        r = min(self.rank, s_kv.shape[0], self.kv_down.weight.shape[0], self.k_up.weight.shape[1], self.v_up.weight.shape[1])
                        self.kv_down.weight.data[:r, :].copy_(vh_kv[:r, :].to(self.kv_down.weight.device, self.kv_down.weight.dtype))
                        up = (u_kv[:, :r] * s_kv[:r].unsqueeze(0)).to(self.k_up.weight.device, self.k_up.weight.dtype)
                        d_k = w_k_f.shape[0]
                        d_v = w_v_f.shape[0]
                        self.k_up.weight.data[:d_k, :r].copy_(up[:d_k, :r])
                        self.v_up.weight.data[:d_v, :r].copy_(up[d_k:d_k+d_v, :r])
                except Exception:
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
        """Forward pass for Multi-Head Latent Attention.

        Args:
            hidden_states (torch.Tensor): Input hidden activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor `[Batch, SeqLen]` or additive mask.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation `[Batch, SeqLen, HiddenDim]` and attention probabilities.
        """
        batch_size, seq_len, _ = hidden_states.shape

        q: torch.Tensor = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        latent: torch.Tensor = self.kv_down(hidden_states)
        k: torch.Tensor = self.k_up(latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v: torch.Tensor = self.v_up(latent).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores: torch.Tensor = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores + (1.0 - attention_mask[:, None, None, :].to(scores.dtype)) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1).to(scores.dtype)
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attention_mask, -10000.0)
                else:
                    scores = scores + attention_mask.to(scores.dtype)

        attn_probs: torch.Tensor = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context: torch.Tensor = torch.matmul(attn_probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        out: torch.Tensor = self.out_proj(context)

        if output_attentions:
            return out, attn_probs
        return out, None


def inject_mla(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> MLAAttention:
    r"""Dynamic bridge injecting MLAAttention into a transformer layer."""
    rank = getattr(comp_config, "mla_rank", 128) if comp_config is not None else 128
    
    class ConfigWrapper:
        def __init__(self, cfg,
        **kwargs):
            self.hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768))
            self.num_attention_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", getattr(cfg, "num_heads", 12)))
            self.dropout = getattr(cfg, "attention_probs_dropout_prob", getattr(cfg, "dropout", 0.1))
            self.mla_rank = rank

    wrapper = ConfigWrapper(model_config)
    new_attn = MLAAttention(wrapper, rank=rank)

    # Pretrained weight transfer
    if old_attn is not None:
        if hasattr(old_attn, "q_lin") and hasattr(new_attn, "q_proj"):
            new_attn.q_proj.weight.data.copy_(old_attn.q_lin.weight.data)
        if hasattr(old_attn, "out_lin") and hasattr(new_attn, "out_proj"):
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            if old_attn.out_lin.bias is not None and new_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
        if hasattr(old_attn, "k_lin") and hasattr(old_attn, "v_lin"):
            # SVD decomposition of K and V to initialize low-rank latent projections
            try:
                k_w = old_attn.k_lin.weight.data
                u_k, s_k, vh_k = torch.linalg.svd(k_w, full_matrices=False)
                # Down projection: V^T, Up projection: U * S
                r = min(rank, s_k.shape[0])
                new_attn.kv_down.weight.data[:r, :] = vh_k[:r, :]
                new_attn.k_up.weight.data[:, :r] = u_k[:, :r] @ torch.diag(s_k[:r])
                
                v_w = old_attn.v_lin.weight.data
                u_v, s_v, vh_v = torch.linalg.svd(v_w, full_matrices=False)
                new_attn.v_up.weight.data[:, :r] = u_v[:, :r] @ torch.diag(s_v[:r])
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass

    layer.attention = new_attn
    if old_attn is not None:
        p = next(old_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    return new_attn
