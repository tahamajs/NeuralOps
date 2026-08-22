r"""
Compressed Multi-Head Attention for DistilBERT via Joint Truncated SVD.

Performs joint singular value decomposition on stacked query, key, and value projection matrices
as well as the final output projection matrix to achieve high-compression zero-shot inference.
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class CompressedDistilBertAttention(nn.Module):
    r"""
    DistilBERT Attention Module with Joint SVD-Compressed Projections.

    Mathematical Formulation:
        Let $W_{\text{qkv}} = [W_Q; W_K; W_V] \in \mathbb{R}^{3D \times D}$ denote the stacked input weight matrix.
        Truncated SVD factorizes $W_{\text{qkv}}$ and output projection $W_{\text{out}}$:
        $$W_{\text{qkv}} \approx U_{\text{qkv}} \Sigma_{\text{qkv}} V_{\text{qkv}}^T, \quad \text{rank } r_{\text{qkv}}$$
        $$W_{\text{out}} \approx U_{\text{out}} \Sigma_{\text{out}} V_{\text{out}}^T, \quad \text{rank } r_{\text{out}}$$

        For input activations $X \in \mathbb{R}^{B \times N \times D}$:
        $$\text{QKV} = (X V_{\text{qkv}}) \Sigma_{\text{qkv}} U_{\text{qkv}}^T + b_{\text{qkv}}$$
        $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V$$
        $$\text{Output} = (\text{Context} \cdot V_{\text{out}}) \Sigma_{\text{out}} U_{\text{out}}^T + b_{\text{out}}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot N \cdot D \cdot r_{\text{qkv}} + B \cdot H \cdot N^2 \cdot d_h + B \cdot N \cdot D \cdot r_{\text{out}})$.
        - Parameter Count: $(3D + D) \cdot r_{\text{qkv}} + 2D \cdot r_{\text{out}} \ll 4D^2$.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, N, D]$
        - SVD Projections: $[B, N, 3D] \to [B, H, N, d_h]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        model_config: Any,
        rank_qkv: int = 32,
        rank_out: int = 32,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn is required for SVD compression initialization")
        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.hidden_size: int = model_config.hidden_size
        self.dropout: float = dropout

        q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", None))
        k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", None))
        v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", None))
        out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None))

        w_q = q_src.weight.data.cpu().numpy()
        w_k = k_src.weight.data.cpu().numpy()
        w_v = v_src.weight.data.cpu().numpy()
        w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)
        w_out = out_src.weight.data.cpu().numpy()

        rank_qkv = max(1, min(rank_qkv, w_qkv.shape[0], w_qkv.shape[1]))
        rank_out = max(1, min(rank_out, w_out.shape[0], w_out.shape[1]))

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_out, full_matrices=False)

        self.register_buffer("U_qkv", torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.register_buffer("S_qkv", torch.from_numpy(S_qkv[:rank_qkv]).float())
        self.register_buffer("Vh_qkv", torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())
        self.register_buffer("U_out", torch.from_numpy(U_out[:, :rank_out]).float())
        self.register_buffer("S_out", torch.from_numpy(S_out[:rank_out]).float())
        self.register_buffer("Vh_out", torch.from_numpy(Vh_out[:rank_out, :]).float())

        if q_src.bias is not None:
            self.register_buffer("bias_q", q_src.bias.data.clone())
        else:
            self.bias_q = None
            
        if k_src.bias is not None:
            self.register_buffer("bias_k", k_src.bias.data.clone())
        else:
            self.bias_k = None
            
        if v_src.bias is not None:
            self.register_buffer("bias_v", v_src.bias.data.clone())
        else:
            self.bias_v = None
            
        if out_src.bias is not None:
            self.register_buffer("bias_out", out_src.bias.data.clone())
        else:
            self.bias_out = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for CompressedDistilBertAttention.

        Args:
            hidden_states (torch.Tensor): Input representation $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, N]$ or $[B, 1, 1, N]$.
            output_attentions (bool): Whether to return attention probabilities.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed context representation $[B, N, D]$.
                - Attention probabilities tensor (if requested) or None.
        """
        B, N, _ = hidden_states.shape
        proj = torch.matmul(hidden_states, self.Vh_qkv.t()) * self.S_qkv.unsqueeze(0).unsqueeze(0)
        qkv_comp = torch.matmul(proj, self.U_qkv.t())
        dim = self.hidden_size
        q = qkv_comp[:, :, :dim]
        k = qkv_comp[:, :, dim : 2 * dim]
        v = qkv_comp[:, :, 2 * dim :]

        if self.bias_q is not None:
            q = q + self.bias_q
        if self.bias_k is not None:
            k = k + self.bias_k
        if self.bias_v is not None:
            v = v + self.bias_v

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.float()
            else:
                scores = scores + attention_mask.float()

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, dim)

        fused = torch.matmul(context, self.Vh_out.t()) * self.S_out.unsqueeze(0).unsqueeze(0)
        out = torch.matmul(fused, self.U_out.t())
        if self.bias_out is not None:
            out = out + self.bias_out
        return out, attn if output_attentions else None


@ATTENTION_REGISTRY.register("compressed_mha")
def inject_compressed_mha(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> CompressedDistilBertAttention:
    r"""
    Injects SVD-compressed multi-head attention into a transformer layer.

    Args:
        layer (nn.Module): Host layer module.
        model_config (Any): Model configuration object.
        comp_config (Any): Compression hyperparameter configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total transformer layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer for SVD initialization.

    Returns:
        CompressedDistilBertAttention: Initialized compressed attention layer.
    """
    rank_qkv = getattr(comp_config, "svd_rank_qkv", 32)
    rank_out = getattr(comp_config, "svd_rank_out", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = CompressedDistilBertAttention(
        model_config, rank_qkv, rank_out, dropout, old_attn
    )
    layer.attention = new_attn
    return new_attn
