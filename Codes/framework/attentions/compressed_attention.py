r"""
Compressed Multi-Head Attention via Truncated Singular Value Decomposition (SVD).

Provides rank-constrained and singular-energy-thresholded low-rank approximations of standard
PyTorch `nn.MultiheadAttention` input (QKV) and output projection matrices.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("compressed_attention_layer")
class CompressedAttentionLayer(nn.Module):
    r"""
    SVD-Compressed Multi-Head Attention Layer with Fixed Target Ranks.

    Mathematical Formulation:
        Approximates combined QKV input projection $W_{\text{in}} \in \mathbb{R}^{3D \times D}$ and output projection
        $W_{\text{out}} \in \mathbb{R}^{D \times D}$ via rank-$r$ truncated SVD:
        $$W_{\text{in}} \approx U_{\text{qkv}} \Sigma_{\text{qkv}} V_{\text{qkv}}^T$$
        $$W_{\text{out}} \approx U_{\text{out}} \Sigma_{\text{out}} V_{\text{out}}^T$$
        The forward transformation evaluates associative low-rank multiplications:
        $$\text{QKV}(x) = x V_{\text{qkv}} \Sigma_{\text{qkv}} U_{\text{qkv}}^T + b_{\text{qkv}}$$
        $$\text{Out}(C) = C V_{\text{out}} \Sigma_{\text{out}} U_{\text{out}}^T + b_{\text{out}}$$

    Complexity:
        - Projection FLOPs: $\mathcal{O}(B \cdot N \cdot (D \cdot r_{\text{qkv}} + r_{\text{qkv}} \cdot 3D + D \cdot r_{\text{out}} + r_{\text{out}} \cdot D))$.
        - Attention Map FLOPs: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_h)$.

    Tensor Dimensions:
        - Input `query` / `key` / `value`: $[B, N, D]$
        - Projected QKV: $[B, N, 3D]$
        - Attention Context $C$: $[B, N, D]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        original_layer: nn.MultiheadAttention,
        rank_qkv: int,
        rank_out: int,
        method: str = "svd",
    ) -> None:
        super().__init__()
        self.original_dim: int = original_layer.embed_dim
        self.num_heads: int = original_layer.num_heads
        self.head_dim: int = self.original_dim // self.num_heads
        self.method: str = method

        w_qkv = original_layer.in_proj_weight.data.cpu().numpy()
        w_o = original_layer.out_proj.weight.data.cpu().numpy()

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_o, full_matrices=False)

        rank_qkv = min(rank_qkv, w_qkv.shape[1], w_qkv.shape[0])
        rank_out = min(rank_out, w_o.shape[1], w_o.shape[0])

        self.rank_qkv: int = rank_qkv
        self.rank_out: int = rank_out

        self.U_qkv: nn.Parameter = nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.S_qkv_sqrt: nn.Parameter = nn.Parameter(
            torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float()
        )
        self.Vh_qkv: nn.Parameter = nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())

        self.U_out: nn.Parameter = nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float())
        self.S_out_sqrt: nn.Parameter = nn.Parameter(
            torch.from_numpy(np.sqrt(S_out[:rank_out])).float()
        )
        self.Vh_out: nn.Parameter = nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float())

        if original_layer.in_proj_bias is not None:
            self.bias_qkv: Optional[nn.Parameter] = nn.Parameter(original_layer.in_proj_bias.data.clone())
        else:
            self.bias_qkv = None

        if original_layer.out_proj.bias is not None:
            self.bias_out: Optional[nn.Parameter] = nn.Parameter(original_layer.out_proj.bias.data.clone())
        else:
            self.bias_out = None

        self.dropout_p: float = original_layer.dropout
        self.batch_first: bool = getattr(original_layer, "batch_first", False)
        self.kdim: Optional[int] = getattr(original_layer, "kdim", None)
        self.vdim: Optional[int] = getattr(original_layer, "vdim", None)
        self.add_bias_kv: bool = getattr(original_layer, "add_bias_kv", False)
        self.add_zero_attn: bool = getattr(original_layer, "add_zero_attn", False)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution pass for SVD-compressed multihead attention.

        Args:
            query (torch.Tensor): Query activations $[B, N, D]$.
            key (Optional[torch.Tensor]): Key activations $[B, N, D]$ (defaults to query if None).
            value (Optional[torch.Tensor]): Value activations $[B, N, D]$ (defaults to query if None).
            key_padding_mask (Optional[torch.Tensor]): Boolean padding mask tensor.
            need_weights (bool): Whether to return attention weights.
            attn_mask (Optional[torch.Tensor]): Additive 2D/3D attention mask.
            average_attn_weights (bool): Whether to average weights across heads.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Compressed attention output $[B, N, D]$.
                - Attention weights tensor (if requested) or None.
        """
        x = query
        dim = self.original_dim

        projected_x = torch.matmul(x, self.Vh_qkv.t()) * self.S_qkv_sqrt.unsqueeze(0).unsqueeze(0)
        qkv_compressed = torch.matmul(projected_x, self.U_qkv.t())
        if self.bias_qkv is not None:
            qkv_compressed = qkv_compressed + self.bias_qkv

        q = qkv_compressed[:, :, :dim]
        k = qkv_compressed[:, :, dim : 2 * dim]
        v = qkv_compressed[:, :, 2 * dim :]

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        context = torch.matmul(attn_weights, v)

        context_fused = torch.matmul(context, self.Vh_out.t()) * self.S_out_sqrt.unsqueeze(0).unsqueeze(0)
        final_output = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            final_output = final_output + self.bias_out

        return final_output, attn_weights if need_weights else None


@ATTENTION_REGISTRY.register("compressed_attention_layer_energy")
class CompressedAttentionLayer2(nn.Module):
    r"""
    Energy-Preserving SVD Compression of Multi-Head Attention.

    Automatically resolves optimal truncation rank $r$ such that cumulative singular value energy
    meets or exceeds $\tau_{\text{energy}}$:
    $$\frac{\sum_{i=1}^r \sigma_i}{\sum_{j=1}^R \sigma_j} \ge \tau_{\text{energy}}, \quad r \ge r_{\min}$$

    Complexity:
        - Computation: $\mathcal{O}(B \cdot N \cdot D \cdot (r_{\text{qkv}} + r_{\text{out}}) + B \cdot N^2 \cdot D)$.
        - Memory: $\mathcal{O}(D \cdot (r_{\text{qkv}} + r_{\text{out}}))$.

    Tensor Dimensions:
        - Input: $[B, N, D]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        original_layer: nn.MultiheadAttention,
        energy_threshold: float = 0.95,
        min_rank: int = 16,
    ) -> None:
        super().__init__()
        self.original_dim: int = original_layer.embed_dim
        self.num_heads: int = original_layer.num_heads
        self.head_dim: int = self.original_dim // self.num_heads
        self.dropout_p: float = original_layer.dropout
        self.batch_first: bool = getattr(original_layer, "batch_first", False)

        w_qkv = original_layer.in_proj_weight.data.cpu().numpy()
        w_out = original_layer.out_proj.weight.data.cpu().numpy()

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        energy_qkv = np.cumsum(S_qkv) / np.sum(S_qkv)
        rank_qkv = int(np.argmax(energy_qkv >= energy_threshold)) + 1
        rank_qkv = max(min_rank, min(rank_qkv, w_qkv.shape[1]))

        U_out, S_out, Vh_out = np.linalg.svd(w_out, full_matrices=False)
        energy_out = np.cumsum(S_out) / np.sum(S_out)
        rank_out = int(np.argmax(energy_out >= energy_threshold)) + 1
        rank_out = max(min_rank, min(rank_out, w_out.shape[1]))

        self.rank_qkv: int = rank_qkv
        self.rank_out: int = rank_out

        self.U_qkv: nn.Parameter = nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.S_qkv_sqrt: nn.Parameter = nn.Parameter(
            torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float()
        )
        self.Vh_qkv: nn.Parameter = nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())

        self.U_out: nn.Parameter = nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float())
        self.S_out_sqrt: nn.Parameter = nn.Parameter(
            torch.from_numpy(np.sqrt(S_out[:rank_out])).float()
        )
        self.Vh_out: nn.Parameter = nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float())

        if original_layer.in_proj_bias is not None:
            self.bias_qkv: Optional[nn.Parameter] = nn.Parameter(original_layer.in_proj_bias.data.clone())
        else:
            self.bias_qkv = None

        if original_layer.out_proj.bias is not None:
            self.bias_out: Optional[nn.Parameter] = nn.Parameter(original_layer.out_proj.bias.data.clone())
        else:
            self.bias_out = None

        self.kdim: Optional[int] = getattr(original_layer, "kdim", None)
        self.vdim: Optional[int] = getattr(original_layer, "vdim", None)
        self.add_bias_kv: bool = getattr(original_layer, "add_bias_kv", False)
        self.add_zero_attn: bool = getattr(original_layer, "add_zero_attn", False)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation for energy-thresholded SVD attention.

        Args:
            query (torch.Tensor): Query activations $[B, N, D]$.
            key (Optional[torch.Tensor]): Key activations.
            value (Optional[torch.Tensor]): Value activations.
            key_padding_mask (Optional[torch.Tensor]): Mask for padding tokens.
            need_weights (bool): Whether to return attention weights.
            attn_mask (Optional[torch.Tensor]): Additive attention mask.
            average_attn_weights (bool): Average across attention heads.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed output tensor $[B, N, D]$.
                - Attention weights or None.
        """
        x = query
        dim = self.original_dim

        projected_x = torch.matmul(x, self.Vh_qkv.t()) * self.S_qkv_sqrt.unsqueeze(0).unsqueeze(0)
        qkv_compressed = torch.matmul(projected_x, self.U_qkv.t())
        if self.bias_qkv is not None:
            qkv_compressed = qkv_compressed + self.bias_qkv

        q = qkv_compressed[:, :, :dim]
        k = qkv_compressed[:, :, dim : 2 * dim]
        v = qkv_compressed[:, :, 2 * dim :]

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        context = torch.matmul(attn_weights, v)

        context_fused = torch.matmul(context, self.Vh_out.t()) * self.S_out_sqrt.unsqueeze(0).unsqueeze(0)
        final_output = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            final_output = final_output + self.bias_out

        return final_output, attn_weights if need_weights else None
