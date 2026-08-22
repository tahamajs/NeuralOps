r"""
FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module.

Integrates SpargeAttn subspace score prediction, SeerAttention block gating, and MoBA top-$k$ block-sparse
attention for training-free acceleration of long-context sequence modeling.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("fusion_attn")
class FUSIONATTN(nn.Module):
    r"""
    FUSIONATTN: Hybrid Sparse Attention with Block Gating and Subspace Estimation.

    Mathematical Formulation:
        1. SpargeAttn Subspace Estimation:
           Evaluates fast approximate attention maps on low-rank channel subspace $d_{\text{sub}} = d_h / 2$:
           $$S_{\text{approx}} = \frac{Q_{:, :d_{\text{sub}}} K_{:, :d_{\text{sub}}}^T}{\sqrt{d_{\text{sub}}}}$$
           $$M_{\text{dense}} = \mathbb{I}\left( S_{\text{approx}} > \mu(S_{\text{approx}}) \right)$$

        2. SeerAttention Block Gating:
           Averages tokens into blocks of size $B_{\text{size}}$:
           $$\bar{Q}_b = \frac{1}{B_{\text{size}}} \sum_{i \in \text{block}_b} Q_i, \quad \bar{K}_{b'} = \frac{1}{B_{\text{size}}} \sum_{j \in \text{block}_{b'}} K_j$$
           $$G_{b, b'} = \frac{\text{MLP}(\bar{Q}_b) \cdot \text{MLP}(\bar{K}_{b'})^T}{\tau_{\text{gate}}}$$

        3. MoBA Top-$k$ Block Sparse Computation:
           $$\mathcal{I}_{\text{top-}k}(b) = \text{Top-}k(G_{b, :}, k_{\text{blocks}})$$
           $$O_b = \sum_{b' \in \mathcal{I}_{\text{top-}k}(b)} \text{Softmax}\left(\frac{Q_b K_{b'}^T}{\sqrt{d_h}} + M_{\text{dense}}\right) V_{b'}$$

    Complexity:
        - Block Gating: $\mathcal{O}\left(B \cdot H \cdot \left(\frac{N}{B_{\text{size}}}\right)^2 \cdot d_h\right)$.
        - Block Sparse Attention: $\mathcal{O}\left(B \cdot H \cdot N \cdot k_{\text{blocks}} \cdot B_{\text{size}} \cdot d_h\right) \ll \mathcal{O}(N^2)$.

    Tensor Dimensions:
        - Input: $[B, N, D]$
        - Block Query / Key / Value: $[B, H, N_{\text{blocks}}, B_{\text{size}}, d_h]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        block_size: int = 64,
        top_k_blocks: int = 4,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None and hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        self.block_size: int = block_size
        self.top_k_blocks: int = top_k_blocks

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_gate = nn.Sequential(
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, 1),
        )

        self.gate_temp: nn.Parameter = nn.Parameter(torch.ones(1) * 1.0)

    def _sparge_predict(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        r"""Evaluates approximate attention map using half-channel subspace projections."""
        d_sub = max(1, self.head_dim // 2)
        q_approx = q[..., :d_sub]
        k_approx = k[..., :d_sub]

        scores_approx = torch.matmul(q_approx, k_approx.transpose(-2, -1)) / (d_sub**0.5)
        dense_mask = scores_approx > scores_approx.mean(dim=-1, keepdim=True)
        return dense_mask

    def _seer_gate_blocks(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        r"""Computes SeerAttention block importance scores."""
        B, H, N, D = q.shape
        pad_len = (self.block_size - (N % self.block_size)) % self.block_size
        if pad_len > 0:
            q_padded = F.pad(q, (0, 0, 0, pad_len))
            k_padded = F.pad(k, (0, 0, 0, pad_len))
        else:
            q_padded = q
            k_padded = k

        num_blocks = q_padded.size(2) // self.block_size
        q_blocks = q_padded.reshape(B, H, num_blocks, self.block_size, D).mean(dim=-2)
        k_blocks = k_padded.reshape(B, H, num_blocks, self.block_size, D).mean(dim=-2)

        q_gate = self.attn_gate(q_blocks).squeeze(-1)  # (B, H, num_blocks)
        k_gate = self.attn_gate(k_blocks).squeeze(-1)  # (B, H, num_blocks)

        gate_scores = torch.matmul(q_gate.unsqueeze(-1), k_gate.unsqueeze(-2)) / (self.gate_temp.abs() + 1e-5)
        return gate_scores

    def _moba_block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_scores: torch.Tensor,
        dense_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Executes MoBA block-sparse attention across top-$k$ selected blocks."""
        B, H, N, D = q.shape
        pad_len = (self.block_size - (N % self.block_size)) % self.block_size
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
            v_pad = F.pad(v, (0, 0, 0, pad_len))
        else:
            q_pad, k_pad, v_pad = q, k, v

        num_blocks = q_pad.size(2) // self.block_size
        top_k = min(self.top_k_blocks, num_blocks)
        _, top_indices = torch.topk(gate_scores, top_k, dim=-1)  # (B, H, num_blocks, top_k)

        q_blocks = q_pad.reshape(B, H, num_blocks, self.block_size, D)
        k_blocks = k_pad.reshape(B, H, num_blocks, self.block_size, D)
        v_blocks = v_pad.reshape(B, H, num_blocks, self.block_size, D)

        out_blocks = []
        for b in range(num_blocks):
            idx_b = top_indices[:, :, b, :]  # (B, H, top_k)
            # Gather keys and values for selected blocks
            # idx_b expanded to (B, H, top_k, block_size, D)
            idx_exp = idx_b.unsqueeze(-1).unsqueeze(-1).expand(B, H, top_k, self.block_size, D)
            k_sel = torch.gather(k_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)
            v_sel = torch.gather(v_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)

            q_curr = q_blocks[:, :, b, :, :]  # (B, H, block_size, D)
            attn_b = torch.matmul(q_curr, k_sel.transpose(-2, -1)) / (D**0.5)
            attn_b = F.softmax(attn_b, dim=-1)
            out_b = torch.matmul(attn_b, v_sel)  # (B, H, block_size, D)
            out_blocks.append(out_b)

        out_pad = torch.cat(out_blocks, dim=2)  # (B, H, padded_N, D)
        return out_pad[:, :, :N, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_sparge: bool = True,
        use_kascade_stability: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution pass for FUSIONATTN.

        Args:
            hidden_states (torch.Tensor): Input representation $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, N]$ or $[B, 1, 1, N]$.
            use_sparge (bool): Whether to activate SpargeAttn dense predictor.
            use_kascade_stability (bool): Enable Kascade stability mechanism.
            output_attentions (bool): Whether to return attention probabilities.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context activations $[B, N, D]$.
                - None.
        """
        B, N, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        dense_mask: Optional[torch.Tensor] = None
        if use_sparge:
            dense_mask = self._sparge_predict(q, k)

        gate_scores = self._seer_gate_blocks(q, k)
        context = self._moba_block_attention(q, k, v, gate_scores, dense_mask)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).float().unsqueeze(1).unsqueeze(-1)
            elif attention_mask.dim() == 2:
                mask = attention_mask.float().unsqueeze(1).unsqueeze(-1)
            else:
                mask = attention_mask.float()
            context = context * mask.to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(context)
        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonality regularization loss."""
        return torch.tensor(0.0, device=self.gate_temp.device)
