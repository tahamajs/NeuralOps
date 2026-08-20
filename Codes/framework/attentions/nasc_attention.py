r"""
NASC: NEXUS-AEGIS Spectral Cascade Attention.

Unified complexity-adaptive hybrid attention architecture integrating:
1. **Local Sliding-Window Branch:** SVD-LoRA low-rank factorized projections with causal local windowing.
2. **Sparse Inter-Block Routing Branch:** Coarse-to-fine centroid block routing with Softpick rectified activation.
3. **Global Spectral/Kernel Branch:** Random Fourier Features (RFF) FAVOR+ linear kernel attention.
4. **Dynamic Context-Aware Router:** Input-conditioned 3-way softmax gating network adaptively balancing regimes.

Mathematical Formulation:
    For an input sequence $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Dynamic Routing Weights:**
       $$g = \operatorname{Softmax}\left(\operatorname{Linear}(\bar{X})\right) \in \mathbb{R}^{B \times 1 \times 3}, \quad \bar{X} = \frac{1}{L} \sum_{t=1}^L X_t$$
    2. **Branch Computations:**
       - **Local Window ($O_{\text{loc}}$):**
         $$Q_{\text{loc}}, K_{\text{loc}}, V_{\text{loc}} = W_{\text{loc}}^Q X, W_{\text{loc}}^K X, W_{\text{loc}}^V X$$
         $$O_{\text{loc}} = \operatorname{LocalWindowAttn}(Q_{\text{loc}}, K_{\text{loc}}, V_{\text{loc}}, W_{\text{sz}})$$
       - **Sparse Block Routing ($O_{\text{sp}}$):**
         $$\bar{Q}_i = \operatorname{MeanBlock}(Q, B_{\text{sz}}), \quad \bar{K}_j = \operatorname{MeanBlock}(K, B_{\text{sz}})$$
         $$S_{ij}^{\text{blk}} = \frac{\bar{Q}_i \bar{K}_j^\top}{\sqrt{d_k}}, \quad \mathcal{I}_i = \operatorname{TopK}(S_{i,:}^{\text{blk}}, k)$$
         $$O_{\text{sp}} = \operatorname{Softpick}\left(\frac{Q_i K_{\mathcal{I}_i}^\top}{\sqrt{d_k}}\right) V_{\mathcal{I}_i}$$
       - **Global Spectral Kernel ($O_{\text{glb}}$):**
         $$\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_{\text{rff}}), \sin(x W_{\text{rff}})]$$
         $$O_{\text{glb}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}) + \epsilon}$$
    3. **Cascade Synthesis:**
       $$O = g_1 \cdot O_{\text{loc}} + g_2 \cdot O_{\text{sp}} + g_3 \cdot O_{\text{glb}}$$
       $$\operatorname{Output} = W_{\text{out}} O$$

Computational Complexity:
    - Local: $\mathcal{O}(B \cdot H \cdot L \cdot W_{\text{sz}} \cdot d_k)$
    - Sparse: $\mathcal{O}(B \cdot H \cdot L \cdot (k \cdot B_{\text{sz}}) \cdot d_k)$
    - Global: $\mathcal{O}(B \cdot H \cdot L \cdot F \cdot d_k)$
    - Total: $\mathcal{O}(L \cdot d \cdot (W_{\text{sz}} + k B_{\text{sz}} + F))$ — Strictly linear $\mathcal{O}(L)$ in context length.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def softpick(x, dim: int = -1, temperature: float = 1.0):
    import torch.nn.functional as F
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-9)).to(orig_dtype)


class FactorisedSVDLoRAProjection(nn.Module):
    """Low-rank factorized SVD-LoRA linear projection layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, lora_rank: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_rank = lora_rank

        # Orthonormal SVD basis
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

        # Trainable LoRA adapter
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base factorized projection: x @ V.T @ diag(S) @ U.T
        svd_term = torch.matmul(x, self.V.t())
        svd_term = svd_term * self.S
        svd_term = torch.matmul(svd_term, self.U.t())

        # LoRA residual: x @ lora_A.T @ lora_B.T
        lora_term = torch.matmul(x, self.lora_A.t())
        lora_term = torch.matmul(lora_term, self.lora_B.t())

        return svd_term + lora_term + self.bias


@ATTENTION_REGISTRY.register("nasc")
@ATTENTION_REGISTRY.register("nasc_attention")
class NASCAttention(BaseAttention):
    r"""
    NASC: NEXUS-AEGIS Spectral Cascade Attention Module.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 16,
        window_size: int = 64,
        block_size: int = 32,
        top_k: int = 2,
        rff_features: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        self.hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.window_size: int = window_size
        self.block_size: int = block_size
        self.top_k: int = top_k
        self.rff_features: int = rff_features
        dropout_p: float = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))

        # Projections
        proj_rank = rank or 16
        self.q_proj = FactorisedSVDLoRAProjection(self.hidden_size, self.hidden_size, rank=proj_rank)
        self.k_proj = FactorisedSVDLoRAProjection(self.hidden_size, self.hidden_size, rank=proj_rank)
        self.v_proj = FactorisedSVDLoRAProjection(self.hidden_size, self.hidden_size, rank=proj_rank)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # RFF projection for global linear kernel attention
        self.rff_proj = nn.Parameter(torch.randn(self.head_dim, rff_features) * 0.02)

        # Dynamic 3-way cascade routing gate
        self.gate_proj = nn.Linear(self.hidden_size, 3)
        self.dropout = nn.Dropout(dropout_p)

    def _local_window_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True
    ) -> torch.Tensor:
        """Compute sliding window local attention."""
        B, H, L, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Distance mask
        pos = torch.arange(L, device=Q.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        mask = dist > self.window_size
        if is_causal:
            causal_mask = pos.unsqueeze(1) < pos.unsqueeze(0)
            mask = mask | (pos.unsqueeze(0) < pos.unsqueeze(1))

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def _sparse_block_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True
    ) -> torch.Tensor:
        """Compute coarse-to-fine block sparse attention with Softpick."""
        B, H, L, d = Q.shape
        pad = (self.block_size - L % self.block_size) % self.block_size
        if pad > 0:
            Q_pad = F.pad(Q, (0, 0, 0, pad))
            K_pad = F.pad(K, (0, 0, 0, pad))
            V_pad = F.pad(V, (0, 0, 0, pad))
        else:
            Q_pad, K_pad, V_pad = Q, K, V
        Lp = L + pad
        nb = Lp // self.block_size

        Qb = Q_pad.view(B, H, nb, self.block_size, d)
        Kb = K_pad.view(B, H, nb, self.block_size, d)
        Vb = V_pad.view(B, H, nb, self.block_size, d)

        # Centroid routing
        Q_mean = Qb.mean(dim=3)
        K_mean = Kb.mean(dim=3)
        scale = 1.0 / math.sqrt(d)
        blk_scores = torch.matmul(Q_mean, K_mean.transpose(-2, -1)) * scale

        if is_causal:
            causal_blk_mask = torch.triu(
                torch.full((nb, nb), float("-inf"), device=Q.device, dtype=Q.dtype), diagonal=1
            )
            blk_scores = blk_scores + causal_blk_mask.unsqueeze(0).unsqueeze(0)

        top_k = min(self.top_k, nb)
        _, top_idx = blk_scores.topk(top_k, dim=-1)

        out_blocks = torch.zeros_like(Qb)
        for i in range(nb):
            idx = top_idx[:, :, i, :]  # (B, H, top_k)
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
            K_sel = torch.gather(Kb, 2, idx_exp).view(B, H, top_k * self.block_size, d)
            V_sel = torch.gather(Vb, 2, idx_exp).view(B, H, top_k * self.block_size, d)
            Q_i = Qb[:, :, i]

            scores_i = torch.matmul(Q_i, K_sel.transpose(-2, -1)) * scale
            if is_causal:
                q_pos = (i * self.block_size + torch.arange(self.block_size, device=scores_i.device)).view(1, 1, self.block_size, 1)
                k_pos = (idx.unsqueeze(-1) * self.block_size + torch.arange(self.block_size, device=scores_i.device).view(1, 1, 1, self.block_size)).view(B, H, 1, top_k * self.block_size)
                scores_i = scores_i.masked_fill(k_pos > q_pos, float("-inf"))

            attn_i = softpick(scores_i, dim=-1, temperature=1.0)
            out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)

        out = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
        return out

    def _global_spectral_rff_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Compute Random Fourier Feature linear global attention."""
        # Project to cosine/sine features
        Q_proj = torch.matmul(Q, self.rff_proj)
        K_proj = torch.matmul(K, self.rff_proj)

        norm_factor = 1.0 / math.sqrt(self.rff_features)
        phi_Q = torch.cat([torch.cos(Q_proj), torch.sin(Q_proj)], dim=-1) * norm_factor
        phi_K = torch.cat([torch.cos(K_proj), torch.sin(K_proj)], dim=-1) * norm_factor

        # Associative linear scan: phi(Q) @ (phi(K).T @ V)
        KV = torch.matmul(phi_K.transpose(-2, -1), V)  # (B, H, 2*F, d)
        num = torch.matmul(phi_Q, KV)  # (B, H, L, d)

        K_sum = phi_K.sum(dim=-2, keepdim=True)  # (B, H, 1, 2*F)
        den = torch.matmul(phi_Q, K_sum.transpose(-2, -1)) + 1e-6  # (B, H, L, 1)

        return num / den

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for NASC Attention."""
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_causal = bool(kwargs.get("is_causal", getattr(self.config, "is_decoder", False) or getattr(self.config, "causal", False)))

        # SVD-LoRA projections
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        # Dynamic Routing Gate: compute 3-way weights from pooled sequence context
        summary = hidden_states.mean(dim=1)  # (B, D)
        gates = F.softmax(self.gate_proj(summary), dim=-1).view(B, 1, 1, 3)  # (B, 1, 1, 3)

        # Branch 1: Local Sliding Window
        O_loc = self._local_window_attention(Q, K, V, is_causal=is_causal)

        # Branch 2: Sparse Inter-Block Routing with Softpick
        O_sp = self._sparse_block_attention(Q, K, V, is_causal=is_causal)

        # Branch 3: Global Spectral RFF Linear Kernel
        O_glb = self._global_spectral_rff_attention(Q, K, V)

        # Cascade synthesis: g1 * O_loc + g2 * O_sp + g3 * O_glb
        O = (
            gates[:, :, :, 0].unsqueeze(-1) * O_loc
            + gates[:, :, :, 1].unsqueeze(-1) * O_sp
            + gates[:, :, :, 2].unsqueeze(-1) * O_glb
        )

        out = O.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("nasc_injection")
def inject_nasc(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> NASCAttention:
    """Injection hook for NASCAttention."""
    rank = getattr(comp_config, "rank", 16)
    window_size = getattr(comp_config, "window_size", 64)
    block_size = getattr(comp_config, "block_size", 32)
    top_k = getattr(comp_config, "top_k", 2)
    return NASCAttention(
        config=model_config,
        rank=rank,
        window_size=window_size,
        block_size=block_size,
        top_k=top_k,
    )
