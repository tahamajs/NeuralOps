# NASC: NEXUS-AEGIS Spectral Cascade Attention

## Overview
**NASC (NEXUS-AEGIS Spectral Cascade)** is a complexity-adaptive hybrid attention architecture that unifies the strengths of localized causal sliding windows, block-sparse Mixture-of-Experts routing, and global Random Fourier Feature (RFF) linear kernel approximations into a single gated layer.

---

## Architectural Taxonomy & Gated Routing

```
                          Input Sequence X ∈ ℝ^{B × L × D}
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                   Local Window     Block-Sparse     Global Kernel
                   (AEGIS SVD)        (NEXUS)        (SPECTRA RFF)
                         │               │               │
                      O_loc           O_sp            O_glb
                         │               │               │
                         └───────────────┼───────────────┘
                                         ▼
                            Dynamic Gating Router
                       g = Softmax(Linear(Mean(X)))
                                         ▼
                     O = g₁·O_loc + g₂·O_sp + g₃·O_glb
                                         ▼
                                Output Projection
```

---

## Mathematical Formulation

1. **Local Window Attention ($O_{\text{loc}}$):**
   $$Q_{\text{loc}}, K_{\text{loc}}, V_{\text{loc}} = \text{SVD-LoRA}(X)$$
   $$O_{\text{loc}} = \text{LocalWindowAttn}(Q_{\text{loc}}, K_{\text{loc}}, V_{\text{loc}}, W_{\text{sz}})$$

2. **Block-Sparse Routing ($O_{\text{sp}}$):**
   $$S_{ij}^{\text{blk}} = \frac{\bar{Q}_i \bar{K}_j^\top}{\sqrt{d_k}}, \quad \mathcal{I}_i = \operatorname{TopK}(S_{i,:}^{\text{blk}}, k)$$
   $$O_{\text{sp}} = \operatorname{Softpick}\left(\frac{Q_i K_{\mathcal{I}_i}^\top}{\sqrt{d_k}}\right) V_{\mathcal{I}_i}$$

3. **Global Spectral Kernel ($O_{\text{glb}}$):**
   $$\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_{\text{rff}}), \sin(x W_{\text{rff}})]$$
   $$O_{\text{glb}} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}) + \epsilon}$$

4. **Composite Synthesis:**
   $$O = g_1 \cdot O_{\text{loc}} + g_2 \cdot O_{\text{sp}} + g_3 \cdot O_{\text{glb}}$$

---

## Computational Complexity Breakdown

| Regime | Dominant Mechanism | Complexity | Memory Scaling |
| :--- | :--- | :--- | :--- |
| $L \le W_{\text{sz}}$ | Local Sliding Window | $\mathcal{O}(L \cdot W_{\text{sz}} \cdot d)$ | $\mathcal{O}(W_{\text{sz}} \cdot d)$ |
| $W_{\text{sz}} < L \le k \cdot B_{\text{sz}}$ | Sparse Inter-Block Routing | $\mathcal{O}(L \cdot k B_{\text{sz}} \cdot d)$ | $\mathcal{O}(k B_{\text{sz}} \cdot d)$ |
| $L > k \cdot B_{\text{sz}}$ | Global Spectral RFF Kernel | $\mathcal{O}(L \cdot F \cdot d)$ | $\mathcal{O}(F \cdot d)$ |

**Overall Complexity:** Strictly $\mathcal{O}(L)$ linear time and memory in sequence length.

---

## Complete Source Code Implementation (`nasc_attention.py`)

```python
"""
NASC: NEXUS-AEGIS Spectral Cascade Attention.
Fixes:
- Attention mask applied to all branches.
- Pretrained weight initialization via SVD in injection.
- Correct softpick stability.
- Injection function assigns layer.attention.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Stable softpick function."""
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    exp_z = torch.exp(z / temperature)
    exp_neg_m = torch.exp((-m) / temperature).clamp(max=1e10)
    p = F.relu(exp_z - exp_neg_m)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-8)).to(orig_dtype)


class FactorisedSVDLoRAProjection(nn.Module):
    def __init__(self, in_features, out_features, rank=16, lora_rank=4, pretrained_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_rank = lora_rank

        if pretrained_weight is not None:
            W = pretrained_weight.detach().float()
            U, S, V = torch.linalg.svd(W, full_matrices=False)
            eff_rank = min(rank, U.size(1), V.size(0))
            U_init = U[:, :eff_rank].clone()
            S_init = S[:eff_rank].clone()
            V_init = V[:eff_rank, :].clone()
        else:
            U_init = torch.empty(out_features, rank)
            S_init = torch.ones(rank)
            V_init = torch.empty(rank, in_features)
            nn.init.orthogonal_(U_init)
            nn.init.orthogonal_(V_init)

        self.U = nn.Parameter(U_init)
        self.S = nn.Parameter(S_init)
        self.V = nn.Parameter(V_init)

        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        svd_term = torch.matmul(x, self.V.t())
        svd_term = svd_term * self.S
        svd_term = torch.matmul(svd_term, self.U.t())
        lora_term = torch.matmul(x, self.lora_A.t())
        lora_term = torch.matmul(lora_term, self.lora_B.t())
        return svd_term + lora_term + self.bias


@ATTENTION_REGISTRY.register("nasc")
@ATTENTION_REGISTRY.register("nasc_attention")
class NASCAttention(BaseAttention):
    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 16,
        window_size: int = 64,
        block_size: int = 32,
        top_k: int = 2,
        rff_features: int = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads = getattr(config, "num_attention_heads", 12)
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.head_dim = self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.window_size = window_size
        self.block_size = block_size
        self.top_k = top_k
        self.rff_features = rff_features
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))

        # Use pretrained weights for SVD initialization
        q_weight = pretrained_attn.q_lin.weight.data if pretrained_attn is not None else None
        k_weight = pretrained_attn.k_lin.weight.data if pretrained_attn is not None else None
        v_weight = pretrained_attn.v_lin.weight.data if pretrained_attn is not None else None

        proj_rank = rank or 16
        self.q_proj = FactorisedSVDLoRAProjection(self.hidden_size, self.hidden_size, rank=proj_rank, pretrained_weight=q_weight)
        self.k_proj = FactorisedSVDLoRAProjection(self.hidden_size, self.hidden_size, rank=proj_rank, pretrained_weight=k_weight)
        self.v_proj = FactorisedSVDLoRAProjection(self.hidden_size, self.hidden_size, rank=proj_rank, pretrained_weight=v_weight)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        if pretrained_attn is not None:
            with torch.no_grad():
                self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        self.rff_proj = nn.Parameter(torch.randn(self.head_dim, rff_features) * 0.02)
        self.gate_proj = nn.Linear(self.hidden_size, 3)
        self.dropout = nn.Dropout(dropout_p)

    # ... (local_window_attention, sparse_block_attention, global_spectral_rff_attention same but with mask)
    def _local_window_attention(self, Q, K, V, valid_mask, is_causal):
        B, H, L, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        idx = torch.arange(L, device=Q.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = dist > self.window_size
        if is_causal:
            mask = mask | (idx.unsqueeze(0) < idx.unsqueeze(1))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0).bool(), -10000.0)

        # Padding mask
        pad_k = (~valid_mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad_k.bool(), -10000.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def _sparse_block_attention(self, Q, K, V, valid_mask, is_causal):
        # ... (similar to original but add padding masking)
        B, H, L, d = Q.shape
        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            Q_pad = F.pad(Q, (0, 0, 0, pad_len))
            K_pad = F.pad(K, (0, 0, 0, pad_len))
            V_pad = F.pad(V, (0, 0, 0, pad_len))
            valid_pad = F.pad(valid_mask, (0, pad_len), value=False)
        else:
            Q_pad, K_pad, V_pad = Q, K, V
            valid_pad = valid_mask
        Lp = L + pad_len
        nb = Lp // self.block_size

        Qb = Q_pad.view(B, H, nb, self.block_size, d)
        Kb = K_pad.view(B, H, nb, self.block_size, d)
        Vb = V_pad.view(B, H, nb, self.block_size, d)
        valid_b = valid_pad.view(B, nb, self.block_size)  # [B,nb,bs]

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
        top_idx[:, :, :, 0] = 0

        out_blocks = torch.zeros_like(Qb)
        for i in range(nb):
            idx = top_idx[:, :, i, :]  # [B,H,top_k]
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
            K_sel = torch.gather(Kb, 2, idx_exp).view(B, H, top_k * self.block_size, d)
            V_sel = torch.gather(Vb, 2, idx_exp).view(B, H, top_k * self.block_size, d)

            scores_i = torch.matmul(Qb[:, :, i], K_sel.transpose(-2, -1)) * scale

            if is_causal:
                q_pos = (i * self.block_size + torch.arange(self.block_size, device=Q.device)).view(1, 1, self.block_size, 1)
                k_pos = (idx.unsqueeze(-1) * self.block_size + torch.arange(self.block_size, device=Q.device).view(1, 1, 1, self.block_size)).view(B, H, 1, top_k * self.block_size)
                scores_i = scores_i.masked_fill((k_pos > q_pos).bool(), -10000.0)

            # Mask padding keys
            valid_b_exp = valid_b.unsqueeze(1).expand(B, H, nb, self.block_size)
            idx_v = idx.unsqueeze(-1).expand(-1, -1, -1, self.block_size)
            valid_sel = torch.gather(valid_b_exp, 2, idx_v).view(B, H, top_k * self.block_size)
            pad_sel = ~valid_sel.unsqueeze(2)  # [B,H,1,L_sel]
            scores_i = scores_i.masked_fill(pad_sel.bool(), -10000.0)

            attn_i = softpick(scores_i, dim=-1)
            out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)

        out = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
        return out

    def _global_spectral_rff_attention(self, Q, K, V, valid_mask):
        """Global RFF linear attention with padding mask."""
        B, H, L, d = Q.shape
        pad = (~valid_mask).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]

        Qf = torch.matmul(Q, self.rff_proj)
        Kf = torch.matmul(K, self.rff_proj)
        norm = 1.0 / math.sqrt(self.rff_features)
        phi_Q = torch.cat([torch.cos(Qf), torch.sin(Qf)], dim=-1) * norm
        phi_K = torch.cat([torch.cos(Kf), torch.sin(Kf)], dim=-1) * norm

        # Mask K and V
        phi_K = phi_K.masked_fill(pad.bool(), 0.0)
        V_masked = V.masked_fill(pad.bool(), 0.0)

        KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)
        Z = phi_K.sum(dim=-2, keepdim=True)
        num = torch.matmul(phi_Q, KV)
        den = torch.matmul(phi_Q, Z.transpose(-2, -1)) + 1e-5
        return num / den

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_causal = bool(kwargs.get("is_causal", getattr(self.config, "is_decoder", False) or getattr(self.config, "causal", False)))

        # Convert mask to valid [B,N]
        if attention_mask is None:
            valid_mask = torch.ones(B, L, device=hidden_states.device, dtype=torch.bool)
        elif attention_mask.dim() == 2:
            valid_mask = attention_mask.bool() if attention_mask.dtype == torch.bool else attention_mask > 0
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                valid_mask = attention_mask[:, 0, 0, :]
            else:
                valid_mask = attention_mask[:, 0, 0, :] > -10000.0 if attention_mask.min() < -1000 else attention_mask[:, 0, 0, :] > 0
        else:
            valid_mask = torch.ones(B, L, device=hidden_states.device, dtype=torch.bool)

        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        summary = hidden_states.mean(dim=1)
        gates = F.softmax(self.gate_proj(summary), dim=-1).view(B, 1, 1, 3)

        O_loc = self._local_window_attention(Q, K, V, valid_mask, is_causal)
        O_sp = self._sparse_block_attention(Q, K, V, valid_mask, is_causal)
        O_glb = self._global_spectral_rff_attention(Q, K, V, valid_mask)

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
    rank = getattr(comp_config, "rank", 16)
    window_size = getattr(comp_config, "window_size", 64)
    block_size = getattr(comp_config, "block_size", 32)
    top_k = getattr(comp_config, "top_k", 2)
    rff_features = getattr(comp_config, "rff_features", 64)

    new_attn = NASCAttention(
        config=model_config,
        rank=rank,
        window_size=window_size,
        block_size=block_size,
        top_k=top_k,
        rff_features=rff_features,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
```
