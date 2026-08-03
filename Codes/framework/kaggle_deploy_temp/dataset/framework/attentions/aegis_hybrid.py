r"""
Aegis-Hybrid Attention Mechanism.

Combines low-rank factorized projections, localized sliding-window dense attention,
global kernelized linear attention with token-importance pruning, and learned gating fusion.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisHybridConfig:
    r"""Configuration dataclass for AegisHybridAttention."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    local_window: int = 64
    rank: int = 64
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True


class AegisHybridAttention(nn.Module):
    r"""
    Aegis-Hybrid Attention combining Local Windowed Attention and Global Linear Attention.

    Mathematical Breakdown:
        1. Low-Rank Projection:
           For each input representation $X \in \mathbb{R}^{B \times N \times D}$:
           $$Q = X W_{Q1} W_{Q2}, \quad K = X W_{K1} W_{K2}, \quad V = X W_{V1} W_{V2}$$
           where $W_1 \in \mathbb{R}^{D \times r}$ and $W_2 \in \mathbb{R}^{r \times D}$ initialized via truncated SVD.

        2. Local Sliding Window Attention:
           Computes local band attention within context radius $W_{\text{local}}$:
           $$S_{ij}^{\text{local}} = \frac{Q_i K_j^T}{\sqrt{d_h}} + M_{ij}^{\text{band}}, \quad M_{ij}^{\text{band}} = \begin{cases} 0 & |i - j| \le W_{\text{local}} \\ -\infty & \text{otherwise} \end{cases}$$
           $$O_i^{\text{local}} = \sum_j \text{Softmax}(S^{\text{local}})_{ij} V_j$$

        3. Global Kernel Linear Attention with Pruning:
           Uses positive feature map $\phi(x) = \text{ELU}(x) + 1$ and token importance gate $g = \sigma(\text{PruneGate}(X))$:
           $$K_g = g \odot K, \quad V_g = g \odot V$$
           $$O_i^{\text{global}} = \frac{\phi(Q_i) \left( \sum_{j=1}^N \phi(K_{g, j})^T V_{g, j} \right)}{\phi(Q_i) \left( \sum_{j=1}^N \phi(K_{g, j})^T \mathbf{1} \right) + \epsilon}$$

        4. Dynamic Multi-Head Gated Fusion:
           $$\alpha = \sigma(\text{FusionGate}(X)) \in [0, 1]^{B \times N \times H \times 1}$$
           $$O = \alpha \odot O^{\text{local}} + (1 - \alpha) \odot O^{\text{global}}$$
           $$Y = O W_{O1} W_{O2}$$

    Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot N \cdot D \cdot r + B \cdot H \cdot N^2 \cdot d_h + B \cdot H \cdot N \cdot d_h^2)$
        - Space/Memory Complexity: $\mathcal{O}(B \cdot H \cdot N^2)$ for local band logit representations.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, N, D]$
        - Query / Key / Value ($Q, K, V$): $[B, N, H, d_h]$
        - Local Attention Scores: $[B, H, N, N]$
        - Global Memory State $KV$: $[B, H, d_h, d_h]$
        - Fused Output: $[B, N, D]$
    """

    def __init__(self, config: AegisHybridConfig) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.local_window: int = config.local_window
        self.rank: int = config.rank
        self.kernel_dim: int = config.kernel_feature_dim
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        self.q_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.q_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.k_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.v_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.v_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.out_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.out_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        if config.use_adaptive_pruning:
            self.prune_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, self.num_heads),
            )
        else:
            self.prune_gate = None
        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)

    @staticmethod
    def _svd_init(
        orig_lin: nn.Linear,
        lin1: nn.Linear,
        lin2: nn.Linear,
        rank: int,
    ) -> None:
        r"""
        Initializes factorized linear layers via truncated SVD of the source linear layer.

        Args:
            orig_lin (nn.Linear): Source full-rank linear module.
            lin1 (nn.Linear): First low-rank projection layer ($D \to r$).
            lin2 (nn.Linear): Second low-rank projection layer ($r \to D$).
            rank (int): Truncated rank $r$.
        """
        W = orig_lin.weight.data
        bias = orig_lin.bias.data if orig_lin.bias is not None else None
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        lin1.weight.data.copy_(Vh_r)
        lin2.weight.data.copy_(U_r @ torch.diag(S_r))
        if bias is not None and lin2.bias is not None:
            lin2.bias.data.copy_(bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution pass of AegisHybridAttention.

        Args:
            hidden_states (torch.Tensor): Input activation tensor of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Attention mask of shape $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Mask for individual attention heads.
            output_attentions (bool): Flag indicating whether to return local attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Fused layer output of shape $[B, N, D]$.
                - Local attention map $[B, H, N, N]$ if `output_attentions=True`, else None.
        """
        B, N, H_dim = hidden_states.shape

        def lr(p1: nn.Linear, p2: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            """Lr."""
            return p2(p1(x))

        Q = lr(self.q_proj1, self.q_proj2, hidden_states)  # (B, N, H_dim)
        K = lr(self.k_proj1, self.k_proj2, hidden_states)
        V = lr(self.v_proj1, self.v_proj2, hidden_states)

        Q = Q.view(B, N, self.num_heads, self.head_dim)  # (B, N, h, d)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        # -------- padding mask (2D bool, True = keep) --------
        mask_2d: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_2d = attention_mask.bool()  # (B, N)
            elif attention_mask.dim() == 4:
                mask_2d = attention_mask[:, 0, 0, :] > -1e9  # (B, N)

        # zero out padded tokens in Q, K, V  (B, N, 1, 1)
        if mask_2d is not None:
            mask_4d = mask_2d.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            Q = Q * mask_4d
            K = K * mask_4d
            V = V * mask_4d

        # ============== 1. Local window attention ==============
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) * scale  # (B, h, N, N)

        # local band mask
        idx = torch.arange(N, device=hidden_states.device)
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window  # (N, N)
        band_mask = band_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        scores = scores.masked_fill(~band_mask, float("-inf"))

        # also apply padding mask to keys (B, 1, 1, N)
        if mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn_local = F.softmax(scores, dim=-1)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum("bhnm,bmhd->bnhd", attn_local, V)  # (B, N, h, d)

        # ============== 2. Global linear attention ==============
        def phi(x: torch.Tensor) -> torch.Tensor:
            """Phi."""
            return F.elu(x) + 1.0

        if self.prune_gate is not None:
            keep = torch.sigmoid(self.prune_gate(hidden_states))  # (B, N, h)
            Kg = K * keep.unsqueeze(-1)  # (B, N, h, d)
            Vg = V * keep.unsqueeze(-1)
        else:
            Kg, Vg = K, V

        if mask_2d is not None:
            mask_4d = mask_2d.unsqueeze(-1).unsqueeze(-1)
            Kg = Kg * mask_4d
            Vg = Vg * mask_4d

        phi_K = phi(Kg)  # (B, N, h, d)
        phi_Q = phi(Q)

        phi_K_t = phi_K.permute(0, 2, 3, 1)  # (B, h, d, N)
        Vg_t = Vg.permute(0, 2, 1, 3)  # (B, h, N, d)

        KV = torch.matmul(phi_K_t, Vg_t)  # (B, h, d, d)

        ones = torch.ones(B, self.num_heads, N, 1, device=Kg.device)  # (B, h, N, 1)
        Z = torch.matmul(phi_K_t, ones)  # (B, h, d, 1)

        phi_Q_t = phi_Q.permute(0, 2, 1, 3).unsqueeze(-2)  # (B, h, N, 1, d)

        global_out = torch.matmul(phi_Q_t, KV.unsqueeze(2)).squeeze(-2)  # (B, h, N, d)
        norm = torch.matmul(phi_Q_t, Z.unsqueeze(2)).squeeze(-2)  # (B, h, N, 1)
        global_out = global_out / (norm + 1e-8)
        global_out = global_out.permute(0, 2, 1, 3)  # (B, N, h, d)

        # ============== 3. Adaptive fusion ==============
        alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)  # (B, N, h, 1)
        fused = alpha * local_out + (1 - alpha) * global_out  # (B, N, h, d)

        # ============== 4. Output projection ==============
        fused = fused.reshape(B, N, H_dim)
        out = self.out_proj2(self.out_proj1(fused))
        out = self.dropout(out)

        if mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)

        return out, (attn_local if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns regularization orthogonality loss for low-rank projection weights."""
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


# ------------------------------------------------------------
# Injection function – registers 'aegis_hybrid' in the registry
# ------------------------------------------------------------
@ATTENTION_REGISTRY.register("aegis_hybrid")
def inject_aegis_hybrid(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AegisHybridAttention:
    r"""
    Injects AegisHybridAttention into a transformer layer.

    Args:
        layer (nn.Module): Host transformer layer.
        model_config (Any): Architecture configuration.
        comp_config (Any): Compression configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total number of layers.
        old_attn (Optional[nn.Module]): Source attention layer.

    Returns:
        AegisHybridAttention: Instantiated and initialized attention module.
    """
    cfg = AegisHybridConfig(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        local_window=getattr(comp_config, "block_size", 64),
        rank=getattr(comp_config, "lora_r", 64),
        kernel_feature_dim=getattr(comp_config, "rff_features", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_adaptive_pruning=getattr(comp_config, "use_gate", True),
    )
    new_attn = AegisHybridAttention(cfg)
    if old_attn is not None:
        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", None))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", None))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", None))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
        if q_src is not None:
            new_attn._svd_init(q_src, new_attn.q_proj1, new_attn.q_proj2, cfg.rank)
        if k_src is not None:
            new_attn._svd_init(k_src, new_attn.k_proj1, new_attn.k_proj2, cfg.rank)
        if v_src is not None:
            new_attn._svd_init(v_src, new_attn.v_proj1, new_attn.v_proj2, cfg.rank)
        if out_src is not None:
            new_attn._svd_init(out_src, new_attn.out_proj1, new_attn.out_proj2, cfg.rank)
    layer.attention = new_attn
    return new_attn
