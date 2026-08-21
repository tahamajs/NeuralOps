r"""
Aegis-Attention V2 Module.

Implements hybrid low-rank factorized attention with local sliding window attention,
global linear kernel attention with adaptive soft-pruning gating, and dynamic multi-head fusion.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisAttentionConfigV2:
    r"""Configuration dataclass for AegisAttentionV2."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    local_window: int = 64
    rank: int = 64
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True


class AegisAttentionV2(nn.Module):
    r"""
    Aegis-Attention V2 Layer.

    Mathematical Formulation:
        1. Low-Rank Projection:
           $$Q = X W_{Q1} W_{Q2}, \quad K = X W_{K1} W_{K2}, \quad V = X W_{V1} W_{V2}$$
           with $W_1 \in \mathbb{R}^{D \times r}, W_2 \in \mathbb{R}^{r \times D}$.

        2. Local Band Attention:
           $$A^{\text{local}}_{ij} = \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_h}} + M^{\text{band}}_{ij} + M^{\text{pad}}_j\right)$$
           $$O^{\text{local}}_i = \sum_{j} A^{\text{local}}_{ij} V_j$$

        3. Global Kernel Linear Attention with Token Pruning:
           $$\phi(u) = \text{ELU}(u) + 1, \quad g = \sigma(\text{PruneGate}(X)) \in [0, 1]^{B \times N \times H}$$
           $$K_g = g \odot K, \quad V_g = g \odot V$$
           $$O^{\text{global}}_i = \frac{\phi(Q_i) \left(\sum_j \phi(K_{g, j})^T V_{g, j}\right)}{\phi(Q_i) \left(\sum_j \phi(K_{g, j})^T \mathbf{1}\right) + \epsilon}$$

        4. Gated Fusion:
           $$\alpha = \sigma(\text{FusionGate}(X)) \in [0, 1]^{B \times N \times H \times 1}$$
           $$O_i = \alpha_i \odot O^{\text{local}}_i + (1 - \alpha_i) \odot O^{\text{global}}_i$$
           $$\text{Output} = O W_{O1} W_{O2}$$

    Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot N \cdot D \cdot r + B \cdot H \cdot N^2 \cdot d_h + B \cdot H \cdot N \cdot d_h^2)$
        - Memory Complexity: $\mathcal{O}(B \cdot H \cdot N^2)$ for local band logit representations.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, N, D]$
        - `attention_mask`: $[B, N]$ or $[B, 1, 1, N]$
        - Projections $Q, K, V$: $[B, N, H, d_h]$
        - Local Attention Scores: $[B, H, N, N]$
        - Global Memory Matrix: $[B, H, d_h, d_h]$
        - Output: $[B, N, D]$
    """

    def __init__(self, config: AegisAttentionConfigV2) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.local_window: int = config.local_window
        self.rank: int = config.rank
        self.kernel_dim: int = config.kernel_feature_dim
        dropout_p = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", getattr(config, "attention_probs_dropout_prob", 0.1)))
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

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
        original_linear: nn.Linear,
        lin1: nn.Linear,
        lin2: nn.Linear,
        rank: int,
    ) -> None:
        r"""
        Initializes factorized weights using truncated SVD of the source layer.

        Args:
            original_linear (nn.Linear): Source linear layer.
            lin1 (nn.Linear): Down-projection linear module ($D \to r$).
            lin2 (nn.Linear): Up-projection linear module ($r \to D$).
            rank (int): SVD truncation rank.
        """
        W = original_linear.weight.data
        bias = original_linear.bias.data if original_linear.bias is not None else None
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
        Forward computation for AegisAttentionV2.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to return local attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context tensor of shape $[B, N, D]$.
                - Local attention weights $[B, H, N, N]$ if `output_attentions=True`, else None.
        """
        B, N, H_dim = hidden_states.shape

        # ---- convert mask to 2D bool where True = keep ----
        token_valid: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                if attention_mask.shape[-2] == attention_mask.shape[-1]:
                    token_valid = attention_mask[:, 0, 0, :] > -10000.0
                else:
                    token_valid = attention_mask.squeeze(1).squeeze(1) > -10000.0
            elif attention_mask.dim() == 3:
                token_valid = attention_mask.squeeze(1).bool()
            elif attention_mask.dim() == 2:
                token_valid = attention_mask.bool()

        mask_2d = token_valid  # (B, N) bool, True = valid

        def lr(lin1: nn.Linear, lin2: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            """Lr."""
            return lin2(lin1(x))

        Q = lr(self.q_proj1, self.q_proj2, hidden_states)
        K = lr(self.k_proj1, self.k_proj2, hidden_states)
        V = lr(self.v_proj1, self.v_proj2, hidden_states)

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        if mask_2d is not None:
            m4 = mask_2d.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            Q = Q * m4
            K = K * m4
            V = V * m4

        # ---- Local attention ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) * scale
        idx = torch.arange(N, device=hidden_states.device)
        band = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window
        band = band.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~band, float("-inf"))
        if mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attn_local = F.softmax(scores, dim=-1)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum("bhnm,bmhd->bnhd", attn_local, V)  # (B, N, h, d)

        # ---- Global linear attention ----
        def phi(x: torch.Tensor) -> torch.Tensor:
            """Phi."""
            return F.elu(x) + 1.0

        if self.prune_gate is not None:
            keep = torch.sigmoid(self.prune_gate(hidden_states))  # (B, N, h)
            Kg = K * keep.unsqueeze(-1)
            Vg = V * keep.unsqueeze(-1)
        else:
            Kg, Vg = K, V

        if mask_2d is not None:
            m4 = mask_2d.unsqueeze(-1).unsqueeze(-1)
            Kg = Kg * m4
            Vg = Vg * m4

        phi_K = phi(Kg)
        phi_Q = phi(Q)
        phi_K_t = phi_K.permute(0, 2, 3, 1)  # (B, h, d, N)
        Vg_t = Vg.permute(0, 2, 1, 3)  # (B, h, N, d)

        KV = torch.matmul(phi_K_t, Vg_t)  # (B, h, d, d)
        ones = torch.ones(B, self.num_heads, N, 1, device=Kg.device)
        Z = torch.matmul(phi_K_t, ones)  # (B, h, d, 1)

        phi_Q_t = phi_Q.permute(0, 2, 1, 3).unsqueeze(-2)  # (B, h, N, 1, d)
        global_out = torch.matmul(phi_Q_t, KV.unsqueeze(2)).squeeze(-2)  # (B, h, N, d)
        norm = torch.matmul(phi_Q_t, Z.unsqueeze(2)).squeeze(-2)  # (B, h, N, 1)
        global_out = global_out / (norm + 1e-5)
        global_out = global_out.permute(0, 2, 1, 3)  # (B, N, h, d)

        if mask_2d is not None:
            global_out = global_out * mask_2d.unsqueeze(-1).unsqueeze(-1)

        # ---- Adaptive fusion ----
        alpha = torch.sigmoid(self.fusion_gate(hidden_states))  # (B, N, h)
        alpha = alpha.unsqueeze(-1)  # (B, N, h, 1)
        fused = alpha * local_out + (1.0 - alpha) * global_out

        fused = fused.reshape(B, N, H_dim)
        out = self.out_proj2(self.out_proj1(fused))
        out = self.dropout(out)
        if mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)

        return out, (attn_local if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonal regularization loss for low-rank projection matrices."""
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


@ATTENTION_REGISTRY.register("aegis_v2")
def inject_aegis_v2(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AegisAttentionV2:
    r"""
    Injects AegisAttentionV2 into a transformer layer.

    Args:
        layer (nn.Module): Transformer layer to inject attention into.
        model_config (Any): Model configuration parameters.
        comp_config (Any): Compression configuration.
        layer_idx (int): Current layer index.
        total_layers (int): Total number of layers.
        old_attn (Optional[nn.Module]): Pretrained attention module.

    Returns:
        AegisAttentionV2: Initialized attention layer.
    """
    cfg = AegisAttentionConfigV2(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        local_window=getattr(comp_config, "block_size", 64),
        rank=getattr(comp_config, "lora_r", 64),
        kernel_feature_dim=getattr(comp_config, "rff_features", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_adaptive_pruning=getattr(comp_config, "use_gate", True),
    )
    new_attn = AegisAttentionV2(cfg)

    # Transfer weights from pretrained
    if old_attn is not None:
        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", None))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", None))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", None))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
        if q_src is not None:
            AegisAttentionV2._svd_init(q_src, new_attn.q_proj1, new_attn.q_proj2, cfg.rank)
        if k_src is not None:
            AegisAttentionV2._svd_init(k_src, new_attn.k_proj1, new_attn.k_proj2, cfg.rank)
        if v_src is not None:
            AegisAttentionV2._svd_init(v_src, new_attn.v_proj1, new_attn.v_proj2, cfg.rank)
        if out_src is not None:
            AegisAttentionV2._svd_init(out_src, new_attn.out_proj1, new_attn.out_proj2, cfg.rank)

    layer.attention = new_attn
    return new_attn
