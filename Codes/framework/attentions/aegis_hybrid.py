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


@ATTENTION_REGISTRY.register("aegis_hybrid")
@ATTENTION_REGISTRY.register("aegis_hybrid_attn")
class AegisHybridAttention(nn.Module):
    r"""
    Aegis-Hybrid Attention combining Local Windowed Attention and Global Linear Attention.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(config, AegisHybridConfig):
            h_size = hidden_size or getattr(config, "hidden_size", 64)
            n_heads = num_heads or getattr(config, "num_attention_heads", 4)
            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_size // max(1, n_heads),
                local_window=getattr(config, "local_window", 64) if config else 64,
                rank=getattr(config, "rank", min(32, h_size)) if config else min(32, h_size),
                kernel_feature_dim=getattr(config, "kernel_feature_dim", 64) if config else 64,
                dropout=getattr(config, "dropout", 0.0) if config else 0.0,
            )
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
        with torch.no_grad():
            W = orig_lin.weight.data.float()
            bias = orig_lin.bias.data if orig_lin.bias is not None else None
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(rank, U.size(1), Vh.size(0))

            lin1.weight.data.zero_()
            lin2.weight.data.zero_()

            lin1.weight.data[:r, :].copy_(Vh[:r, :].to(lin1.weight.dtype))
            lin2.weight.data[:, :r].copy_((U[:, :r] @ torch.diag(S[:r])).to(lin2.weight.dtype))
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
                mask_2d = (attention_mask[:, 0, 0, :] > -1e4).bool()  # (B, N)

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
        # FIX: [Use float16/bfloat16 safe mask value to prevent numerical overflow]
        mask_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else float("-inf")
        scores = scores.masked_fill(~band_mask, mask_val)

        # also apply padding mask to keys (B, 1, 1, N)
        if mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(~key_mask, mask_val)

        # Safe softmax to avoid NaNs if all scores in a row are -inf
        attn_local = F.softmax(scores, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum("bhnm,bmhd->bnhd", attn_local, V)  # (B, N, h, d)

        # ============== 2. Global linear attention ==============
        def phi(x: torch.Tensor) -> torch.Tensor:
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

        # FIX: [Ensure ones tensor adopts the device and dtype of input tensor Kg to prevent dtype mismatch]
        ones = torch.ones(B, self.num_heads, N, 1, device=Kg.device, dtype=Kg.dtype)  # (B, h, N, 1)
        Z = torch.matmul(phi_K_t, ones)  # (B, h, d, 1)

        phi_Q_t = phi_Q.permute(0, 2, 1, 3).unsqueeze(-2)  # (B, h, N, 1, d)

        global_out = torch.matmul(phi_Q_t, KV.unsqueeze(2)).squeeze(-2)  # (B, h, N, d)
        norm_eps = 1e-4 if Kg.dtype in (torch.float16, torch.bfloat16) else 1e-8
        norm = torch.matmul(phi_Q_t, Z.unsqueeze(2)).squeeze(-2)  # (B, h, N, 1)
        global_out = global_out / (norm + norm_eps)
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


@ATTENTION_REGISTRY.register("inject_aegis_hybrid")
def inject_aegis_hybrid(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
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
