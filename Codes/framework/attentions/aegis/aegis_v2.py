"""
Aegis‑Attention V2: Low‑rank projections, local dense window,
global linear kernel attention, and adaptive soft masking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

@dataclass
class AegisAttentionConfigV2:
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    local_window: int = 64
    rank: int = 64
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True


class AegisAttentionV2(nn.Module):
    def __init__(self, config: AegisAttentionConfigV2):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.local_window = config.local_window
        self.rank = config.rank
        self.kernel_dim = config.kernel_feature_dim
        self.dropout = nn.Dropout(config.dropout)

        self.q_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.q_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.k_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.v_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.v_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        self.out_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.out_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        if config.use_adaptive_pruning:
            self.prune_gate = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, self.num_heads)
            )
        else:
            self.prune_gate = None

        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)

    @staticmethod
    def _svd_init(original_linear, lin1, lin2, rank):
        W = original_linear.weight.data
        bias = original_linear.bias.data if original_linear.bias is not None else None
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        lin1.weight.data = Vh_r
        lin2.weight.data = U_r @ torch.diag(S_r)
        if bias is not None:
            lin2.bias.data = bias

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, H = hidden_states.shape
        # ---- convert mask to 2D bool where True = keep ----
        token_valid = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                if attention_mask.shape[-2] == attention_mask.shape[-1]:
                    token_valid = attention_mask[:, 0, 0, :] > -1e9
                else:
                    token_valid = attention_mask.squeeze(1).squeeze(1) > -1e9
            elif attention_mask.dim() == 3:
                token_valid = attention_mask.squeeze(1).bool()
            elif attention_mask.dim() == 2:
                token_valid = attention_mask.bool()

        mask_2d = token_valid   # (B,N) bool, True = valid

        def lr(lin1, lin2, x): return lin2(lin1(x))
        Q = lr(self.q_proj1, self.q_proj2, hidden_states)
        K = lr(self.k_proj1, self.k_proj2, hidden_states)
        V = lr(self.v_proj1, self.v_proj2, hidden_states)

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        if mask_2d is not None:
            m4 = mask_2d.unsqueeze(-1).unsqueeze(-1)   # (B,N,1,1)
            Q = Q * m4
            K = K * m4
            V = V * m4

        # ---- Local attention ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum('bnhd,bmhd->bhnm', Q, K) * scale
        idx = torch.arange(N, device=hidden_states.device)
        band = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window
        band = band.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~band, float('-inf'))
        if mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)   # (B,1,1,N)
            scores = scores.masked_fill(~key_mask, float('-inf'))

        attn_local = F.softmax(scores, dim=-1)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum('bhnm,bmhd->bnhd', attn_local, V)   # (B,N,h,d)

        # ---- Global linear attention ----
        phi = lambda x: F.elu(x) + 1

        if self.prune_gate is not None:
            keep = torch.sigmoid(self.prune_gate(hidden_states))   # (B,N,h)
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
        phi_K_t = phi_K.permute(0, 2, 3, 1)          # (B,h,d,N)
        Vg_t = Vg.permute(0, 2, 1, 3)                # (B,h,N,d)

        KV = torch.matmul(phi_K_t, Vg_t)              # (B,h,d,d)
        ones = torch.ones(B, self.num_heads, N, 1, device=Kg.device)
        Z = torch.matmul(phi_K_t, ones)               # (B,h,d,1)

        phi_Q_t = phi_Q.permute(0, 2, 1, 3).unsqueeze(-2)   # (B,h,N,1,d)
        global_out = torch.matmul(phi_Q_t, KV.unsqueeze(2)).squeeze(-2)  # (B,h,N,d)
        norm = torch.matmul(phi_Q_t, Z.unsqueeze(2)).squeeze(-2)          # (B,h,N,1)
        global_out = global_out / (norm + 1e-8)
        global_out = global_out.permute(0, 2, 1, 3)     # (B,N,h,d)

        if mask_2d is not None:
            global_out = global_out * mask_2d.unsqueeze(-1).unsqueeze(-1)

        # ---- Adaptive fusion ----
        alpha = torch.sigmoid(self.fusion_gate(hidden_states))   # (B,N,h)
        alpha = alpha.unsqueeze(-1)   # (B,N,h,1)
        fused = alpha * local_out + (1 - alpha) * global_out

        fused = fused.reshape(B, N, H)
        out = self.out_proj2(self.out_proj1(fused))
        out = self.dropout(out)
        if mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)

        return (out, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


# ------------------------------------------------------------
# Registry integration
# ------------------------------------------------------------
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("aegis_v2")
def inject_aegis_v2(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = AegisAttentionConfigV2(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        local_window=getattr(comp_config, 'block_size', 64),
        rank=getattr(comp_config, 'lora_r', 64),
        kernel_feature_dim=getattr(comp_config, 'rff_features', 128),
        dropout=getattr(comp_config, 'dropout', 0.1),
        use_adaptive_pruning=getattr(comp_config, 'use_gate', True),
    )
    new_attn = AegisAttentionV2(cfg)

    # Transfer weights from pretrained
    AegisAttentionV2._svd_init(old_attn.q_lin, new_attn.q_proj1, new_attn.q_proj2, cfg.rank)
    AegisAttentionV2._svd_init(old_attn.k_lin, new_attn.k_proj1, new_attn.k_proj2, cfg.rank)
    AegisAttentionV2._svd_init(old_attn.v_lin, new_attn.v_proj1, new_attn.v_proj2, cfg.rank)
    AegisAttentionV2._svd_init(old_attn.out_lin, new_attn.out_proj1, new_attn.out_proj2, cfg.rank)

    layer.attention = new_attn
    return new_attn