"""Aegis-Hybrid: low‑rank projections + local window + global linear + adaptive pruning."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisHybridConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    local_window: int = 64
    rank: int = 64
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True


class AegisHybridAttention(nn.Module):
    def __init__(self, config: AegisHybridConfig):
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
                nn.Linear(self.hidden_size // 4, self.num_heads),
            )
        else:
            self.prune_gate = None
        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)

    @staticmethod
    def _svd_init(orig_lin, lin1, lin2, rank):
        W = orig_lin.weight.data
        bias = orig_lin.bias.data if orig_lin.bias is not None else None
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        lin1.weight.data = Vh_r
        lin2.weight.data = U_r @ torch.diag(S_r)
        if bias is not None:
            lin2.bias.data = bias

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, H = hidden_states.shape

        def lr(p1, p2, x):
            return p2(p1(x))

        Q = lr(self.q_proj1, self.q_proj2, hidden_states)  # (B, N, H)
        K = lr(self.k_proj1, self.k_proj2, hidden_states)
        V = lr(self.v_proj1, self.v_proj2, hidden_states)

        Q = Q.view(B, N, self.num_heads, self.head_dim)  # (B, N, h, d)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        # -------- padding mask (2D bool, True = keep) --------
        mask_2d = None
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
        band_mask = (
            idx.unsqueeze(1) - idx.unsqueeze(0)
        ).abs() <= self.local_window  # (N, N)
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
        def phi(x):
            return F.elu(x) + 1

        if self.prune_gate is not None:
            keep = torch.sigmoid(self.prune_gate(hidden_states))  # (B, N, h)
            Kg = K * keep.unsqueeze(-1)  # (B, N, h, d)
            Vg = V * keep.unsqueeze(-1)
        else:
            Kg, Vg = K, V

        # zero out padded tokens again (mask_4d already applied to K,V but we re-check Kg,Vg)
        if mask_2d is not None:
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
        alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(
            -1
        )  # (B, N, h, 1)
        fused = alpha * local_out + (1 - alpha) * global_out  # (B, N, h, d)

        # ============== 4. Output projection ==============
        fused = fused.reshape(B, N, H)
        out = self.out_proj2(self.out_proj1(fused))
        out = self.dropout(out)

        if mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)  # zero out padding in output

        return (out, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


# ------------------------------------------------------------
# Injection function – registers 'aegis_hybrid' in the registry
# ------------------------------------------------------------
@ATTENTION_REGISTRY.register("aegis_hybrid")
def inject_aegis_hybrid(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
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
    new_attn._svd_init(old_attn.q_lin, new_attn.q_proj1, new_attn.q_proj2, cfg.rank)
    new_attn._svd_init(old_attn.k_lin, new_attn.k_proj1, new_attn.k_proj2, cfg.rank)
    new_attn._svd_init(old_attn.v_lin, new_attn.v_proj1, new_attn.v_proj2, cfg.rank)
    new_attn._svd_init(
        old_attn.out_lin, new_attn.out_proj1, new_attn.out_proj2, cfg.rank
    )
    layer.attention = new_attn
    return new_attn
