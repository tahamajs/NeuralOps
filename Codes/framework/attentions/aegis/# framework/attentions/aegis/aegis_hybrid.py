# framework/attentions/aegis/aegis_hybrid.py
"""Aegis‑Hybrid: low‑rank projections + local window + global linear attention + adaptive pruning."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.attentions.core.registry import ATTENTION_REGISTRY

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

        # Low‑rank projections
        self.q_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.q_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.k_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)
        self.v_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.v_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        # Low‑rank output projection
        self.out_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.out_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        # Adaptive pruning gate
        if config.use_adaptive_pruning:
            self.prune_gate = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, self.num_heads)
            )
        else:
            self.prune_gate = None

        # Fusion gate: learn per‑head scalar α
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

        # 1. Low‑rank projections
        def lr_forward(lin1, lin2, x):
            return lin2(lin1(x))
        Q = lr_forward(self.q_proj1, self.q_proj2, hidden_states)
        K = lr_forward(self.k_proj1, self.k_proj2, hidden_states)
        V = lr_forward(self.v_proj1, self.v_proj2, hidden_states)

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        # Mask handling
        mask_2d = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_2d = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_2d = (attention_mask[:, 0, 0, :] > -1e9)
        if mask_2d is not None:
            Q = Q * mask_2d.unsqueeze(1).unsqueeze(-1)
            K = K * mask_2d.unsqueeze(1).unsqueeze(-1)
            V = V * mask_2d.unsqueeze(1).unsqueeze(-1)

        # 2. Local dense attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum('bnhd,bmhd->bhnm', Q, K) * scale

        # Band mask
        idx = torch.arange(N, device=hidden_states.device)
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window
        band_mask = band_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~band_mask, float('-inf'))

        if mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, float('-inf'))

        attn_local = F.softmax(scores, dim=-1)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum('bhnm,bmhd->bnhd', attn_local, V)  # (B, N, h, d)

        # 3. Global linear attention (kernel trick)
        phi = lambda x: F.elu(x) + 1

        if self.prune_gate is not None:
            gate_logits = self.prune_gate(hidden_states)  # (B, N, h)
            keep_prob = torch.sigmoid(gate_logits)        # (B, N, h)
            K_global = K * keep_prob.unsqueeze(-1)
            V_global = V * keep_prob.unsqueeze(-1)
        else:
            K_global = K
            V_global = V

        if mask_2d is not None:
            K_global = K_global * mask_2d.unsqueeze(1).unsqueeze(-1)
            V_global = V_global * mask_2d.unsqueeze(1).unsqueeze(-1)

        phi_K = phi(K_global)   # (B, N, h, d)
        phi_Q = phi(Q)

        phi_K_t = phi_K.permute(0, 2, 3, 1)         # (B, h, d, N)
        V_global_t = V_global.permute(0, 2, 1, 3)   # (B, h, N, d)
        KV = torch.matmul(phi_K_t, V_global_t)       # (B, h, d, d)

        ones = torch.ones(B, self.num_heads, N, 1, device=hidden_states.device)
        Z = torch.matmul(phi_K_t, ones)              # (B, h, d, 1)

        phi_Q_t = phi_Q.permute(0, 2, 1, 3).unsqueeze(-2)
        global_out = torch.matmul(phi_Q_t, KV.unsqueeze(2)).squeeze(-2)  # (B, h, N, d)

        norm = torch.matmul(phi_Q_t, Z.unsqueeze(2)).squeeze(-2)
        global_out = global_out / (norm + 1e-8)
        global_out = global_out.permute(0, 2, 1, 3)   # (B, N, h, d)

        # 4. Fusion gate
        alpha = torch.sigmoid(self.fusion_gate(hidden_states))   # (B, N, h)
        alpha = alpha.unsqueeze(-1)
        fused = alpha * local_out + (1 - alpha) * global_out

        # 5. Output projection
        fused_flat = fused.reshape(B, N, H)
        output = self.out_proj2(self.out_proj1(fused_flat))
        output = self.dropout(output)

        if mask_2d is not None:
            output = output * mask_2d.unsqueeze(-1)

        return (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


# Injection function registered in the framework
@ATTENTION_REGISTRY.register("aegis_hybrid")
def inject_aegis_hybrid(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Replace DistilBERT attention with AegisHybrid attention and copy pretrained weights."""
    cfg = AegisHybridConfig(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        local_window=getattr(comp_config, 'block_size', 64),
        rank=getattr(comp_config, 'lora_r', 64),
        kernel_feature_dim=getattr(comp_config, 'rff_features', 128),
        dropout=getattr(comp_config, 'dropout', 0.1),
        use_adaptive_pruning=getattr(comp_config, 'aks_enabled', True),
    )
    new_attn = AegisHybridAttention(cfg)

    # Transfer weights using SVD
    new_attn._svd_init(old_attn.q_lin, new_attn.q_proj1, new_attn.q_proj2, cfg.rank)
    new_attn._svd_init(old_attn.k_lin, new_attn.k_proj1, new_attn.k_proj2, cfg.rank)
    new_attn._svd_init(old_attn.v_lin, new_attn.v_proj1, new_attn.v_proj2, cfg.rank)
    new_attn._svd_init(old_attn.out_lin, new_attn.out_proj1, new_attn.out_proj2, cfg.rank)

    layer.attention = new_attn
    return new_attn