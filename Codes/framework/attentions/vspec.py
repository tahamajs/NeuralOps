"""
VSpec++ Attention: Variational Spectral Token‑Adaptive Attention.
"""

import torch
import torch.nn as nn
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class VariationalSpectralGate(nn.Module):
    """Produces per‑batch scaling of singular values via reparameterisation."""

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, rank * 2)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, hidden_dim)
        stats = self.net(pooled)  # (B, rank*2)
        mu, log_sigma = stats.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_sigma)
        g = torch.sigmoid(z)  # (B, rank)
        return g, mu, log_sigma


class HeadGate(nn.Module):
    """Per‑example head importance scalars."""

    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_heads)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.net(pooled))  # (B, n_heads)


class TokenRouter(nn.Module):
    """Selects top‑k tokens based on learned score."""

    def __init__(self, hidden_dim, keep_ratio=0.5, min_tokens=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens

    def forward(self, x, valid_mask=None):
        B, N, D = x.shape
        scores = torch.sigmoid(self.net(x).squeeze(-1))  # (B, N)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -1e9)
        keep = max(self.min_tokens, int(N * self.keep_ratio))
        idx = scores.topk(keep, dim=1).indices
        # Gather
        routed = torch.stack([x[b, idx[b]] for b in range(B)])
        return routed, idx


class DynamicLowRankProjector(nn.Module):
    """SVD‑based low‑rank projector with dynamic gate scaling."""

    def __init__(self, W: torch.Tensor, rank: int):
        super().__init__()
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        self.register_buffer("U", U[:, :rank])
        self.register_buffer("Vh", Vh[:rank, :])
        self.S = nn.Parameter(S[:rank].clone())
        self.rank = rank

    def forward(self, x, gate):
        # x: (B, N, in_features), gate: (B, rank)
        S_eff = self.S.unsqueeze(0) * gate  # (B, rank)
        mid = torch.matmul(x, self.Vh.t())  # (B, N, rank)
        mid = mid * S_eff.unsqueeze(1)  # (B, N, rank)
        out = torch.matmul(mid, self.U.t())  # (B, N, out_features)
        return out


@ATTENTION_REGISTRY.register("vspec")
class VSpecAttention(BaseAttention):
    """
    VSpec++ Attention layer with dynamic SVD gates, head gates, and token routing.
    """

    def __init__(self, config, rank: int = 32, pretrained_attn=None):
        super().__init__(config, rank)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if pretrained_attn is None:
            raise ValueError("pretrained_attn required")

        # Projectors – use the original weight matrices (correct shape)
        # Note: original weights are (all_head_size, hidden_size)
        self.Pq = DynamicLowRankProjector(pretrained_attn.q_lin.weight.data, rank)
        self.Pk = DynamicLowRankProjector(pretrained_attn.k_lin.weight.data, rank)
        self.Pv = DynamicLowRankProjector(pretrained_attn.v_lin.weight.data, rank)
        self.Po = DynamicLowRankProjector(pretrained_attn.out_lin.weight.data, rank)

        self.sg_q = VariationalSpectralGate(self.hidden_size, rank)
        self.sg_k = VariationalSpectralGate(self.hidden_size, rank)
        self.sg_v = VariationalSpectralGate(self.hidden_size, rank)

        self.head_gate = HeadGate(self.hidden_size, self.num_heads)
        self.router = TokenRouter(self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, D = hidden_states.shape

        # Convert mask to boolean valid
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(hidden_states.device)
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
        else:
            valid = None

        # Token routing – route only valid tokens
        routed, idx = self.router(hidden_states, valid)  # idx shape (B, K)

        # Spectral gates from full sequence
        gq, _, _ = self.sg_q(hidden_states)
        gk, _, _ = self.sg_k(hidden_states)
        gv, _, _ = self.sg_v(hidden_states)

        # Low‑rank projections on routed tokens
        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)

        # Reshape to heads
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Head gates
        hg = self.head_gate(hidden_states).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim**0.5)
        scores = scores * hg

        # Create a mask for the routed tokens (no padding inside this subset)
        # The scores are already over the selected tokens only, so no additional masking needed.
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, D)

        # Output projection (using query gate for output, could be separate)
        out = self.Po(context, gq)

        # Reverse routing: keep original hidden_states for unselected tokens
        output = hidden_states.clone()
        for b in range(B):
            output[b, idx[b]] = out[b]

        if output_attentions:
            return output, attn_weights
        return output, None


def inject_vspec(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, "vspec_rank", 32)
    new_attn = VSpecAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
