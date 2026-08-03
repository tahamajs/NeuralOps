# framework/attentions/specialized/lsh_attention.py
"""
Locality‑Sensitive Hashing (LSH) Attention – Reformer style.
Paper: https://arxiv.org/abs/2001.04451
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("lsh")
class LSHAttention(nn.Module):
    """
    Multi‑round LSH attention with angular hashing. Outputs are averaged across rounds.
    """

    def __init__(
        self,
        config,
        rounds: int = 4,
        bucket_size: int = 64,
        pretrained_attn: nn.Module = None,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rounds = rounds
        self.bucket_size = bucket_size  # number of hash bits

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
        with torch.no_grad():
            self.q_proj.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.q_proj.bias.copy_(src.q_lin.bias)
            self.k_proj.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.k_proj.bias.copy_(src.k_lin.bias)
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _hash(self, vectors, proj):
        """Angular LSH: sign of dot product with random hyperplanes."""
        norm = F.normalize(vectors, p=2, dim=-1)  # (B, H, N, D)
        dots = torch.matmul(norm, proj)  # (B, H, N, bucket_size)
        bits = (dots > 0).to(torch.long)
        multipliers = 2 ** torch.arange(
            self.bucket_size, device=vectors.device, dtype=torch.long
        )
        return (bits * multipliers).sum(dim=-1)  # (B, H, N)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, L, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        Q = (
            self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )  # (B, H, L, d)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        # Prepare mask for key‑padding (will be used per round)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(Q.device)  # (B, L)
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
        else:
            valid = None

        out = torch.zeros_like(V)
        scale = d**-0.5

        for r in range(self.rounds):
            # Generate random projection matrix (bucket_size hyperplanes)
            proj = torch.randn(d, self.bucket_size, device=Q.device)
            proj = F.normalize(proj, p=2, dim=0)

            q_hash = self._hash(Q, proj)  # (B, H, N)
            k_hash = self._hash(K, proj)

            # Process per batch and head (could be vectorised, but kept for clarity)
            for b in range(B):
                for h in range(H):
                    qh = q_hash[b, h]  # (N,)
                    kh = k_hash[b, h]
                    # Find unique hash codes present in both Q and K
                    qh_u = qh.unique()
                    kh_u = kh.unique()
                    unique_codes = qh_u[torch.isin(qh_u, kh_u)]
                    if len(unique_codes) == 0:
                        continue
                    # For each bucket
                    for code in unique_codes:
                        q_mask = qh == code
                        k_mask = kh == code
                        # Skip if masked queries
                        if valid is not None and not q_mask[valid[b]].any():
                            continue
                        Qs = Q[b, h][q_mask]  # (Nq, d)
                        Ks = K[b, h][k_mask]  # (Nk, d)
                        Vs = V[b, h][k_mask]
                        # Apply padding mask to keys if necessary
                        if valid is not None:
                            kv_valid = valid[b][k_mask].unsqueeze(-1).to(Ks.dtype)
                            Ks = Ks * kv_valid
                            Vs = Vs * kv_valid
                        scores = torch.matmul(Qs, Ks.T) * scale
                        attn = F.softmax(scores, dim=-1)
                        attn = self.dropout(attn)
                        ctx = torch.matmul(attn, Vs)
                        out[b, h][q_mask] += ctx

        # Average over rounds
        out = out / self.rounds

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, None  # We do not materialise full attention
        return out, None


def inject_lsh(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rounds = getattr(comp_config, "rounds", 4)
    bucket_size = getattr(comp_config, "bucket_size", 64)
    new_attn = LSHAttention(
        config=model_config,
        rounds=rounds,
        bucket_size=bucket_size,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
