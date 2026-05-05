import torch
import torch.nn as nn
import math
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("lsh")
class LSHAttention(nn.Module):
    """
    Reformer's LSH Attention.
    Uses Angular LSH with multiple rounds to bucket queries and keys.
    """
    def __init__(self, config, rounds=4, bucket_size=64):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rounds = rounds
        self.bucket_size = bucket_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _hash(self, vectors, random_matrices):
        # vectors: (B, H, L, D), random_matrices: list of (D, bucket_size) tensors
        # Return hash codes for each round.
        codes = []
        for rot in random_matrices:
            proj = torch.einsum('bhnd,db->bhnb', F.normalize(vectors, dim=-1), rot)  # (B, H, L, B)
            # Sign to bits
            hash_int = (proj > 0).int().mul(torch.tensor([2**i for i in range(proj.size(-1))], device=vectors.device)).sum(-1)
            codes.append(hash_int)
        return codes

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1,2)  # (B, H, L, d)
        K = self.k_proj(hidden_states).view(B, L, H, d).transpose(1,2)
        V = self.v_proj(hidden_states).view(B, L, H, d).transpose(1,2)

        # Generate random rotation matrices for each round
        rot_mats = [torch.randn(d, self.bucket_size, device=Q.device) for _ in range(self.rounds)]
        rot_mats = [F.normalize(rm, dim=0) for rm in rot_mats]

        # Hash Q and K
        q_codes = self._hash(Q, rot_mats)   # list of (B, H, L) per round
        k_codes = self._hash(K, rot_mats)

        out = torch.zeros_like(V)
        for rnd in range(self.rounds):
            q_hash = q_codes[rnd]  # (B, H, L)
            k_hash = k_codes[rnd]
            # For each batch and head, group queries and keys by hash code
            # We'll compute attention per bucket.
            # This is O(N^2) in worst case but used for bucket size control.
            # For simplicity, we iterate over unique hash codes in each head.
            for b in range(B):
                for h in range(H):
                    # Find unique buckets for this round batch, head
                    uniq = torch.unique(torch.cat([q_hash[b,h], k_hash[b,h]]))
                    for code in uniq:
                        q_mask = (q_hash[b,h] == code)
                        k_mask = (k_hash[b,h] == code)
                        if q_mask.sum() == 0 or k_mask.sum() == 0:
                            continue
                        Q_sel = Q[b,h][q_mask]  # (Nq, d)
                        K_sel = K[b,h][k_mask]
                        V_sel = V[b,h][k_mask]
                        scale = d ** -0.5
                        scores = torch.mm(Q_sel, K_sel.T) * scale
                        # Apply mask (optional)
                        attn = F.softmax(scores, dim=-1)
                        ctx = torch.mm(attn, V_sel)
                        # Scatter back
                        idx = torch.nonzero(q_mask, as_tuple=True)[0]
                        out[b,h, idx] += ctx

        context = out.transpose(1,2).contiguous().view(B, L, D)
        return self.out_proj(context), None