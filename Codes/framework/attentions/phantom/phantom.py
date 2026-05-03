"""
PHANTOM — Parameter‑Hashed AttentioN with Token‑wise Orthogonal Mapping
CUR‑hashed projections + LSH attention for DistilBERT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PHANTOMConfig:
    cur_rank: int = 16
    lsh_rounds: int = 4
    lsh_bucket_size: int = 32
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Softpick activation (sparse‑friendly rectified softmax)
# ---------------------------------------------------------------------------
def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


# ---------------------------------------------------------------------------
# CUR‑based factorised linear layer
# ---------------------------------------------------------------------------
class FactorisedLinear_CUR(nn.Module):
    """
    Approximate a weight matrix W via CUR decomposition:
      W ≈ C @ U @ R, rank = r.
    C and R are column and row subsets respectively, U is a small linking matrix.
    The compressed representation uses two linear layers:
      down: x -> R space (input to row subspace)
      up:   -> output (column subspace)
    Weights are randomly initialised from CUR.
    """
    def __init__(self, pretrained_weight: torch.Tensor, pretrained_bias: Optional[torch.Tensor],
                 rank: int):
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        self.rank = rank

        # Perform CUR decomposition (approximate leverage score sampling)
        with torch.no_grad():
            W = pretrained_weight.float()
            # Compute leverage scores for rows and columns using SVD
            U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)
            # Column leverage scores (from V)
            col_leverage = (Vt_full ** 2).sum(dim=0)
            col_leverage /= col_leverage.sum()
            # Row leverage scores (from U)
            row_leverage = (U_full ** 2).sum(dim=1)
            row_leverage /= row_leverage.sum()

            # Sample r rows and columns with probability proportional to leverage
            # For simplicity, take top‑r according to scores (deterministic)
            _, col_idx = torch.topk(col_leverage, k=rank, largest=True)
            _, row_idx = torch.topk(row_leverage, k=rank, largest=True)

            C = W[:, col_idx]                            # (out_dim, r)
            R = W[row_idx, :]                            # (r, in_dim)
            W_intersection = W[row_idx][:, col_idx]      # (r, r)
            # Compute pseudoinverse of the intersection to get U
            U_int = torch.linalg.pinv(W_intersection.float())  # (r, r)

            # CUR weights: W ≈ C @ U_int @ R
            # We'll embed C and (U_int @ R) into the two linear layers.
            # down: weights = R (so that down(x) = R @ x), output size = r
            # up:   weights = C @ U_int (size: out_dim × r)
            self.down = nn.Linear(in_dim, rank, bias=False)
            self.up   = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))

            self.down.weight.data = R.float()
            self.up.weight.data   = (C @ U_int).float()

            if pretrained_bias is not None:
                self.up.bias.data = pretrained_bias.float()
            else:
                self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


# ---------------------------------------------------------------------------
# LSH attention (angular hash)
# ---------------------------------------------------------------------------
def lsh_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  rounds: int, bucket_size: int,
                  softpick_temp: float = 1.0,
                  dropout: Optional[nn.Module] = None,
                  token_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute attention using multi‑round angular LSH.
    Q, K, V: (B, heads, N, D)
    token_valid: (B, N) boolean mask (True = keep)
    """
    B, H, N, D = Q.shape
    scale = D ** -0.5

    # Normalise for angular LSH
    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)

    out = torch.zeros_like(V)
    for r in range(rounds):
        # Generate random hyperplanes for hashing
        # Use a different projection each round (shared across batch and heads)
        hash_proj = torch.randn(D, bucket_size, device=Q.device, dtype=Q.dtype)
        hash_proj = F.normalize(hash_proj, p=2, dim=0)

        # Hash code: sign(Q_norm @ proj) > 0 → bucket index
        # Turn into a binary code and reduce to integer
        hash_code_q = (torch.matmul(Q_norm, hash_proj) > 0).int()   # (B, H, N, bucket_size)
        hash_code_q = hash_code_q.mul(torch.tensor([2**i for i in range(bucket_size)],
                                                   device=Q.device).view(1,1,1,-1)).sum(dim=-1)  # (B, H, N)

        hash_code_k = (torch.matmul(K_norm, hash_proj) > 0).int()
        hash_code_k = hash_code_k.mul(torch.tensor([2**i for i in range(bucket_size)],
                                                   device=Q.device).view(1,1,1,-1)).sum(dim=-1)

        # For each head, sort by hash code and perform chunked attention within each bucket
        for b in range(B):
            for h in range(H):
                q_h = Q[b, h]                     # (N, D)
                k_h = K[b, h]
                v_h = V[b, h]
                if token_valid is not None:
                    valid = token_valid[b]
                    q_h = q_h[valid]
                    k_h = k_h[valid]
                    v_h = v_h[valid]
                    hq = hash_code_q[b, h][valid]
                    hk = hash_code_k[b, h][valid]
                    N_eff = q_h.size(0)
                    if N_eff == 0:
                        continue
                else:
                    hq = hash_code_q[b, h]
                    hk = hash_code_k[b, h]
                    N_eff = N

                # Sort queries by hash code
                sorted_q_idx = torch.argsort(hq)
                q_sorted = q_h[sorted_q_idx]      # (N_eff, D)
                v_sorted = v_h[sorted_q_idx]
                # For keys, we need to lookup by the same sorting; we'll compute scores in a double loop (simplified)
                # Actually efficient bucket attention would use a sorted list. For brevity, we do naive O(N^2) per head
                # but reduce the compute by only attending within same bucket.
                # I'll implement bucket gathering:
                buckets = {}
                for i, code in enumerate(hk):
                    buckets.setdefault(code.item(), []).append(i)
                # For each query, only attend to keys in the same bucket.
                attn_ctx = torch.zeros_like(q_sorted)
                for i, q_i in enumerate(q_sorted):
                    code = hq[sorted_q_idx[i]].item()
                    key_indices = buckets.get(code, [])
                    if not key_indices:
                        continue
                    k_bucket = k_h[key_indices]      # (L_b, D)
                    v_bucket = v_h[key_indices]
                    # Compute scores with softpick
                    scores = (q_i * scale) @ k_bucket.T
                    # Apply softpick
                    attn_w = softpick(scores.unsqueeze(0), dim=-1, temperature=softpick_temp).squeeze(0)
                    if dropout is not None and dropout.p > 0.:
                        attn_w = dropout(attn_w)
                    ctx = attn_w @ v_bucket
                    attn_ctx[i] = ctx
                # Unsort
                unsorted_idx = torch.argsort(sorted_q_idx)
                attn_unsorted = attn_ctx[unsorted_idx]
                # Place back into output
                if token_valid is not None:
                    out[b, h][valid] = attn_unsorted
                else:
                    out[b, h] = attn_unsorted
    return out


# ---------------------------------------------------------------------------
# Main PHANTOM Attention module
# ---------------------------------------------------------------------------
class PHANTOMAttention(nn.Module):
    def __init__(self, model_config, phantom_cfg: PHANTOMConfig,
                 pretrained_attn=None, bert_mode: bool = False):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("PHANTOM requires pretrained_attn for CUR init")
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = phantom_cfg

        # CUR‑factorised projections (rank = cur_rank)
        self.q_proj = FactorisedLinear_CUR(pretrained_attn.q_lin.weight.data,
                                           pretrained_attn.q_lin.bias.data,
                                           phantom_cfg.cur_rank)
        self.k_proj = FactorisedLinear_CUR(pretrained_attn.k_lin.weight.data,
                                           pretrained_attn.k_lin.bias.data,
                                           phantom_cfg.cur_rank)
        self.v_proj = FactorisedLinear_CUR(pretrained_attn.v_lin.weight.data,
                                           pretrained_attn.v_lin.bias.data,
                                           phantom_cfg.cur_rank)
        self.out_proj = FactorisedLinear_CUR(pretrained_attn.out_lin.weight.data,
                                             pretrained_attn.out_lin.bias.data,
                                             phantom_cfg.cur_rank)

        self.dropout = nn.Dropout(phantom_cfg.dropout)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        # Convert mask to token_valid boolean
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        token_valid = None
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()

        # LSH attention with multi‑round
        context = lsh_attention(
            Q, K, V,
            rounds=self.cfg.lsh_rounds,
            bucket_size=self.cfg.lsh_bucket_size,
            softpick_temp=self.cfg.softpick_temperature if self.cfg.softpick_enabled else 1.0,
            dropout=self.dropout if self.cfg.softpick_enabled else None,
            token_valid=token_valid,
        )

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(context)
        return (output, None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


# ------------------------------------------------------------
# Framework injection & registration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("phantom")
def inject_phantom(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = PHANTOMConfig(
        cur_rank=getattr(comp_config, 'cur_rank', 16),
        lsh_rounds=getattr(comp_config, 'lsh_rounds', 4),
        lsh_bucket_size=getattr(comp_config, 'lsh_bucket_size', 32),
        softpick_enabled=True,
        dropout=getattr(comp_config, 'dropout', 0.1),
    )
    new_attn = PHANTOMAttention(model_config, cfg, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn