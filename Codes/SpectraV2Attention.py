"""
SPECTRA‑v2 – Sparse Per‑head Efficient Context Transformer with Rectified Attention
Fix: true weighted inter‑block top‑k + order‑preserving FASA via relative bias.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpectraConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    # Block structure
    block_size: int = 64
    top_k_inter: int = 16          # per query token, how many inter‑block keys to keep
    # Intra‑block: always full (order preserved)
    # Inter‑block: selected from immediate neighbours (prev + next blocks)
    # Order preservation
    use_rel_pos_bias: bool = True
    rel_pos_max_distance: int = 128
    # Adaptive span (if desired, keep per‑head)
    adaptive_span: bool = True
    max_span: float = 512.0
    # Global rectification (every N layers)
    rectify_interval: int = 4
    rff_features: int = 128
    sph_features: int = 128
    aks_enabled: bool = True


class SpectraV2Attention(nn.Module):
    def __init__(self, model_config, spectra_cfg: SpectraConfig, layer_idx: int, total_layers: int, bert_mode=False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.cfg = spectra_cfg
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.bert_mode = bert_mode

        # Standard projections
        self.query = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.value = nn.Linear(model_config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, model_config.hidden_size)
        self.dropout = nn.Dropout(spectra_cfg.dropout)

        # Adaptive span (optional)
        if spectra_cfg.adaptive_span:
            self.span = nn.Parameter(torch.full((self.num_heads,), spectra_cfg.max_span, dtype=torch.float))
        else:
            self.span = None

        # Relative position bias (order‑preserving inter‑block selection)
        self.rel_pos_bias = None
        if spectra_cfg.use_rel_pos_bias:
            max_dist = spectra_cfg.rel_pos_max_distance
            # Learnable bias: (num_heads, 2*max_dist+1)
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.num_heads, 2 * max_dist + 1))
        self.max_rel_dist = spectra_cfg.rel_pos_max_distance if spectra_cfg.use_rel_pos_bias else 0

        # Global rectification kernel (every N layers)
        self.rect_interval = spectra_cfg.rectify_interval
        self.is_rectify_layer = (layer_idx % self.rect_interval == 0) and spectra_cfg.aks_enabled
        if self.is_rectify_layer:
            from .hypernova_attention import AdaptiveKernelSwitch  # reuse HYPERNOVA component
            self.aks = AdaptiveKernelSwitch(
                self.head_dim,
                spectra_cfg.rff_features,
                spectra_cfg.sph_features
            )
        else:
            self.aks = None

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).transpose(1, 2)   # (B, H, N, D)

    def _adaptive_span_logit(self, N: int, device: torch.device):
        """Return a log-weight mask of shape (H, N, N) that softly limits the attention span."""
        if self.span is None:
            return 0.0
        span = self.span.clamp(min=1.0)            # (H,)
        idx = torch.arange(N, device=device, dtype=torch.float)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().unsqueeze(0)   # (1, N, N)
        # Soft mask: large negative logit outside span, ~0 inside
        logit = torch.log((F.sigmoid((span.unsqueeze(1).unsqueeze(1) - dist) * 5.0) + 1e-9))
        return logit   # (H, N, N)

    def _get_token_valid_mask(self, mask, B, N, device):
        """Convert any attention mask to (B, N) boolean (True = valid)."""
        if mask is None:
            return None
        if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
            token_valid = mask[:, 0, 0, :] > -1e9
        elif mask.dim() == 4:
            token_valid = mask.squeeze(1).squeeze(1) > -1e9
        elif mask.dim() == 3:
            token_valid = mask.squeeze(1).bool()
        elif mask.dim() == 2:
            token_valid = mask.bool()
        else:
            raise ValueError(f"Unexpected mask dim: {mask.dim()}")
        return token_valid.to(device)

    def _block_sparse_attention(self, Q, K, V, token_mask):
        """
        Intra‑block full + inter‑block per‑query top‑k with relative bias.
        Each query token attends to:
          - all tokens in its own block (full, order preserved)
          - up to top_k_inter tokens selected from the previous and next block.
        """
        B, H, N, D = Q.shape
        scale = self.head_dim ** -0.5
        block_size = self.cfg.block_size
        num_blocks = (N + block_size - 1) // block_size
        top_k = self.cfg.top_k_inter

        # 1) Adaptive span log mask (applied to intra scores only)
        span_logit = self._adaptive_span_logit(N, Q.device)   # (H, N, N)

        # 2) Precompute relative position indices for bias
        if self.rel_pos_bias is not None:
            # index = (pos_q - pos_k) + max_dist, clipped
            pos = torch.arange(N, device=Q.device)
            rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N)
            rel_pos_clipped = rel_pos.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
            # rel_pos_clipped: (N, N) of values [0, 2*max_dist]

        outputs = []
        for b_idx in range(num_blocks):
            start = b_idx * block_size
            end = min(start + block_size, N)
            Q_b = Q[:, :, start:end, :]                     # (B, H, Bs, D)
            Bs = end - start

            # ---- Intra‑block keys/values (always full) ----
            K_intra = K[:, :, start:end, :]
            V_intra = V[:, :, start:end, :]

            # ---- Inter‑block candidates from left and right ----
            inter_keys = []
            inter_vals  = []
            inter_positions = []   # store global positions for bias
            if b_idx > 0:
                p_start = max(0, (b_idx-1)*block_size)
                p_end   = start
                inter_keys.append(K[:, :, p_start:p_end, :])
                inter_vals.append(V[:, :, p_start:p_end, :])
                inter_positions.append(torch.arange(p_start, p_end, device=Q.device))
            if b_idx < num_blocks - 1:
                n_start = end
                n_end   = min(N, (b_idx+2)*block_size)   # next block only (can extend)
                inter_keys.append(K[:, :, n_start:n_end, :])
                inter_vals.append(V[:, :, n_start:n_end, :])
                inter_positions.append(torch.arange(n_start, n_end, device=Q.device))

            if len(inter_keys) > 0:
                K_inter = torch.cat(inter_keys, dim=2)   # (B, H, Li, D)
                V_inter = torch.cat(inter_vals, dim=2)
                pos_inter = torch.cat(inter_positions, dim=0)   # (Li,)
                Li = K_inter.size(2)

                # Compute raw scores for inter tokens (before bias)
                scores_inter = torch.matmul(Q_b, K_inter.transpose(-2, -1)) * scale   # (B, H, Bs, Li)

                # Add relative position bias (order‑preserving selection)
                if self.rel_pos_bias is not None:
                    q_pos = torch.arange(start, end, device=Q.device).view(1, 1, Bs, 1)
                    k_pos = pos_inter.view(1, 1, 1, Li)
                    rel = (q_pos - k_pos).clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
                    # rel: (1, 1, Bs, Li) with values 0..2*max_dist
                    bias = self.rel_pos_bias[:, rel]   # (H, Bs, Li) -> expects (B,H,Bs,Li) after expand
                    # bias is (H, Bs, Li), we need (B, H, Bs, Li)
                    bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
                    scores_inter = scores_inter + bias

                # Per‑query token top‑k selection from inter tokens
                if Li > top_k:
                    # Select top_k for each query token independently
                    topk_vals, topk_idx = scores_inter.topk(top_k, dim=-1)   # (B, H, Bs, top_k)
                    # Gather the selected inter keys/values
                    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
                    K_inter_sel = K_inter.unsqueeze(2).expand(-1, -1, Bs, -1, -1).gather(3, topk_idx_expanded)
                    V_inter_sel = V_inter.unsqueeze(2).expand(-1, -1, Bs, -1, -1).gather(3, topk_idx_expanded)
                    # K_inter_sel: (B, H, Bs, top_k, D)
                    K_inter_sel = K_inter_sel.view(B, H, Bs * top_k, D)   # flatten for concatenation
                    V_inter_sel = V_inter_sel.view(B, H, Bs * top_k, D)
                else:
                    K_inter_sel = K_inter
                    V_inter_sel = V_inter

                # Concatenate intra + selected inter
                K_total = torch.cat([K_intra, K_inter_sel], dim=2)   # (B, H, Bs + Bs*topk, D)
                V_total = torch.cat([V_intra, V_inter_sel], dim=2)
            else:
                K_total = K_intra
                V_total = V_intra

            # Compute scores with combined set
            scores = torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale   # (B, H, Bs, K_total_len)

            # Apply adaptive span only to intra part (first Bs keys)
            intra_scores = scores[:, :, :, :Bs]
            intra_scores = intra_scores + span_logit[:, start:end, start:end].unsqueeze(0)
            scores[:, :, :, :Bs] = intra_scores

            # Apply token validity mask (padding) if provided
            if token_mask is not None:
                # We need to build a mask for the combined keys. Intra keys come from same block, inter keys from their original positions.
                # We'll fetch the validity of the inter keys using token_mask.
                intra_valid = token_mask[:, start:end].unsqueeze(1).unsqueeze(-1)   # (B, 1, Bs, 1)
                if len(inter_keys) > 0:
                    inter_valid = token_mask[:, pos_inter].unsqueeze(1).unsqueeze(2)   # (B, 1, 1, Li)
                    if Li > top_k:
                        # Gather validity for the selected tokens
                        inter_valid = inter_valid.expand(-1, -1, end-start, -1).gather(3, topk_idx.unsqueeze(0))  # (B, 1, Bs, top_k)
                        inter_valid = inter_valid.view(B, 1, Bs * top_k)
                    else:
                        inter_valid = inter_valid.view(B, 1, Li)
                    total_valid = torch.cat([intra_valid.squeeze(-1), inter_valid], dim=-1)  # (B, 1, Bs+len)
                else:
                    total_valid = intra_valid.squeeze(-1)   # (B, 1, Bs)
                scores = scores.masked_fill(total_valid.unsqueeze(2) == 0, -10000.0)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_b = torch.matmul(attn, V_total)   # (B, H, Bs, D)
            outputs.append(out_b)

        out = torch.cat(outputs, dim=2)   # (B, H, N, D)
        return out

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, D = hidden_states.shape
        H = self.num_heads

        Q = self._reshape(self.query(hidden_states))
        K = self._reshape(self.key(hidden_states))
        V = self._reshape(self.value(hidden_states))

        # Token validity mask
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        token_mask = self._get_token_valid_mask(mask, B, N, Q.device)

        if self.is_rectify_layer:
            # Global linear attention via AKS (like HYPERNOVA)
            Q_feat, K_feat = self.aks(Q, K, training=self.training)
            KV = torch.matmul(K_feat.transpose(-2, -1), V)
            Z = K_feat.sum(dim=2, keepdim=True)
            context = torch.matmul(Q_feat, KV) / torch.matmul(Q_feat, Z.transpose(-2, -1))
        else:
            # Block sparse attention with true top‑k weighted sum
            context = self._block_sparse_attention(Q, K, V, token_mask)

        # Apply token validity to output
        if token_mask is not None:
            context = context * token_mask.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(self.dropout(context))
        return (output, None) if output_attentions else (output,)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.weight.device)