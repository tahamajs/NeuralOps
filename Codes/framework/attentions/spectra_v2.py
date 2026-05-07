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
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@dataclass
class SpectraConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    # Block structure
    block_size: int = 64
    top_k_inter: int = 16          # per query token, how many inter‑block keys to keep
    # Order preservation
    use_rel_pos_bias: bool = True
    rel_pos_max_distance: int = 128
    # Adaptive span (if desired, keep per‑head)
    adaptive_span: bool = True
    max_span: float = 512.0
    # Global rectification (every N layers)
    rectify_interval: int = 4
    rff_features: int = 128
    aks_enabled: bool = True


# ----------------------------------------------------------------------
# Simple RFF kernel for global rectification (if AKS not available)
# ----------------------------------------------------------------------
class SimpleRFFKernel(nn.Module):
    def __init__(self, head_dim: int, rff_dim: int, sph_dim: Optional[int] = None):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(head_dim, rff_dim) * 0.02)

    def forward(self, q: torch.Tensor, k: torch.Tensor, training: bool = True):
        def rff(x):
            p = torch.matmul(x, self.proj)
            return torch.cat([torch.cos(p), torch.sin(p)], dim=-1) / (self.proj.size(1) ** 0.25)
        return rff(q), rff(k)


# ----------------------------------------------------------------------
# Main SPECTRA‑v2 Attention (no external dependencies)
# ----------------------------------------------------------------------
class SpectraV2Attention(nn.Module):
    def __init__(self, model_config, spectra_cfg: SpectraConfig,
                 layer_idx: int, total_layers: int, bert_mode=False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = spectra_cfg
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.bert_mode = bert_mode

        # Standard projections
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key   = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)
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
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.num_heads, 2 * max_dist + 1))
        self.max_rel_dist = spectra_cfg.rel_pos_max_distance if spectra_cfg.use_rel_pos_bias else 0

        # Global rectification kernel (every N layers)
        self.rect_interval = spectra_cfg.rectify_interval
        self.is_rectify_layer = (layer_idx % self.rect_interval == 0) and spectra_cfg.aks_enabled
        if self.is_rectify_layer:
            self.aks = SimpleRFFKernel(self.head_dim, spectra_cfg.rff_features)
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

        # Adaptive span log mask (applied to intra scores only)
        span_logit = self._adaptive_span_logit(N, Q.device)   # (H, N, N)

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
                n_end   = min(N, (b_idx+2)*block_size)   # next block only
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
                    # rel: (1, 1, Bs, Li)
                    bias = self.rel_pos_bias[:, rel.squeeze(0).squeeze(0)]   # (H, Bs, Li)
                    bias = bias.unsqueeze(0).expand(B, -1, -1, -1)           # (B, H, Bs, Li)
                    scores_inter = scores_inter + bias

                # Per‑query token top‑k selection from inter tokens
                if Li > top_k:
                    _, topk_idx = scores_inter.topk(top_k, dim=-1)   # (B, H, Bs, top_k)
                    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
                    K_inter_sel = K_inter.unsqueeze(2).expand(-1, -1, Bs, -1, -1).gather(3, topk_idx_expanded)
                    V_inter_sel = V_inter.unsqueeze(2).expand(-1, -1, Bs, -1, -1).gather(3, topk_idx_expanded)
                    # Flatten the inter tokens across the block dimension
                    K_inter_sel = K_inter_sel.view(B, H, Bs * top_k, D)
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
            scores = torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale   # (B, H, Bs, total_len)

            # Apply adaptive span only to intra part (first Bs keys)
            intra_scores = scores[:, :, :, :Bs]
            intra_scores = intra_scores + span_logit[:, start:end, start:end].unsqueeze(0)
            scores[:, :, :, :Bs] = intra_scores

            # Apply token validity mask (padding) if provided
            if token_mask is not None:
                # Intra keys validity (from same block)
                intra_valid = token_mask[:, start:end].unsqueeze(1).unsqueeze(-1)   # (B, 1, Bs, 1)
                if len(inter_keys) > 0:
                    inter_valid = token_mask[:, pos_inter].unsqueeze(1).unsqueeze(2)   # (B, 1, 1, Li)
                    if Li > top_k:
                        inter_valid = inter_valid.expand(-1, -1, Bs, -1).gather(3, topk_idx.unsqueeze(0))
                        inter_valid = inter_valid.view(B, 1, Bs * top_k)
                    else:
                        inter_valid = inter_valid.view(B, 1, Li)
                    total_valid = torch.cat([intra_valid.squeeze(-1), inter_valid], dim=-1)  # (B, 1, Bs+len)
                else:
                    total_valid = intra_valid.squeeze(-1)   # (B, 1, Bs)
                scores = scores.masked_fill(total_valid.unsqueeze(2) == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_b = torch.matmul(attn, V_total)   # (B, H, Bs, D)
            outputs.append(out_b)

        out = torch.cat(outputs, dim=2)   # (B, H, N, D)
        return out

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
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
            context = torch.matmul(Q_feat, KV) / (torch.matmul(Q_feat, Z.transpose(-2, -1)) + 1e-8)
        else:
            # Block sparse attention with true top‑k weighted sum
            context = self._block_sparse_attention(Q, K, V, token_mask)

        # Apply token validity to output
        if token_mask is not None:
            context = context * token_mask.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(self.dropout(context))

        # Always return a tuple (output, attentions) – second may be None
        return (output, None) if not output_attentions else (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.weight.device)


# ----------------------------------------------------------------------
# Registration and injection (for use with the framework registry)
# ----------------------------------------------------------------------
@ATTENTION_REGISTRY.register("spectra_v2")
def inject_spectra_v2(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """
    Registry‑compatible injection function.
    comp_config can be a dict or dataclass with overrides.
    """
    cfg = SpectraConfig(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        dropout=getattr(comp_config, 'dropout', 0.1),
        block_size=getattr(comp_config, 'block_size', 64),
        top_k_inter=getattr(comp_config, 'top_k_inter', 16),
        use_rel_pos_bias=getattr(comp_config, 'use_rel_pos_bias', True),
        rel_pos_max_distance=getattr(comp_config, 'rel_pos_max_distance', 128),
        adaptive_span=getattr(comp_config, 'adaptive_span', True),
        max_span=getattr(comp_config, 'max_span', 512.0),
        rectify_interval=getattr(comp_config, 'rectify_interval', 4),
        rff_features=getattr(comp_config, 'rff_features', 128),
        aks_enabled=getattr(comp_config, 'aks_enabled', True),
    )
    new_attn = SpectraV2Attention(
        model_config=model_config,
        spectra_cfg=cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    # Copy pretrained weights (if old_attn is provided)
    if old_attn is not None:
        new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
        new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
        new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
        new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
        new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
        new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
        new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
        new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    layer.attention = new_attn
    return new_attn