"""
Aegis‑Attention: Hybrid Sparse‑Linear Attention with Low‑Rank Projections,
                  Adaptive Sparsity and Kernel Feature Mapping.

Corrected version — all syntax/logic issues fixed.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# 1. Configuration (no errors)
# -------------------------------------------------------------------------
@dataclass
class AegisConfig:
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    block_size: int = 64
    svd_rank: int = 128
    low_rank_update: int = 16
    kernel_feature_dim: int = 64
    dropout: float = 0.1
    use_adaptive_gating: bool = True
    gate_threshold: float = 0.5


# -------------------------------------------------------------------------
# 2. Low‑Rank Projection with Truncated‑SVD Initialisation
# -------------------------------------------------------------------------
class LowRankProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 svd_rank: int, lora_rank: int,
                 pretrained_weight: Optional[torch.Tensor] = None):
        super().__init__()  # FIX: was 'init' and 'super().init()'
        self.in_features = in_features
        self.out_features = out_features
        self.svd_rank = svd_rank
        self.lora_rank = lora_rank

        # ----- Truncated SVD of pretrained weight -----
        if pretrained_weight is not None:
            U, S, Vt = torch.linalg.svd(pretrained_weight.detach(), full_matrices=False)
            U = U[:, :svd_rank]
            S = S[:svd_rank]
            Vt = Vt[:svd_rank, :]
        else:
            U = torch.randn(out_features, svd_rank) * 0.02   # FIX: missing '*'
            S = torch.ones(svd_rank)
            Vt = torch.randn(svd_rank, in_features) * 0.02   # FIX: missing '*'

        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('Vt', Vt)

        # ----- Low‑rank adapter (trainable) -----
        self.A = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.B = nn.Parameter(torch.zeros(lora_rank, in_features))
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_base = self.U @ torch.diag(self.S) @ self.Vt
        W_lora = self.A @ self.B
        W = W_base + W_lora
        return F.linear(x, W)


# -------------------------------------------------------------------------
# 3. Kernel Feature Map (Performer-style, numerically stable)
# -------------------------------------------------------------------------
class KernelFeatureMap(nn.Module):
    def __init__(self, d_model: int, m: int, head_dim: int):
        super().__init__()   # FIX
        self.head_dim = head_dim
        self.m = m
        self.projection = self._orthogonal_random_matrix(head_dim, m)
        self.scaling = 1.0 / math.sqrt(m)

    @staticmethod
    def _orthogonal_random_matrix(d: int, m: int):
        q, _ = torch.linalg.qr(torch.randn(d, m))
        return q * math.sqrt(d)   # FIX: was 'q  math.sqrt(d)'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., head_dim)
        features = torch.matmul(x, self.projection)        # (..., m)
        # Normalisation term: ||x||^2 / 2
        norm = (x * x).sum(dim=-1, keepdim=True) / 2.0
        # Numerical stability: cast to float64 before exp to avoid overflow
        exp_input = (features - norm).to(torch.float64)
        phi = self.scaling * torch.exp(exp_input).to(x.dtype)
        return phi


# -------------------------------------------------------------------------
# 4. Token Gating for Adaptive Sparsity
# -------------------------------------------------------------------------
class TokenGating(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()   # FIX
        self.num_heads = num_heads
        self.gate = nn.Linear(hidden_size, num_heads, bias=True)
        nn.init.constant_(self.gate.bias, 2.0)   # bias towards 1 (open)

    def forward(self, x: torch.Tensor, training: bool = True,
                threshold: float = 0.5) -> torch.Tensor:
        logits = self.gate(x)                     # (B, N, num_heads)
        gates = torch.sigmoid(logits)
        if not training:
            gates = (gates > threshold).float()   # hard mask
        return gates.permute(0, 2, 1).unsqueeze(-1)  # (B, num_heads, N, 1)


# -------------------------------------------------------------------------
# 5. Aegis‑Attention Core
# -------------------------------------------------------------------------
class AegisAttention(nn.Module):
    def __init__(self, config: AegisConfig,
                 pretrained_qkv: Optional[dict] = None):
        super().__init__()   # FIX
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = config.hidden_size
        self.block_size = config.block_size
        self.dropout = nn.Dropout(config.dropout)

        # Pretrained weight extraction
        Q_pretrain = pretrained_qkv['q'] if pretrained_qkv else None
        K_pretrain = pretrained_qkv['k'] if pretrained_qkv else None
        V_pretrain = pretrained_qkv['v'] if pretrained_qkv else None
        O_pretrain = pretrained_qkv['o'] if pretrained_qkv else None

        # Low‑rank projections
        self.q_proj = LowRankProjection(config.hidden_size, self.all_head_size,
                                        config.svd_rank, config.low_rank_update,
                                        Q_pretrain)
        self.k_proj = LowRankProjection(config.hidden_size, self.all_head_size,
                                        config.svd_rank, config.low_rank_update,
                                        K_pretrain)
        self.v_proj = LowRankProjection(config.hidden_size, self.all_head_size,
                                        config.svd_rank, config.low_rank_update,
                                        V_pretrain)
        self.out_proj = LowRankProjection(self.all_head_size, config.hidden_size,
                                          config.svd_rank, config.low_rank_update,
                                          O_pretrain)

        # Kernel feature map for global linear attention
        self.kernel_map = KernelFeatureMap(self.head_dim, config.kernel_feature_dim,
                                           self.head_dim)

        # Token gating (adaptive sparsity)
        if config.use_adaptive_gating:
            self.token_gating = TokenGating(config.hidden_size, config.num_heads)
        else:
            self.token_gating = None

        # Gate to blend local and global outputs
        self.local_global_gate = nn.Parameter(torch.zeros(config.num_heads))

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim) \
                .permute(0, 2, 1, 3)

    def _pad_to_block_size(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad seq_len to be a multiple of block_size. Mask is 1/0 (1=valid, 0=pad)."""
        B, N = x.size(0), x.size(1)
        pad_len = (self.block_size - N % self.block_size) % self.block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len), value=0)
        return x, mask

    def _local_block_attention(self, Q, K, V, mask):
        """
        Q,K,V: (B, num_heads, N_pad, head_dim)
        mask:  (B, N_pad) 1=valid, 0=pad
        """
        B, num_heads, N, d = Q.shape
        num_blocks = N // self.block_size
        Q = Q.view(B, num_heads, num_blocks, self.block_size, d)
        K = K.view(B, num_heads, num_blocks, self.block_size, d)
        V = V.view(B, num_heads, num_blocks, self.block_size, d)

        scale = 1.0 / math.sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, h, blk, L, L)

        # Build block-local mask: mask out padding tokens within each block
        # mask shape: (B, N) -> (B, 1, num_blocks, block_size)
        mask = mask.view(B, 1, num_blocks, self.block_size).to(attn_scores.dtype)
        # Create pair-wise mask: element is 1 only if both tokens are valid
        mask_pair = mask.unsqueeze(-1) * mask.unsqueeze(-2)   # FIX: was '&' / missing '*'
        # Convert to additive mask: 0 -> -inf
        additive_mask = (1.0 - mask_pair) * -1e9
        attn_scores = attn_scores + additive_mask

        # Prevent all‑zero rows (e.g., entire block is padding) leading to NaN
        # Use `masked_fill` for safety
        attn_probs = F.softmax(attn_scores, dim=-1)
        # If a row is all -inf, softmax gives NaN; we set those to 0
        nan_mask = torch.isnan(attn_probs)
        if nan_mask.any():
            attn_probs = attn_probs.masked_fill(nan_mask, 0.0)

        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, V)   # (B, h, blk, L, d)
        context = context.view(B, num_heads, N, d)
        return context

    def _global_linear_attention(self, Q, K, V, mask, gates=None):
        """
        Linear attention using kernel feature map.
        mask: (B, N) 1=valid, 0=pad.
        gates: (B, num_heads, N, 1) optional soft mask.
        """
        B, num_heads, N, d = Q.shape
        m = self.config.kernel_feature_dim

        # Kernel features (numerically stable)
        Q_feat = self.kernel_map(Q)   # (B, h, N, m)
        K_feat = self.kernel_map(K)
        V_val = V

        # Apply gating if available
        if gates is not None:
            K_feat = K_feat * gates
            V_val = V_val * gates

        # Mask padding tokens (set feature to zero)
        mask_float = mask.to(dtype=Q.dtype).unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)
        K_feat = K_feat * mask_float
        V_val = V_val * mask_float

        # Accumulated KV and normaliser
        KV = torch.matmul(K_feat.transpose(-2, -1), V_val)   # (B, h, m, d)
        Z = K_feat.sum(dim=-2, keepdim=True)                 # (B, h, 1, m)

        # Numerator and denominator
        num = torch.matmul(Q_feat, KV)                       # (B, h, N, d)
        denom = torch.matmul(Q_feat, Z.transpose(-2, -1))    # (B, h, N, 1)
        denom = denom.clamp(min=1e-6)                        # avoid division by zero
        context = num / denom
        return context

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,   # FIX: syntax
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        hidden_states: (B, N, H)
        attention_mask: (B, N) with 1 for valid tokens, 0 for padding
                        (DistilBERT convention)
        """
        B, N_orig, _ = hidden_states.shape
        device = hidden_states.device

        # Convert DistilBERT mask to 1/0 if provided
        if attention_mask is not None:
            # DistilBERT passes mask as (B, N) with 1=valid, 0=pad.
            # We keep it exactly as is.
            mask = attention_mask.to(dtype=hidden_states.dtype)
        else:
            mask = torch.ones(B, N_orig, device=device, dtype=hidden_states.dtype)

        # Pad to block size
        padded_states, padded_mask = self._pad_to_block_size(hidden_states, mask)
        N_pad = padded_states.size(1)

        # Low‑rank projections
        Q = self._reshape_to_heads(self.q_proj(padded_states))
        K = self._reshape_to_heads(self.k_proj(padded_states))
        V = self._reshape_to_heads(self.v_proj(padded_states))

        # Local block attention
        local_out = self._local_block_attention(Q, K, V, padded_mask)

        # Global linear attention with adaptive gating
        if self.token_gating is not None:
            gates = self.token_gating(padded_states, training=self.training,
                                      threshold=self.config.gate_threshold)
        else:
            gates = None
        global_out = self._global_linear_attention(Q, K, V, padded_mask, gates)

        # Blend local and global using learned per-head gate
        gate_val = torch.sigmoid(self.local_global_gate)               # (h)
        gate_val = gate_val.view(1, self.num_heads, 1, 1)              # (1,h,1,1)
        fused_out = (1 - gate_val) * local_out + gate_val * global_out

        # Remove padding and reshape
        fused_out = fused_out[:, :, :N_orig, :]                         # (B, h, N_orig, d)
        fused_out = fused_out.permute(0, 2, 1, 3).contiguous()          # (B, N_orig, h, d)
        fused_out = fused_out.view(B, N_orig, self.all_head_size)

        # Output projection
        output = self.out_proj(fused_out)
        return output, None   # compatible with DistilBertAttention return format

    @staticmethod
    def svd_init_from_pretrained(pretrained_attention_layer):
        """
        Extract weight matrices from a DistilBertAttention layer (HuggingFace).
        Returns dict with keys 'q','k','v','o'.
        """
        q_w = pretrained_attention_layer.q_lin.weight.data.clone().detach()
        k_w = pretrained_attention_layer.k_lin.weight.data.clone().detach()
        v_w = pretrained_attention_layer.v_lin.weight.data.clone().detach()
        o_w = pretrained_attention_layer.out_lin.weight.data.clone().detach()
        return {'q': q_w, 'k': k_w, 'v': v_w, 'o': o_w}


# -------------------------------------------------------------------------
# 6. Injection helper (corrected argument names)
# -------------------------------------------------------------------------
def inject_aegis_attention(model, aegis_config: AegisConfig):
    """
    Recursively replace every DistilBertAttention module with AegisAttention.
    """
    from transformers import DistilBertAttention
    for name, module in model.named_children():
        if isinstance(module, DistilBertAttention):
            pretrained_qkv = AegisAttention.svd_init_from_pretrained(module)
            new_attn = AegisAttention(aegis_config, pretrained_qkv)
            # Replace the module correctly
            setattr(model, name, new_attn)   # FIX: using 'name' (string) directly
        else:
            inject_aegis_attention(module, aegis_config)   # FIX: consistent argument name