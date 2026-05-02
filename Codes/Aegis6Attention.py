"""
Aegis‑Attention v2: Hybrid Sliding‑Window + Linear Attention with
    - Low‑rank projections (SVD + LoRA, trainable singular values)
    - Overlapping sliding‑window local attention (Flash‑friendly)
    - Query‑aware + input‑driven gating for linear global attention
    - Token‑wise fusion of local and global outputs
    - RMS pre‑norm for kernel features
    - Numerically stable kernel feature map (orthogonal random features)
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------
@dataclass
class AegisConfigV2:
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    window_size: int = 64          # sliding window size (local attention)
    svd_rank: int = 128            # rank for truncated SVD of original weights
    low_rank_update: int = 16      # LoRA rank
    kernel_feature_dim: int = 128  # m = 2 * head_dim by default
    dropout: float = 0.1
    use_query_gate: bool = True    # query‑aware gating on Q_feat
    use_input_gate: bool = True    # input‑driven gating on K,V
    use_pre_norm: bool = True      # RMS norm before kernel map


# -------------------------------------------------------------------------
# 2. Low‑Rank Projection with trainable singular values
# -------------------------------------------------------------------------
class LowRankProjectionV2(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 svd_rank: int, lora_rank: int,
                 pretrained_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.svd_rank = svd_rank
        self.lora_rank = lora_rank

        # Truncated SVD of pretrained weight (or random init)
        if pretrained_weight is not None:
            U, S, Vt = torch.linalg.svd(pretrained_weight.detach(), full_matrices=False)
            U = U[:, :svd_rank].clone()
            S = S[:svd_rank].clone()
            Vt = Vt[:svd_rank, :].clone()
        else:
            U = torch.randn(out_features, svd_rank) * 0.02
            S = torch.ones(svd_rank)
            Vt = torch.randn(svd_rank, in_features) * 0.02

        # Trainable singular values (initialised from pretrained)
        self.U = nn.Parameter(U, requires_grad=False)      # freeze left vectors
        self.S = nn.Parameter(S)                            # trainable!
        self.Vt = nn.Parameter(Vt, requires_grad=False)     # freeze right vectors

        # LoRA adapter (full rank = lora_rank)
        self.A = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.B = nn.Parameter(torch.zeros(lora_rank, in_features))
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_base = self.U @ torch.diag(self.S) @ self.Vt
        W_lora = self.A @ self.B
        W = W_base + W_lora
        return F.linear(x, W)


# -------------------------------------------------------------------------
# 3. Numerically Stable Kernel Feature Map (RMS pre‑norm available)
# -------------------------------------------------------------------------
class KernelFeatureMapV2(nn.Module):
    def __init__(self, head_dim: int, m: int):
        super().__init__()
        self.head_dim = head_dim
        self.m = m
        # Orthogonal random matrix (fixed)
        self.register_buffer('projection', self._orthogonal_random_matrix(head_dim, m))
        self.scaling = 1.0 / math.sqrt(m)

    @staticmethod
    def _orthogonal_random_matrix(d: int, m: int):
        q, _ = torch.linalg.qr(torch.randn(d, m))
        return q * math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., head_dim)
        features = torch.matmul(x, self.projection)           # (..., m)
        norm = (x * x).sum(dim=-1, keepdim=True) / 2.0
        # Cast to float64 for exp stability, then back
        exp_in = (features - norm).to(torch.float64)
        phi = self.scaling * torch.exp(exp_in).to(x.dtype)
        return phi


# -------------------------------------------------------------------------
# 4. Gates (query‑aware, input‑driven, token‑wise fusion)
# -------------------------------------------------------------------------
class GatesV2(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 use_query_gate: bool, use_input_gate: bool):
        super().__init__()
        self.use_query_gate = use_query_gate
        self.use_input_gate = use_input_gate
        if use_query_gate:
            # Gate scores from Q (per head, per token)
            self.query_gate = nn.Linear(head_dim, 1, bias=False)
        if use_input_gate:
            # Gate scores from input hidden states (mask irrelevant tokens)
            self.input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.input_gate.bias, 2.0)  # bias toward open

    def forward(self, hidden_states, Q):
        """
        Returns:
            q_gate  : (B, num_heads, N, 1)  or None
            in_gate : (B, num_heads, N, 1)  or None
        """
        q_gate = None
        in_gate = None
        if self.use_query_gate:
            # Q: (B, num_heads, N, head_dim)
            q_gate = torch.sigmoid(self.query_gate(Q))       # (B, h, N, 1)
        if self.use_input_gate:
            # hidden_states: (B, N, hidden_size)
            logits = self.input_gate(hidden_states)          # (B, N, h)
            in_gate = torch.sigmoid(logits).permute(0, 2, 1).unsqueeze(-1)  # (B, h, N, 1)
        return q_gate, in_gate


# -------------------------------------------------------------------------
# 5. Aegis‑Attention v2 core
# -------------------------------------------------------------------------
class AegisAttentionV2(nn.Module):
    def __init__(self, config: AegisConfigV2,
                 pretrained_qkv: Optional[dict] = None):
        super().__init__()
        self.cfg = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = config.hidden_size
        self.window_size = config.window_size
        self.dropout = config.dropout

        # Pretrained weights for initialisation
        Q_p = pretrained_qkv['q'] if pretrained_qkv else None
        K_p = pretrained_qkv['k'] if pretrained_qkv else None
        V_p = pretrained_qkv['v'] if pretrained_qkv else None
        O_p = pretrained_qkv['o'] if pretrained_qkv else None

        # Low‑rank projections
        self.q_proj = LowRankProjectionV2(config.hidden_size, self.all_head_size,
                                         config.svd_rank, config.low_rank_update, Q_p)
        self.k_proj = LowRankProjectionV2(config.hidden_size, self.all_head_size,
                                         config.svd_rank, config.low_rank_update, K_p)
        self.v_proj = LowRankProjectionV2(config.hidden_size, self.all_head_size,
                                         config.svd_rank, config.low_rank_update, V_p)
        self.out_proj = LowRankProjectionV2(self.all_head_size, config.hidden_size,
                                           config.svd_rank, config.low_rank_update, O_p)

        # RMS pre‑norm for kernel features
        self.use_pre_norm = config.use_pre_norm
        if config.use_pre_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Kernel feature map
        self.kernel_map = KernelFeatureMapV2(self.head_dim, config.kernel_feature_dim)

        # Gates
        self.gates = GatesV2(config.hidden_size, config.num_heads, config.head_dim,
                            config.use_query_gate, config.use_input_gate)

        # Token‑wise fusion gate (local vs global)
        self.fusion_gate = nn.Linear(config.hidden_size, config.num_heads, bias=True)
        nn.init.constant_(self.fusion_gate.bias, 1.0)  # initially biased toward global (soft)

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _create_sliding_window_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Create a band mask for sliding window attention:
        (N, N) with 1 inside the window, 0 outside.
        """
        idx = torch.arange(N, device=device)
        dist = torch.abs(idx.unsqueeze(-1) - idx.unsqueeze(-2))
        mask = (dist < self.window_size).to(torch.float32)
        # Convert to additive mask for scaled_dot_product_attention:
        #  0.0 -> allowed, -inf -> masked
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

    def _local_sliding_attention(self, Q, K, V, mask_pad):
        """
         Q,K,V: (B, num_heads, N, head_dim)
         mask_pad: (B, 1, 1, N) with 1 for valid, 0 for pad
        """
        B, h, N, d = Q.shape
        # Sliding window mask (N, N)
        window_mask = self._create_sliding_window_mask(N, Q.device)  # (N, N) with 0 / -inf
        # Combine with padding mask: pad tokens must not be attended to
        # mask_pad: (B, 1, 1, N) -> (B, 1, N, 1) for keys
        # We'll add both: (B, 1, N, N) additive mask
        pad_mask_K = mask_pad.to(Q.dtype)                           # (B, 1, 1, N)
        # For query padding: we also want to mask out whole rows if query is pad (though output later discarded)
        build_mask = window_mask.unsqueeze(0).unsqueeze(0)          # (1, 1, N, N)
        # If key is pad, set mask to -inf
        build_mask = build_mask + (1.0 - pad_mask_K.transpose(-2, -1)) * -1e9  # correct: pad_mask_K is (B,1,1,N) -> (B,1,N,1)
        # If query is pad, set entire row to -inf (though we'll ignore those positions later)
        build_mask = build_mask + (1.0 - mask_pad) * -1e9           # mask_pad: (B,1,1,N)

        # Use PyTorch's optimized scaled_dot_product_attention
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=build_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return attn_out

    def _global_linear_attention(self, Q, K, V, mask_pad,
                                 q_gate, in_gate):
        """
        Linear attention with query‑aware and input gating.
        mask_pad: (B, 1, 1, N) 1=valid
        """
        B, h, N, d = Q.shape
        m = self.cfg.kernel_feature_dim

        # Pre‑norm for stability
        if self.use_pre_norm:
            Q_for_kernel = self.q_norm(Q)
            K_for_kernel = self.k_norm(K)
        else:
            Q_for_kernel = Q
            K_for_kernel = K

        Q_feat = self.kernel_map(Q_for_kernel)   # (B, h, N, m)
        K_feat = self.kernel_map(K_for_kernel)   # (B, h, N, m)
        V_val = V

        # Query‑aware gating: modulate Q features
        if q_gate is not None:
            Q_feat = Q_feat * q_gate

        # Input gating: mask irrelevant tokens in K and V
        if in_gate is not None:
            K_feat = K_feat * in_gate
            V_val = V_val * in_gate

        # Padding mask: zero out padded positions
        pad_float = mask_pad.to(Q.dtype)          # (B, 1, 1, N)
        K_feat = K_feat * pad_float.transpose(-2, -1)
        V_val = V_val * pad_float.transpose(-2, -1)

        # Accumulate KV and normaliser
        KV = torch.matmul(K_feat.transpose(-2, -1), V_val)   # (B, h, m, d)
        Z = K_feat.sum(dim=-2, keepdim=True)                 # (B, h, 1, m)

        # Numerator and denominator
        num = torch.matmul(Q_feat, KV)                       # (B, h, N, d)
        denom = torch.matmul(Q_feat, Z.transpose(-2, -1))    # (B, h, N, 1)
        denom = denom.clamp(min=1e-6)                         # avoid NaN
        context = num / denom
        return context

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N_orig, _ = hidden_states.shape
        device = hidden_states.device

        # DistilBERT‑style mask: 1 = valid, 0 = pad
        if attention_mask is not None:
            mask_pad = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
        else:
            mask_pad = torch.ones(B, 1, 1, N_orig, device=device, dtype=hidden_states.dtype)

        # Low‑rank projections
        Q = self._reshape_to_heads(self.q_proj(hidden_states))
        K = self._reshape_to_heads(self.k_proj(hidden_states))
        V = self._reshape_to_heads(self.v_proj(hidden_states))

        # Compute gates (query‑aware, input‑driven)
        q_gate, in_gate = self.gates(hidden_states, Q)

        # 1. Local sliding‑window attention
        local_out = self._local_sliding_attention(Q, K, V, mask_pad)

        # 2. Global linear attention
        global_out = self._global_linear_attention(Q, K, V, mask_pad, q_gate, in_gate)

        # 3. Token‑wise fusion
        fusion_logits = self.fusion_gate(hidden_states)                # (B, N, h)
        gate_val = torch.sigmoid(fusion_logits).permute(0, 2, 1).unsqueeze(-1)  # (B, h, N, 1)
        fused_out = (1 - gate_val) * local_out + gate_val * global_out

        # Reshape back to (B, N_orig, all_head_size)
        fused_out = fused_out.permute(0, 2, 1, 3).contiguous()
        fused_out = fused_out.view(B, N_orig, self.all_head_size)

        # Output projection
        output = self.out_proj(fused_out)
        return output, None   # compatible return format

    @staticmethod
    def svd_init_from_pretrained(pretrained_layer):
        """Extract weight matrices from a DistilBertAttention layer."""
        return {
            'q': pretrained_layer.q_lin.weight.data.clone().detach(),
            'k': pretrained_layer.k_lin.weight.data.clone().detach(),
            'v': pretrained_layer.v_lin.weight.data.clone().detach(),
            'o': pretrained_layer.out_lin.weight.data.clone().detach()
        }


# -------------------------------------------------------------------------
# 6. Injection helper
# -------------------------------------------------------------------------
def inject_aegis_v2(model, config: AegisConfigV2):
    from transformers import DistilBertAttention
    for name, module in model.named_children():
        if isinstance(module, DistilBertAttention):
            pretrained = AegisAttentionV2.svd_init_from_pretrained(module)
            new_attn = AegisAttentionV2(config, pretrained)
            setattr(model, name, new_attn)
        else:
            inject_aegis_v2(module, config)