"""
AEGIS-Attention V6: Unified Hybrid Transformer Attention Module.
Fixes:
- Renamed injection to "aegis_v6_inject" to avoid duplicate registration.
- Added mask conversion to binary [B,1,1,N] for padding.
- Ensured AegisConfigV3 includes num_attention_heads for compatibility with BaseAttention.
- Fixed orthogonality loss placeholder.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisConfigV3:
    hidden_size: int = 768
    num_heads: int = 12
    num_attention_heads: int = 12   # alias for BaseAttention compatibility
    head_dim: int = 64
    window_size: int = 64
    svd_rank: int = 128
    low_rank_update: int = 16
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_query_gate: bool = True
    use_input_gate: bool = True
    use_pre_norm: bool = True
    long_context_threshold: int = 2048
    max_global_tokens: int = 1024
    kernel_method: str = "orthogonal"
    kernel_clamp_min: float = -20.0
    kernel_clamp_max: float = 20.0
    kernel_use_float64: bool = False
    kernel_seed: Optional[int] = None

    def __post_init__(self):
        self.num_attention_heads = self.num_heads
        self.head_dim = self.hidden_size // self.num_heads


class FactorisedLowRankProjection(nn.Module):
    # (unchanged from original, but kept for completeness)
    def __init__(self, in_features, out_features, svd_rank, lora_rank, pretrained_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if pretrained_weight is not None:
            U, S, V = torch.linalg.svd(pretrained_weight.detach().float(), full_matrices=False)
            effective_rank = min(svd_rank, U.size(1), V.size(0))
            U_init = U[:, :effective_rank].clone()
            S_init = S[:effective_rank].clone()
            V_init = V[:effective_rank, :].clone().t()
        else:
            U_init = torch.randn(out_features, svd_rank) * 0.02
            S_init = torch.ones(svd_rank)
            V_init = torch.randn(in_features, svd_rank) * 0.02

        self.register_buffer("U", U_init)
        self.S = nn.Parameter(S_init)
        self.register_buffer("V", V_init)
        self.A = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.B = nn.Parameter(torch.zeros(lora_rank, in_features))
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x):
        xV = torch.matmul(x, self.V)
        xV_S = xV * self.S.unsqueeze(0).unsqueeze(0)
        out_svd = torch.matmul(xV_S, self.U.t())
        xB = torch.matmul(x, self.B.t())
        out_lora = torch.matmul(xB, self.A.t())
        return out_svd + out_lora


@dataclass
class KernelFeatureConfig:
    head_dim: int
    num_features: int
    method: str = "orthogonal"
    clamp_min: float = -20.0
    clamp_max: float = 20.0
    use_float64_exp: bool = False
    eps: float = 1e-5
    scaling: Optional[float] = None
    projection_scale: Optional[float] = None
    seed: Optional[int] = None
    custom_proj: Optional[torch.Tensor] = None


class StableKernelFeatureMap(nn.Module):
    # (unchanged)
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.head_dim = config.head_dim
        self.m = config.num_features
        self.scale = config.scaling if config.scaling is not None else 1.0 / math.sqrt(self.m + config.eps)
        self.proj = self._build_projection()
        self.register_buffer("proj", self.proj)

    def _build_projection(self):
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        d, m = self.head_dim, self.m
        if self.cfg.method == "orthogonal":
            if m <= d:
                q, _ = torch.linalg.qr(torch.randn(d, m))
            else:
                q_full, _ = torch.linalg.qr(torch.randn(d, d))
                extra, _ = torch.linalg.qr(torch.randn(d, m - d))
                q = torch.cat([q_full, extra], dim=1)
            return q * math.sqrt(d)
        elif self.cfg.method == "random":
            return torch.randn(d, m) * math.sqrt(d)
        elif self.cfg.method == "cosine":
            # simplified; not used typically
            freq = torch.linspace(0.1, 10.0, steps=m)
            phase = torch.linspace(0.0, 2 * math.pi, steps=m)
            idx = torch.arange(d, dtype=torch.float32).unsqueeze(1)
            cols = []
            for j in range(m):
                if j % 2 == 0:
                    col = torch.cos(freq[j] * idx.squeeze(1) + phase[j])
                else:
                    col = torch.sin(freq[j] * idx.squeeze(1) + phase[j])
                cols.append(col)
            return torch.stack(cols, dim=1) * math.sqrt(d)
        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")

    def forward(self, x):
        phi = torch.matmul(x, self.proj)
        norm_sq = (x * x).sum(dim=-1, keepdim=True) / 2.0
        exp_in = phi - norm_sq
        exp_in = torch.clamp(exp_in, min=self.cfg.clamp_min, max=self.cfg.clamp_max)
        if self.cfg.use_float64_exp:
            exp_in = exp_in.to(torch.float64)
        out = self.scale * torch.exp(exp_in)
        if self.cfg.use_float64_exp and x.dtype != torch.float64:
            out = out.to(x.dtype)
        return out


class GatingModule(nn.Module):
    # unchanged
    def __init__(self, hidden_size, num_heads, head_dim, use_query, use_input):
        super().__init__()
        self.use_query = use_query
        self.use_input = use_input
        if use_query:
            self.query_gate = nn.Linear(head_dim, 1, bias=False)
        else:
            self.query_gate = None
        if use_input:
            self.input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.input_gate.bias, 2.0)
        else:
            self.input_gate = None

    def forward(self, hidden_states, Q):
        q_gate = None
        in_gate = None
        if self.use_query and self.query_gate is not None:
            q_gate = torch.sigmoid(self.query_gate(Q))
        if self.use_input and self.input_gate is not None:
            logits = self.input_gate(hidden_states)
            in_gate = torch.sigmoid(logits).permute(0, 2, 1).unsqueeze(-1)
        return q_gate, in_gate


class ContextFusionGate(nn.Module):
    # unchanged
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj = nn.Linear(hidden_size // num_heads + 2 * head_dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, hidden_states, local_out, global_out):
        B, h, N, d = local_out.shape
        hidden_per_head = hidden_states.view(B, N, h, d).permute(0, 2, 1, 3)
        concat = torch.cat([hidden_per_head, local_out, global_out], dim=-1)
        logits = self.proj(concat)
        return torch.sigmoid(logits)


@ATTENTION_REGISTRY.register("aegis_v6")
class AegisAttentionV3(BaseAttention):
    def __init__(self, config: AegisConfigV3, pretrained_qkv: Optional[Dict[str, torch.Tensor]] = None):
        # BaseAttention expects config with hidden_size and num_attention_heads
        # AegisConfigV3 now has those fields, so it's safe.
        super().__init__(config)
        self.cfg = config
        self.h = config.num_heads
        self.d = config.head_dim
        self.all_head_size = self.h * self.d
        self.window = config.window_size

        Qp = pretrained_qkv["q"] if pretrained_qkv else None
        Kp = pretrained_qkv["k"] if pretrained_qkv else None
        Vp = pretrained_qkv["v"] if pretrained_qkv else None
        Op = pretrained_qkv["o"] if pretrained_qkv else None

        self.q_proj = FactorisedLowRankProjection(config.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update, Qp)
        self.k_proj = FactorisedLowRankProjection(config.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update, Kp)
        self.v_proj = FactorisedLowRankProjection(config.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update, Vp)
        self.out_proj = FactorisedLowRankProjection(self.all_head_size, config.hidden_size, config.svd_rank, config.low_rank_update, Op)

        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.RMSNorm(self.d, eps=1e-5)
            self.k_norm = nn.RMSNorm(self.d, eps=1e-5)

        kernel_cfg = KernelFeatureConfig(
            head_dim=self.d,
            num_features=config.kernel_feature_dim,
            method=config.kernel_method,
            clamp_min=config.kernel_clamp_min,
            clamp_max=config.kernel_clamp_max,
            use_float64_exp=config.kernel_use_float64,
            seed=config.kernel_seed,
        )
        self.kernel_map = StableKernelFeatureMap(kernel_cfg)

        self.gates = GatingModule(config.hidden_size, self.h, self.d, config.use_query_gate, config.use_input_gate)
        self.fusion = ContextFusionGate(config.hidden_size, self.h, self.d)

    def _reshape(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d).permute(0, 2, 1, 3)

    def _convert_mask_to_binary(self, attention_mask, B, N, device, dtype):
        """Return binary mask [B,1,1,N] with 1 for valid, 0 for padding."""
        if attention_mask is None:
            return torch.ones(B, 1, 1, N, device=device, dtype=dtype)
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask
            else:
                valid = attention_mask > 0
            return valid.unsqueeze(1).unsqueeze(2).to(dtype)
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask[:, 0, 0, :]
            else:
                # Could be additive or 0/1
                if attention_mask.min() < -1000:
                    valid = attention_mask[:, 0, 0, :] > -10000.0
                else:
                    valid = attention_mask[:, 0, 0, :] > 0
            return valid.unsqueeze(1).unsqueeze(2).to(dtype)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    def _tiled_sliding_window_attention(self, Q, K, V, pad_mask_binary, square_mask=None):
        B, h, N, d = Q.shape
        W = self.window
        out = torch.zeros_like(Q)
        # pad_mask_binary: [B,1,1,N] with 1 valid
        # Convert to additive for SDPA: (1 - mask) * -10000
        pad_additive = (1.0 - pad_mask_binary) * -10000.0

        num_tiles = (N + W - 1) // W
        for i in range(num_tiles):
            start = i * W
            end = min(start + W, N)
            k_start = max(0, start - W)
            k_end = min(N, end + W)

            q_tile = Q[:, :, start:end, :]
            k_tile = K[:, :, k_start:k_end, :]
            v_tile = V[:, :, k_start:k_end, :]

            # Key mask
            kv_mask = pad_additive[:, :, :, k_start:k_end]
            # Query mask (not used for masking scores in SDPA, but we can zero output later)
            q_mask = pad_mask_binary[:, :, :, start:end]

            # Sliding window mask (additive)
            q_idx = torch.arange(start, end, device=Q.device)
            kv_idx = torch.arange(k_start, k_end, device=Q.device)
            dist = (q_idx.unsqueeze(-1) - kv_idx.unsqueeze(0)).abs()
            win_mask = (dist < W).float()  # 1 if within window
            win_add = (1.0 - win_mask) * -10000.0

            attn_mask = kv_mask + win_add.unsqueeze(0).unsqueeze(0)

            if square_mask is not None:
                tile_sq = square_mask[:, :, start:end, k_start:k_end]
                attn_mask = attn_mask + tile_sq

            # SDPA expects mask shape [B, H, q_len, k_len] or [B, 1, q_len, k_len]
            tile_out = F.scaled_dot_product_attention(
                q_tile, k_tile, v_tile,
                attn_mask=attn_mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
            # Apply query mask to zero out padding outputs
            tile_out = tile_out * q_mask
            out[:, :, start:end, :] = tile_out
        return out

    def _global_linear_attention(self, Q, K, V, pad_mask_binary, q_gate, in_gate):
        B, h, N, d = Q.shape
        pad = pad_mask_binary  # [B,1,1,N] with 1 valid
        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)

        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None:
            K_feat = K_feat * in_gate
            V = V * in_gate

        # Apply padding mask to K and V
        K_feat = K_feat * pad.transpose(-2, -1)   # pad shape [B,1,1,N] -> [B,1,N,1] for key dim
        V = V * pad.transpose(-2, -1)

        KV = torch.matmul(K_feat.transpose(-2, -1), V)   # [B,h,m,d]
        Z = K_feat.sum(dim=-2, keepdim=True)             # [B,h,1,m]
        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-2, -1)) + 1e-5
        out = num / den

        # Mask query padding
        out = out * pad
        return out

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        B, N, _ = hidden_states.shape

        # Convert mask to binary valid mask [B,1,1,N]
        pad_mask_binary = self._convert_mask_to_binary(
            attention_mask, B, N, hidden_states.device, hidden_states.dtype
        )

        Q = self._reshape(self.q_proj(hidden_states))
        K = self._reshape(self.k_proj(hidden_states))
        V = self._reshape(self.v_proj(hidden_states))

        q_gate, in_gate = self.gates(hidden_states, Q)

        local_out = self._tiled_sliding_window_attention(Q, K, V, pad_mask_binary)
        global_out = self._global_linear_attention(Q, K, V, pad_mask_binary, q_gate, in_gate)

        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = (1.0 - fuse_gate) * local_out + fuse_gate * global_out
        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(fused)
        return out, None

    def get_ortho_loss(self):
        # Placeholder; can be implemented if needed
        return torch.tensor(0.0, device=self.out_proj.A.device)

    @staticmethod
    def init_from_pretrained(layer):
        q_src = getattr(layer, "q_lin", getattr(layer, "query", None))
        k_src = getattr(layer, "k_lin", getattr(layer, "key", None))
        v_src = getattr(layer, "v_lin", getattr(layer, "value", None))
        out_src = getattr(layer, "out_lin", getattr(layer, "dense", None))
        return {
            "q": q_src.weight.data.clone().detach() if q_src is not None else None,
            "k": k_src.weight.data.clone().detach() if k_src is not None else None,
            "v": v_src.weight.data.clone().detach() if v_src is not None else None,
            "o": out_src.weight.data.clone().detach() if out_src is not None else None,
        }


@ATTENTION_REGISTRY.register("aegis_v6_inject")
def inject_aegis_v6(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = AegisConfigV3(
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        window_size=getattr(comp_config, "window_size", 64),
        svd_rank=getattr(comp_config, "svd_rank", 128),
        low_rank_update=getattr(comp_config, "low_rank_update", 16),
        kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_query_gate=getattr(comp_config, "use_query_gate", True),
        use_input_gate=getattr(comp_config, "use_input_gate", True),
        use_pre_norm=getattr(comp_config, "use_pre_norm", True),
        long_context_threshold=getattr(comp_config, "long_context_threshold", 2048),
        max_global_tokens=getattr(comp_config, "max_global_tokens", 1024),
        kernel_method=getattr(comp_config, "kernel_method", "orthogonal"),
        kernel_clamp_min=getattr(comp_config, "kernel_clamp_min", -20.0),
        kernel_clamp_max=getattr(comp_config, "kernel_clamp_max", 20.0),
        kernel_use_float64=getattr(comp_config, "kernel_use_float64", False),
        kernel_seed=getattr(comp_config, "kernel_seed", None),
    )
    pretrained_dict = AegisAttentionV3.init_from_pretrained(old_attn) if old_attn is not None else None
    new_attn = AegisAttentionV3(cfg, pretrained_dict)
    layer.attention = new_attn
    return new_attn