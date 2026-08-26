r"""
AEGIS-Attention V6: Unified Hybrid Transformer Attention Module.

Combines SVD low-rank factorized projections with LoRA adapters,
sliding window attention, global linear kernel attention with adaptive spectral gating,
and context fusion initialized to prioritize high-fidelity local attention.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisConfigV3:
    hidden_size: int = 768
    num_heads = 4
    num_attention_heads: int = 12
    head_dim: int = 64
    window_size: int = 64
    svd_rank: int = 128
    low_rank_update: int = 16
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_query_gate: bool = True
    use_input_gate: bool = True
    use_pre_norm: bool = False
    long_context_threshold: int = 2048
    max_global_tokens: int = 1024
    kernel_method: str = "orthogonal"
    kernel_clamp_min: float = -20.0
    kernel_clamp_max: float = 20.0
    kernel_use_float64: bool = False
    kernel_seed: Optional[int] = None
    fusion_gate_bias: float = 1.8

    def __post_init__(self):
        self.num_attention_heads = self.num_heads
        self.head_dim = self.hidden_size // self.num_heads


class FactorisedLowRankProjection(nn.Module):
    r"""
    Low-rank SVD projection with LoRA update branch:
    $$y = x (V \Sigma U^T) + x (B^T A^T) + b$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        lora_rank: int,
        pretrained_weight: Optional[torch.Tensor] = None,
        pretrained_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if pretrained_weight is not None:
            W = pretrained_weight.detach().float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            S = torch.clamp(S, min=1e-6)
            effective_rank = min(svd_rank, U.size(1), Vt.size(0))
            U_init = U[:, :effective_rank].clone()
            S_init = S[:effective_rank].clone()
            V_init = Vt[:effective_rank, :].clone().t()
        else:
            effective_rank = min(svd_rank, in_features, out_features)
            U_init = torch.randn(out_features, effective_rank) * 0.02
            S_init = torch.ones(effective_rank)
            V_init = torch.randn(in_features, effective_rank) * 0.02

        self.register_buffer("U", U_init)
        self.S = nn.Parameter(S_init)
        self.register_buffer("V", V_init)
        self.A = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.B = nn.Parameter(torch.zeros(lora_rank, in_features))
        nn.init.normal_(self.B, std=0.02)
        if pretrained_bias is not None:
            self.bias = nn.Parameter(pretrained_bias.detach().clone().float())
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        svd_w = (self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t()
        lora_w = self.A.float() @ self.B.float()
        return (svd_w + lora_w).to(self.A.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        xV = torch.matmul(x_f, self.V.float())
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)
        out_svd = torch.matmul(xV_S, self.U.float().t())
        xB = torch.matmul(x_f, self.B.float().t())
        out_lora = torch.matmul(xB, self.A.float().t())
        out = (out_svd + out_lora + self.bias.float()).to(orig_dtype)
        return out


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
    def __init__(self, config: KernelFeatureConfig):
        super().__init__()
        self.cfg = config
        self.head_dim = config.head_dim
        self.m = config.num_features
        self.scale = config.scaling if config.scaling is not None else 1.0 / math.sqrt(self.m + config.eps)
        self.proj = self._build_projection()
        self.register_buffer("proj_buf", self.proj)

    def _build_projection(self) -> torch.Tensor:
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
        else:
            return torch.randn(d, m) * math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.matmul(x, self.proj_buf.to(x.dtype))
        norm_sq = (x * x).sum(dim=-1, keepdim=True) / 2.0
        exp_in = phi - norm_sq
        exp_in = torch.clamp(exp_in, min=self.cfg.clamp_min, max=self.cfg.clamp_max)
        out = self.scale * torch.exp(exp_in)
        return out


class GatingModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, use_query: bool, use_input: bool):
        super().__init__()
        self.use_query = use_query
        self.use_input = use_input
        if use_query:
            self.query_gate = nn.Linear(head_dim, 1, bias=False)
        else:
            self.query_gate = None
        if use_input:
            self.input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.input_gate.bias, 1.8)
        else:
            self.input_gate = None

    def forward(self, hidden_states: torch.Tensor, Q: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        q_gate = None
        in_gate = None
        if self.use_query and self.query_gate is not None:
            q_gate = torch.sigmoid(self.query_gate(Q))
        if self.use_input and self.input_gate is not None:
            logits = self.input_gate(hidden_states)
            in_gate = torch.sigmoid(logits).permute(0, 2, 1).unsqueeze(-1)
        return q_gate, in_gate


class ContextFusionGate(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, init_bias: float = 1.8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj = nn.Linear(hidden_size // num_heads + 2 * head_dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, hidden_states: torch.Tensor, local_out: torch.Tensor, global_out: torch.Tensor) -> torch.Tensor:
        B, h, N, d = local_out.shape
        hidden_per_head = hidden_states.view(B, N, h, d).permute(0, 2, 1, 3)
        concat = torch.cat([hidden_per_head, local_out, global_out], dim=-1)
        logits = self.proj(concat)
        return torch.sigmoid(logits)


class AegisAttentionV3(BaseAttention):
    def __init__(
        self,
        config: Optional[Any] = None,
        pretrained_qkv: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisConfigV3):
            h_size = getattr(config, "hidden_size", 768) if config is not None else 768
            n_heads = getattr(config, "num_attention_heads", 4) if config is not None else 12
            config = AegisConfigV3(
                hidden_size=h_size,
                num_heads=n_heads,
                window_size=getattr(config, "window_size", 64) if config is not None else 64,
                svd_rank=getattr(config, "svd_rank", 128) if config is not None else 128,
                low_rank_update=getattr(config, "low_rank_update", 16) if config is not None else 16,
                kernel_feature_dim=getattr(config, "kernel_feature_dim", 128) if config is not None else 128,
                dropout=getattr(config, "dropout", 0.1) if config is not None else 0.1,
            )
        super().__init__(config)
        self.cfg: AegisConfigV3 = config
        self.h: int = config.num_heads
        self.d: int = config.head_dim
        self.all_head_size: int = self.h * self.d
        self.window: int = config.window_size

        if pretrained_attn is not None and pretrained_qkv is None:
            pretrained_qkv = self.init_from_pretrained(pretrained_attn)

        Qp = pretrained_qkv.get("q") if pretrained_qkv else None
        Kp = pretrained_qkv.get("k") if pretrained_qkv else None
        Vp = pretrained_qkv.get("v") if pretrained_qkv else None
        Op = pretrained_qkv.get("o") if pretrained_qkv else None

        Qp_b = pretrained_qkv.get("q_b") if pretrained_qkv else None
        Kp_b = pretrained_qkv.get("k_b") if pretrained_qkv else None
        Vp_b = pretrained_qkv.get("v_b") if pretrained_qkv else None
        Op_b = pretrained_qkv.get("o_b") if pretrained_qkv else None

        self.q_proj = FactorisedLowRankProjection(config.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update, Qp, Qp_b)
        self.k_proj = FactorisedLowRankProjection(config.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update, Kp, Kp_b)
        self.v_proj = FactorisedLowRankProjection(config.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update, Vp, Vp_b)
        self.out_proj = FactorisedLowRankProjection(self.all_head_size, config.hidden_size, config.svd_rank, config.low_rank_update, Op, Op_b)

        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.LayerNorm(self.d, eps=1e-5)
            self.k_norm = nn.LayerNorm(self.d, eps=1e-5)

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
        self.fusion = ContextFusionGate(config.hidden_size, self.h, self.d, init_bias=config.fusion_gate_bias)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisAttentionV3":
        pretrained_dict = cls.init_from_pretrained(old_attn)
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 4))
        return cls(config=model_config, pretrained_qkv=pretrained_dict, **kwargs)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d).permute(0, 2, 1, 3)

    def _convert_mask_to_binary(self, attention_mask: Optional[torch.Tensor], B: int, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(B, 1, 1, N, device=device, dtype=dtype)
        if attention_mask.dim() == 2:
            valid = attention_mask > 0 if attention_mask.dtype != torch.bool else attention_mask
            return valid.unsqueeze(1).unsqueeze(2).to(dtype)
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask[:, 0, 0, :]
            else:
                valid = attention_mask[:, 0, 0, :] > -10000.0 if attention_mask.min() < -1000 else attention_mask[:, 0, 0, :] > 0
            return valid.unsqueeze(1).unsqueeze(2).to(dtype)
        return torch.ones(B, 1, 1, N, device=device, dtype=dtype)

    def _sliding_window_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, pad_mask_binary: torch.Tensor) -> torch.Tensor:
        B, h, N, d = Q.shape
        W = self.window
        scale = d**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        idx = torch.arange(N, device=Q.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        band = (dist <= W).unsqueeze(0).unsqueeze(0)
        mask_val = -10000.0 if scores.dtype in (torch.float16, torch.bfloat16) else -10000.0
        scores = scores.masked_fill(~band, mask_val)
        pad_mask = (1.0 - pad_mask_binary) * mask_val
        scores = scores + pad_mask

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = F.dropout(attn, p=self.cfg.dropout, training=self.training)
        return torch.matmul(attn, V)

    def _global_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, pad_mask_binary: torch.Tensor, q_gate: Optional[torch.Tensor], in_gate: Optional[torch.Tensor]) -> torch.Tensor:
        B, h, N, d = Q.shape
        pad = pad_mask_binary
        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)

        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None:
            K_feat = K_feat * in_gate
            V = V * in_gate

        K_feat = K_feat * pad.transpose(-2, -1)
        V = V * pad.transpose(-2, -1)

        KV = torch.matmul(K_feat.transpose(-2, -1), V)
        Z = K_feat.sum(dim=-2, keepdim=True)
        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-2, -1)).clamp(min=1e-5)
        out = num / den
        out = out * pad.transpose(-2, -1)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        pad_mask_binary = self._convert_mask_to_binary(
            attention_mask, B, N, hidden_states.device, hidden_states.dtype
        )

        Q = self._reshape(self.q_proj(hidden_states))
        K = self._reshape(self.k_proj(hidden_states))
        V = self._reshape(self.v_proj(hidden_states))

        q_gate, in_gate = self.gates(hidden_states, Q)

        local_out = self._sliding_window_attention(Q, K, V, pad_mask_binary)
        global_out = self._global_linear_attention(Q, K, V, pad_mask_binary, q_gate, in_gate)

        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = fuse_gate * local_out + (1.0 - fuse_gate) * global_out
        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(fused)
        return out, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.A.device)

    @staticmethod
    def init_from_pretrained(layer: nn.Module) -> Dict[str, torch.Tensor]:
        def _get_proj(p_name: str):
            for obj in [layer, getattr(layer, "self", None), getattr(layer, "output", None)]:
                if obj is not None:
                    for attr in [p_name, f"{p_name}_proj", f"{p_name}_lin", "query" if p_name == "q" else ("key" if p_name == "k" else ("value" if p_name == "v" else "dense")), "out_proj" if p_name == "o" else None]:
                        if attr and hasattr(obj, attr):
                            mod = getattr(obj, attr)
                            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                                return mod.weight.data.clone().detach(), (mod.bias.data.clone().detach() if mod.bias is not None else None)
            return None, None

        qw, qb = _get_proj("q")
        kw, kb = _get_proj("k")
        vw, vb = _get_proj("v")
        ow, ob = _get_proj("o")
        return {
            "q": qw, "k": kw, "v": vw, "o": ow,
            "q_b": qb, "k_b": kb, "v_b": vb, "o_b": ob,
        }


@ATTENTION_REGISTRY.register("aegis_v6")
@ATTENTION_REGISTRY.register("aegis_v6_inject")
def inject_aegis_v6(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> AegisAttentionV3:
    r"""Injects AEGIS-V6 attention into a transformer layer."""
    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768)))) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)))) if model_config is not None else 12

    cfg = AegisConfigV3(
        hidden_size=h_size,
        num_heads=n_heads,
        head_dim=h_size // max(1, n_heads),
        window_size=getattr(comp_config, "window_size", 64),
        svd_rank=getattr(comp_config, "svd_rank", 128),
        low_rank_update=getattr(comp_config, "low_rank_update", 16),
        kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        fusion_gate_bias=getattr(comp_config, "fusion_gate_bias", 1.8),
    )
    if model_config is None and old_attn is not None:
        new_attn = AegisAttentionV3.from_distilbert_pretrained(old_attn, config=cfg)
    else:
        new_attn = AegisAttentionV3(config=cfg, pretrained_attn=old_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn