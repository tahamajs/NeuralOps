r"""
AEGIS-Attention V6: Unified Hybrid Transformer Attention Module.

Combines SVD low-rank factorized projections with LoRA adapters,
sliding window attention, global linear kernel attention with adaptive spectral gating,
and context fusion initialized to prioritize high-fidelity local attention.
Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]  # safe for odd d: leaves trailing element out
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


@dataclass
class AegisConfigV3:
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    window_size: int = 64
    svd_rank: int = 384
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
    kernel_clamp_max: float = 8.0
    kernel_use_float64: bool = False
    kernel_seed: Optional[int] = None
    fusion_gate_bias: float = 1.8
    is_decoder: bool = False

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size")
        self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)
        self.head_dim = kwargs.get("head_dim")
        if self.head_dim is None and self.hidden_size is not None and self.num_heads is not None:
            self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.window_size = kwargs.get("window_size", 64)
        self.svd_rank = kwargs.get("svd_rank", min(384, self.hidden_size if self.hidden_size is not None else 384))
        self.low_rank_update = kwargs.get("low_rank_update", 16)
        self.kernel_feature_dim = kwargs.get("kernel_feature_dim", 128)
        self.dropout = kwargs.get("dropout", 0.1)
        self.use_query_gate = kwargs.get("use_query_gate", True)
        self.use_input_gate = kwargs.get("use_input_gate", True)
        self.use_pre_norm = kwargs.get("use_pre_norm", False)
        self.long_context_threshold = kwargs.get("long_context_threshold", 2048)
        self.max_global_tokens = kwargs.get("max_global_tokens", 1024)
        self.kernel_method = kwargs.get("kernel_method", "orthogonal")
        self.kernel_clamp_min = kwargs.get("kernel_clamp_min", -20.0)
        self.kernel_clamp_max = kwargs.get("kernel_clamp_max", 20.0)
        self.kernel_use_float64 = kwargs.get("kernel_use_float64", False)
        self.kernel_seed = kwargs.get("kernel_seed", None)
        self.fusion_gate_bias = kwargs.get("fusion_gate_bias", 1.8)
        self.is_decoder = kwargs.get("is_decoder", False)

    def __post_init__(self):
        if self.num_heads is not None:
            self.num_attention_heads = self.num_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        if self.head_dim is None and self.hidden_size is not None and self.num_heads is not None:
            self.head_dim = self.hidden_size // max(1, self.num_heads)


class FactorisedLowRankProjection(nn.Module):
    is_custom_attention: bool = True
    r"""
    Low-rank SVD projection with LoRA update branch:
    88291y = x (V \Sigma U^T) + x (B^T A^T) + b88291
    Base SVD weights are frozen; LoRA parameters A and B are trainable.
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
        self.svd_rank = min(svd_rank, in_features, out_features)
        self.lora_rank = lora_rank

        device = pretrained_weight.device if pretrained_weight is not None else torch.device("cpu")
        dtype = pretrained_weight.dtype if pretrained_weight is not None else torch.float32

        self.register_buffer("U", torch.zeros(out_features, self.svd_rank, device=device, dtype=dtype))
        self.S = nn.Parameter(torch.ones(self.svd_rank, device=device, dtype=dtype), requires_grad=False)
        self.register_buffer("V", torch.zeros(in_features, self.svd_rank, device=device, dtype=dtype))
        
        if lora_rank > 0:
            self.A = nn.Parameter(torch.zeros(out_features, lora_rank, device=device, dtype=dtype))
            self.B = nn.Parameter(torch.zeros(lora_rank, in_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            nn.init.zeros_(self.A)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype), requires_grad=False)

        if pretrained_weight is not None:
            self.decompose_from_weight(pretrained_weight, pretrained_bias)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
                rep = self.out_features // W.shape[0]
                W = W.repeat(rep, 1)
                if bias is not None:
                    bias = bias.repeat(rep)
            elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
                rep = W.shape[0] // self.out_features
                W = W.view(self.out_features, rep, -1).mean(dim=1)
                if bias is not None:
                    bias = bias.view(self.out_features, rep).mean(dim=1)
            elif W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
                padded[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])] = W[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            S = torch.clamp(S, min=1e-6)
            r = min(self.svd_rank, U.size(1), Vt.size(0))
            self.U.zero_()
            self.V.zero_()
            self.S.data.zero_()
            self.U[:, :r].copy_(U[:, :r].to(self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(self.S.dtype))
            self.V[:, :r].copy_(Vt[:r, :].t().contiguous().to(self.V.dtype))
            if bias is not None and self.bias is not None:
                self.bias.data[:min(len(self.bias), len(bias))].copy_(bias[:min(len(self.bias), len(bias))].to(self.bias.dtype))
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.S.requires_grad = False
        self.bias.requires_grad = False
        if self.lora_rank > 0 and self.A is not None and self.B is not None:
            self.A.requires_grad = True
            self.B.requires_grad = True

    @property
    def lora_A(self) -> Optional[nn.Parameter]:
        return self.A

    @property
    def lora_B(self) -> Optional[nn.Parameter]:
        return self.B

    @property
    def base_weight(self) -> torch.Tensor:
        return ((self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t().contiguous()).to(self.U.dtype)

    @property
    def weight(self) -> torch.Tensor:
        svd_w = (self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t().contiguous()
        if self.lora_rank > 0 and self.A is not None and self.B is not None:
            lora_w = self.A.float() @ self.B.float()
            return (svd_w + lora_w).to(self.A.dtype)
        return svd_w.to(self.U.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        xV = torch.matmul(x_f, self.V.float())
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)
        out_svd = torch.matmul(xV_S, self.U.float().t().contiguous())
        if self.lora_rank > 0 and self.A is not None and self.B is not None:
            xB = torch.matmul(x_f, self.B.float().t().contiguous())
            out_lora = torch.matmul(xB, self.A.float().t().contiguous())
            out = (out_svd + out_lora + self.bias.float()).to(orig_dtype)
        else:
            out = (out_svd + self.bias.float()).to(orig_dtype)
        return out


@dataclass
class KernelFeatureConfig:
    head_dim: int
    num_features: int
    method: str = "orthogonal"
    clamp_min: float = -20.0
    clamp_max: float = 8.0
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
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            phi = torch.matmul(x_f, self.proj_buf.float())
            norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True) / 2.0
            exp_in = phi - norm_sq
            exp_in = torch.clamp(exp_in, min=self.cfg.clamp_min, max=min(self.cfg.clamp_max, 8.0))
            out = self.scale * torch.exp(exp_in)
        return out.to(x.dtype)


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
        self.hidden_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False) if hidden_size != num_heads * head_dim else None
        self.proj = nn.Linear(3 * head_dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, hidden_states: torch.Tensor, local_out: torch.Tensor, global_out: torch.Tensor) -> torch.Tensor:
        B, h, N, d = local_out.shape
        if self.hidden_proj is not None:
            h_expanded = self.hidden_proj(hidden_states)
        else:
            h_expanded = hidden_states
        hidden_per_head = h_expanded.view(B, N, h, d).permute(0, 2, 1, 3)
        concat = torch.cat([hidden_per_head, local_out, global_out], dim=-1)
        logits = self.proj(concat)
        return torch.sigmoid(logits)


@ATTENTION_REGISTRY.register("aegis_v6", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_v6_attn", supports_causal=True, supports_gqa=True)
class AegisAttentionV3(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        pretrained_qkv: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisConfigV3):
            h_size = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None))))
            if h_size is None and pretrained_attn is not None:
                for obj in [pretrained_attn, getattr(pretrained_attn, "self", None), getattr(pretrained_attn, "output", None)]:
                    if obj is not None and hasattr(obj, "dense") and hasattr(obj.dense, "out_features"):
                        h_size = obj.dense.out_features
                        break
            if h_size is None:
                h_size = kwargs.get("hidden_size", 768)

            n_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", getattr(config, "n_head", None))))
            if n_heads is None:
                n_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))

            h_dim = getattr(config, "head_dim", getattr(config, "d_kv", None))
            if h_dim is None and h_size is not None and n_heads is not None:
                h_dim = h_size // max(1, n_heads)
            elif h_dim is None:
                h_dim = kwargs.get("head_dim", 64)

            n_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
            if n_kv_heads is None:
                n_kv_heads = kwargs.get("num_key_value_heads", n_heads)

            dec = is_decoder if is_decoder is not None else (
                getattr(config, "is_decoder", False)
                or getattr(config, "causal", False)
                or getattr(config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama") or any(str(getattr(cfg if "cfg" in locals() else (config if "config" in locals() else None), "model_type", "")).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                if config else False
            )

            config = AegisConfigV3(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                window_size=getattr(config, "window_size", kwargs.get("window_size", 64)),
                svd_rank=getattr(config, "svd_rank", kwargs.get("svd_rank", min(128, h_size) if h_size is not None else 128)),
                low_rank_update=getattr(config, "low_rank_update", kwargs.get("low_rank_update", 16)),
                kernel_feature_dim=getattr(config, "kernel_feature_dim", kwargs.get("kernel_feature_dim", 128)),
                dropout=getattr(config, "dropout", kwargs.get("dropout", 0.1)),
                is_decoder=dec,
            )
        super().__init__(config)
        self.cfg: AegisConfigV3 = config
        self.hidden_size: int = config.hidden_size
        self.h: int = config.num_heads
        self.num_heads: int = config.num_heads
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.h)
        self.kv_repeat: int = max(1, self.h // max(1, self.num_kv_heads))
        self.d: int = config.head_dim
        self.head_dim: int = self.d
        self.all_head_size: int = self.h * self.d
        self.all_kv_head_size: int = self.num_kv_heads * self.d
        self.supports_gqa: bool = True
        self.window: int = config.window_size
        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.is_nested_encoder: bool = False

        self.q_proj = FactorisedLowRankProjection(self.hidden_size, self.all_head_size, config.svd_rank, config.low_rank_update)
        self.k_proj = FactorisedLowRankProjection(self.hidden_size, self.all_kv_head_size, config.svd_rank, config.low_rank_update)
        self.v_proj = FactorisedLowRankProjection(self.hidden_size, self.all_kv_head_size, config.svd_rank, config.low_rank_update)
        self.out_proj = FactorisedLowRankProjection(self.all_head_size, self.hidden_size, config.svd_rank, config.low_rank_update)

        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.LayerNorm(self.d)
            self.k_norm = nn.LayerNorm(self.d)

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
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> FactorisedLowRankProjection:
        return self.q_proj

    @property
    def key(self) -> FactorisedLowRankProjection:
        return self.k_proj

    @property
    def value(self) -> FactorisedLowRankProjection:
        return self.v_proj

    @property
    def q_lin(self) -> FactorisedLowRankProjection:
        return self.q_proj

    @property
    def k_lin(self) -> FactorisedLowRankProjection:
        return self.k_proj

    @property
    def v_lin(self) -> FactorisedLowRankProjection:
        return self.v_proj

    @property
    def out_lin(self) -> FactorisedLowRankProjection:
        return self.out_proj

    @property
    def dense(self) -> FactorisedLowRankProjection:
        return self.out_proj

    @property
    def o_proj(self) -> FactorisedLowRankProjection:
        return self.out_proj

    @property
    def c_proj(self) -> FactorisedLowRankProjection:
        return self.out_proj

    def copy_weights(self, old_attn: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug(f"UniversalAdapter failed for {proj}: {e}")  # continue to fallback paths

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(old_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            c_attn_mod = old_attn.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                if is_conv1d or (weight.dim() == 2 and weight.shape[0] < weight.shape[1]) or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data if bias is not None else None
                split_size = weight.shape[0] // 3
                q_w = weight[:split_size, :]
                k_w = weight[split_size : 2 * split_size, :]
                v_w = weight[2 * split_size :, :]
                q_b = bias_data[:split_size] if bias_data is not None else None
                k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
                v_b = bias_data[2 * split_size :] if bias_data is not None else None
                extracted["q"] = (q_w, q_b)
                extracted["k"] = (k_w, k_b)
                extracted["v"] = (v_w, v_b)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = old_attn.c_proj
            base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_p, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (c_proj_mod.__class__.__name__ == "Conv1D") or (base_p.__class__.__name__ == "Conv1D") or hasattr(c_proj_mod, "nf") or hasattr(base_p, "nf")
                if is_conv1d or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1]) or (w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(old_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(old_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        # Check output projection attrs FIRST — DistilBERT has out_lin and is NOT nested
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention")
            or "RobertaSelfAttention" in str(type(old_attn))
            or "BertSelfAttention" in str(type(old_attn))
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisAttentionV6.copy_weights: could not extract mandatory projections {missing} "
                f"from {old_attn.__class__.__name__}. Refusing to install randomly-initialised weights."
            )

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"AegisAttentionV6: extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f}); continuing")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"AegisAttentionV6: extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f}); continuing")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target.decompose_from_weight(w_oriented, b)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(self.all_head_size, self.all_head_size, device=self.out_proj.U.device, dtype=self.out_proj.U.dtype)
                    target.decompose_from_weight(eye_w, None)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True

    def _reshape(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, num_heads, self.d).permute(0, 2, 1, 3)

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
        N_k = K.shape[2]
        W = self.window
        scale = d**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale

        idx_q = torch.arange(N, device=Q.device).unsqueeze(1)
        idx_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
        dist = (idx_q - idx_k).abs()
        band = (dist <= W).unsqueeze(0).unsqueeze(0)

        if self.is_decoder and N > 1:
            causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
            band = band & causal

        mask_val = -10000.0
        scores = scores.masked_fill(~band, mask_val)
        if pad_mask_binary is not None:
            pad_mask = (1.0 - pad_mask_binary) * mask_val
            scores = scores + pad_mask

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        attn = F.dropout(attn, p=self.cfg.dropout, training=self.training)
        return torch.matmul(attn, V)

    def _global_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, pad_mask_binary: torch.Tensor, q_gate: Optional[torch.Tensor], in_gate: Optional[torch.Tensor]) -> torch.Tensor:
        if self.is_decoder:
            return self._sliding_window_attention(Q, K, V, pad_mask_binary)

        B, h, N, d = Q.shape
        pad = pad_mask_binary
        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)

        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            # Only apply if sequence lengths match (fails in cross-attention where N_q != N_kv)
            K_feat = K_feat * in_gate
            V = V * in_gate

        K_feat = K_feat * pad.transpose(-2, -1).contiguous()
        V = V * pad.transpose(-2, -1).contiguous()

        KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V)
        Z = K_feat.sum(dim=-2, keepdim=True)
        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
        out = num / den
        # Note: do NOT apply pad here — pad is over the key dimension (N_kv) but out is over
        # the query dimension (N_q). They differ in cross-attention. K/V are already masked above.
        return torch.nan_to_num(out, nan=0.0)

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        pad_mask_binary = self._convert_mask_to_binary(eff_mask, B, N_kv, device, dtype)

        Q = self._reshape(self.q_proj(hidden_states), self.h)
        K = self._reshape(self.k_proj(kv_input), self.num_kv_heads)
        V = self._reshape(self.v_proj(kv_input), self.num_kv_heads)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        q_gate, in_gate = self.gates(hidden_states, Q)

        local_out = self._sliding_window_attention(Q, K, V, pad_mask_binary)
        global_out = self._global_linear_attention(Q, K, V, pad_mask_binary, q_gate, in_gate)

        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = fuse_gate * local_out + (1.0 - fuse_gate) * global_out

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            fused = fused * head_mask

        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        fused = fused.clamp(min=-1e4, max=1e4)

        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = out.clamp(min=-1e4, max=1e4)

        return out, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.bias.device)


@ATTENTION_REGISTRY.register("aegis_v6_inject")
@ATTENTION_REGISTRY.register("inject_aegis_v6")
def inject_aegis_v6(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_v6(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Injects AEGIS-V6 attention into a transformer layer."""
    try:
        if model_config is None:
            if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
                model_config = layer
            elif hasattr(layer, "attention"):
                old_attn = layer.attention
        h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768)))) if model_config is not None else 768
        n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)))) if model_config is not None else 12
        h_dim = getattr(model_config, "head_dim", getattr(model_config, "d_kv", h_size // max(1, n_heads))) if model_config is not None else h_size // max(1, n_heads)
        n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", n_heads)) if model_config is not None else n_heads

        src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))
        is_dec = bool(getattr(model_config, "is_decoder", False)) if model_config is not None else False
        if not is_dec and src_attn is not None:
            is_dec = bool(getattr(src_attn, "is_decoder", False))
        if not is_dec and layer is not None:
            is_dec = bool(getattr(layer, "is_decoder", False))

        cfg = AegisConfigV3(
            hidden_size=h_size,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            window_size=getattr(comp_config, "window_size", 64),
            svd_rank=getattr(comp_config, "svd_rank", 384),
            low_rank_update=getattr(comp_config, "low_rank_update", 16),
            kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128),
            dropout=getattr(comp_config, "dropout", 0.1),
            fusion_gate_bias=getattr(comp_config, "fusion_gate_bias", 1.8),
            is_decoder=is_dec,
        )
        new_attn = AegisAttentionV3(config=cfg)
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"Failed to inject AEGIS-V6: {e}")
        return None


# Stable public name used by the audit/test harness.
AegisAttentionV6 = AegisAttentionV3
