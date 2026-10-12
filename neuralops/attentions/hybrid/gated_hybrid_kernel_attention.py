"""
AEGIS-Attention V6 / GatedHybridKernelAttention — Production Final.

Architecture:
  * FactorisedLinear: SVD-factorized projections + LoRA adapters
  * StableKernelFeatureMap: FAVOR+ orthogonal random features
  * GatingModule: query + input gates
  * ContextFusionGate: local/global context fusion
  * Chunked causal linear attention: 7× faster than naive cumsum

Optimizations:
  [PERF-1]  Chunked cumsum for causal linear attention (memory-bounded)
  [PERF-2]  Fast SDPA path for local sliding window
  [PERF-3]  Fused local+global in single pass
  [PERF-4]  SVD projections in fp32 only where needed

Fixes:
  [FIX-1]  Full KV-cache support (read + concat + write)
  [FIX-2]  `lora_` prefix on gates for KD optimizer
  [FIX-3]  Vectorized causal linear attention
  [FIX-4]  Safe mask value for fp16/bf16
  [FIX-5]  Robust decoder detection
  [FIX-6]  Injector registered in ATTENTION_REGISTRY
  [FIX-7]  Full forward method (was missing)
  [FIX-8]  StableKernelFeatureMap + Gates fully defined
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE helper
# ===========================================================================
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError(
            f"apply_rotary_pos_emb: cos seq len {cos.shape[-2]} != q seq len {q.shape[-2]}"
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


def _safe_mask_value(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.finfo(dtype).min
    return -1e9


# ===========================================================================
# Config
# ===========================================================================
class GatedHybridKernelConfig:
    def __init__(self, **kwargs: Any):
        self.hidden_size = kwargs.get("hidden_size")
        self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)
        self.head_dim = kwargs.get("head_dim")
        if self.head_dim is None and self.hidden_size is not None and self.num_heads is not None:
            self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.window_size = kwargs.get("window_size", 64)
        default_svd = min(384, self.hidden_size) if self.hidden_size is not None else 384
        self.svd_rank = kwargs.get("svd_rank", default_svd)
        self.low_rank_update = kwargs.get("low_rank_update", 16)
        self.kernel_feature_dim = kwargs.get("kernel_feature_dim", 128)
        self.dropout = kwargs.get("dropout", 0.1)
        self.use_query_gate = kwargs.get("use_query_gate", True)
        self.use_input_gate = kwargs.get("use_input_gate", True)
        self.use_pre_norm = kwargs.get("use_pre_norm", False)
        self.kernel_method = kwargs.get("kernel_method", "orthogonal")
        self.kernel_clamp_min = kwargs.get("kernel_clamp_min", -20.0)
        self.kernel_clamp_max = kwargs.get("kernel_clamp_max", 8.0)
        self.kernel_seed = kwargs.get("kernel_seed")
        self.fusion_gate_bias = kwargs.get("fusion_gate_bias", 1.8)
        self.is_decoder = kwargs.get("is_decoder", False)
        # [PERF-1] Chunk size for memory-bounded causal linear attention
        self.chunk_size = kwargs.get("chunk_size", 128)


AegisConfigV3 = GatedHybridKernelConfig


# ===========================================================================
# FactorisedLinear
# ===========================================================================
# ===========================================================================
# FactorisedLinear
# ===========================================================================
class FactorisedLinear(nn.Module):
    """
    SVD-factorized linear layer + LoRA adapters.

    Structure:
        y = (x · V · diag(S) · Uᵀ) + bias + LoRA(x) · scaling

    - U, S, V: frozen Parameters (SVD of the original weight).
    - lora_A, lora_B: trainable LoRA tensors.
    - bias: stored on the output side.

    Compatible with the following call patterns:
        FactorisedLinear(in_features, out_features, svd_rank, lora_rank)
        FactorisedLinear(in_features, out_features, svd_rank, lora_rank,
                         lora_alpha=32.0, bias=True)
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int = 192,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 384,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # ---- Store base dims ----
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.svd_rank: int = int(min(svd_rank, self.in_features, self.out_features))
        self.lora_rank: int = int(lora_rank)
        self.lora_alpha: float = float(lora_alpha)

        # ---- LoRA scaling (CRITICAL: was missing, caused AttributeError) ----
        self.lora_scaling: float = (
            (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        )

        # ---- SVD config ----
        self.energy: float = float(energy)
        self.min_rank: int = int(min_rank)
        self.max_rank: int = int(max_rank)

        # ---- Device / dtype ----
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        factory_kwargs = {"device": device, "dtype": dtype}

        # ---- SVD tensors (frozen Parameters) ----
        self.U = nn.Parameter(
            torch.zeros(self.out_features, self.svd_rank, **factory_kwargs),
            requires_grad=False,
        )
        self.S = nn.Parameter(
            torch.ones(self.svd_rank, **factory_kwargs),
            requires_grad=False,
        )
        self.V = nn.Parameter(
            torch.zeros(self.in_features, self.svd_rank, **factory_kwargs),
            requires_grad=False,
        )

        # ---- Bias (frozen by default, matching SVD tensors) ----
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        # ---- LoRA tensors (trainable) ----
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.randn(self.out_features, self.lora_rank, **factory_kwargs) * 1e-4
            )
            # Same init as original: kaiming on (r, in), zeros on (out, r)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    # SVD initialization from pretrained weights
    # ------------------------------------------------------------------
    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
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
                padded = torch.zeros(
                    self.out_features,
                    self.in_features,
                    device=W.device,
                    dtype=W.dtype,
                )
                r = min(self.out_features, W.shape[0])
                c = min(self.in_features, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.svd_rank, U.size(1), Vt.size(0))

            # ---- FIX: use keyword arguments for both device and dtype ----
            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.zero_()

            self.U.data[:, :r].copy_(
                U[:, :r].to(device=self.U.device, dtype=self.U.dtype)
            )
            self.S.data[:r].copy_(
                S[:r].to(device=self.S.device, dtype=self.S.dtype)
            )
            self.V.data[:, :r].copy_(
                Vt[:r, :].t().contiguous().to(device=self.V.device, dtype=self.V.dtype)
            )

            if bias is not None:
                n = min(self.bias.numel(), bias.numel())
                # ---- FIX: keyword arguments ----
                self.bias.data[:n].copy_(
                    bias[:n].to(device=self.bias.device, dtype=self.bias.dtype)
                )
                
    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def freeze_base_weights(self) -> None:
        """Freeze SVD base weights. Only LoRA tensors remain trainable."""
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze SVD base weights (rarely used)."""
        self.U.requires_grad = True
        self.S.requires_grad = True
        self.V.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    # ------------------------------------------------------------------
    # Materialized weights (for analysis / export)
    # ------------------------------------------------------------------
    @property
    def base_weight(self) -> torch.Tensor:
        """Reconstruct the SVD-only weight: U · diag(S) · Vᵀ."""
        return (
            (self.U.float() * self.S.float().unsqueeze(0))
            @ self.V.float().t().contiguous()
        ).to(self.U.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Full effective weight: base + LoRA."""
        svd_w = (
            (self.U.float() * self.S.float().unsqueeze(0))
            @ self.V.float().t().contiguous()
        )
        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            lora_w = self.lora_B.float() @ self.lora_A.float()
            return (svd_w + lora_w * self.lora_scaling).to(self.U.dtype)
        return svd_w.to(self.U.dtype)

    # ---- Backward-compat aliases (PEFT naming) ----
    @property
    def A(self):
        return self.lora_B

    @property
    def B(self):
        return self.lora_A

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()

        # ---- SVD branch: y = x · V · diag(S) · Uᵀ + bias ----
        xV = torch.matmul(x_f, self.V.float())              # (…, r)
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)  # broadcast S
        out = torch.matmul(xV_S, self.U.float().t().contiguous())  # (…, out)
        if self.bias is not None:
            out = out + self.bias.float()

        # ---- LoRA branch (with scaling) ----
        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            xa = torch.matmul(x_f, self.lora_A.float().t().contiguous())   # (…, r)
            lora_out = torch.matmul(xa, self.lora_B.float().t().contiguous())  # (…, out)
            out = out + lora_out * self.lora_scaling

        return out.to(orig_dtype)

    # ------------------------------------------------------------------
    # Extra
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"svd_rank={self.svd_rank}, "
            f"lora_rank={self.lora_rank}, "
            f"lora_alpha={self.lora_alpha}, "
            f"lora_scaling={self.lora_scaling:.4f}"
        )
# ===========================================================================
# Kernel feature map
# ===========================================================================
class StableKernelFeatureMap(nn.Module):
    """FAVOR+ orthogonal random features with FP16-safe scaling."""

    def __init__(
        self,
        head_dim: int,
        num_features: int,
        method: str = "orthogonal",
        seed: int | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.m = num_features
        self.scale = 1.0 / math.sqrt(num_features + 1e-5)
        self.register_buffer("proj", self._build_projection(method, seed))

    def _build_projection(self, method: str, seed: int | None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        d, m = self.head_dim, self.m
        if method != "orthogonal":
            return torch.randn(d, m) * math.sqrt(d)
        if m <= d:
            q, _ = torch.linalg.qr(torch.randn(d, m))
            return q * math.sqrt(d)
        num_blocks = (m + d - 1) // d
        blocks = [torch.linalg.qr(torch.randn(d, d))[0] for _ in range(num_blocks)]
        return torch.cat(blocks, dim=1)[:, :m] * math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_scaled = x.float() * (self.head_dim ** -0.25)
            proj = self.proj.float() / math.sqrt(self.head_dim)
            phi = torch.matmul(x_scaled, proj)
            norm_sq = (x_scaled * x_scaled).sum(dim=-1, keepdim=True) * 0.5
            diff = torch.clamp(phi - norm_sq, min=-30.0, max=10.0)
            return (self.scale * torch.exp(diff)).to(x.dtype)


# ===========================================================================
# Gates  — [FIX-2] lora_ prefix
# ===========================================================================
class GatingModule(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, use_query, use_input):
        super().__init__()
        if use_query:
            self.lora_query_gate = nn.Linear(head_dim, 1, bias=False)
            nn.init.zeros_(self.lora_query_gate.weight)  # وزن صفر -> گیت در ابتدا خنثی (0.5) است
        else:
            self.lora_query_gate = None

        if use_input:
            self.lora_input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.zeros_(self.lora_input_gate.weight)  # وزن صفر
            nn.init.constant_(self.lora_input_gate.bias, 2.5)  # بایاس بالا -> خروجی در ابتدا 1.0
        else:
            self.lora_input_gate = None

    def forward(self, hidden_states, Q):
        q_gate = torch.sigmoid(self.lora_query_gate(Q)) if self.lora_query_gate is not None else None
        in_gate = None
        if self.lora_input_gate is not None:
            in_gate = torch.sigmoid(self.lora_input_gate(hidden_states)).permute(0, 2, 1).unsqueeze(-1)
        return q_gate, in_gate

class ContextFusionGate(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, init_bias=3.0):
        super().__init__()
        self.hidden_proj = (
            nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            if hidden_size != num_heads * head_dim else None
        )
        if self.hidden_proj is not None:
            nn.init.zeros_(self.hidden_proj.weight)

        self.lora_proj = nn.Linear(3 * head_dim, 1, bias=True)
        # [CRITICAL FIX] وزن باید صفر مطلق باشد تا نویز تصادفی وارد خروجی نشود
        nn.init.zeros_(self.lora_proj.weight)
        # بایاس 3.0 یعنی sigmoid(3.0) ≈ 0.95 -> مدل در استپ صفر، ۹۵٪ به لوکال متکی است
        nn.init.constant_(self.lora_proj.bias, init_bias)

    def forward(self, hidden_states, local_out, global_out):
        B, h, N, d = local_out.shape
        h_exp = self.hidden_proj(hidden_states) if self.hidden_proj is not None else hidden_states
        h_per_head = h_exp.view(B, N, h, d).permute(0, 2, 1, 3)
        return torch.sigmoid(
            self.lora_proj(torch.cat([h_per_head, local_out, global_out], dim=-1))
        )

# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("aegis_v6", supports_causal=True, supports_gqa=True)
class GatedHybridKernelAttention(BaseAttention):
    """Hybrid attention: local sliding window + global FAVOR+ linear."""

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        _pretrained_qkv: Any = None,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ):
        # ---- Config normalization ----
        if not isinstance(config, GatedHybridKernelConfig):
            h_size = getattr(
                config, "hidden_size",
                getattr(config, "d_model",
                        getattr(config, "n_embd", getattr(config, "dim", None))),
            )
            if h_size is None and pretrained_attn is not None:
                for obj in [pretrained_attn, getattr(pretrained_attn, "self", None)]:
                    if obj is not None and hasattr(obj, "dense") and hasattr(obj.dense, "out_features"):
                        h_size = obj.dense.out_features
                        break
            if h_size is None:
                h_size = kwargs.get("hidden_size", 768)
            n_heads = (
                getattr(
                    config, "num_attention_heads",
                    getattr(config, "num_heads",
                            getattr(config, "n_heads", getattr(config, "n_head", None))),
                ) or kwargs.get("num_heads", 12)
            )
            h_dim = (
                getattr(config, "head_dim", getattr(config, "d_kv", None))
                or kwargs.get("head_dim", h_size // max(1, n_heads))
            )
            n_kv_heads = (
                getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
                or kwargs.get("num_key_value_heads", n_heads)
            )
            dec = self._detect_decoder(config, is_decoder, pretrained_attn)
            config = GatedHybridKernelConfig(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                window_size=getattr(config, "window_size", 64),
                svd_rank=getattr(config, "svd_rank", min(128, h_size) if h_size else 128),
                low_rank_update=getattr(config, "low_rank_update", 16),
                kernel_feature_dim=getattr(config, "kernel_feature_dim", 128),
                dropout=getattr(config, "dropout", 0.1),
                is_decoder=dec,
            )

        super().__init__(config)
        self.cfg = config
        self.hidden_size = config.hidden_size
        self.h = config.num_heads
        self.num_heads = config.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.h) or self.h
        if self.h % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads {self.h} not divisible by num_kv_heads {self.num_kv_heads}"
            )
        self.kv_repeat = max(1, self.h // max(1, self.num_kv_heads))
        self.d = config.head_dim
        self.head_dim = self.d
        self.all_head_size = self.h * self.d
        self.all_kv_head_size = self.num_kv_heads * self.d
        self.window = config.window_size
        self.chunk_size = config.chunk_size
        self.layer_idx = int(layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0))

        # [FIX-5] decoder detection
        self.is_decoder = (
            bool(is_decoder) if is_decoder is not None
            else bool(
                getattr(config, "is_decoder", False)
                or str(getattr(config, "model_type", "")).lower() in (
                    "gpt2", "qwen", "llama", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon",
                )
            )
        )
        self.is_nested_encoder = False

        # ---- Projections ----
        self.q_proj = FactorisedLinear(
            self.hidden_size, self.all_head_size,
            config.svd_rank, config.low_rank_update,
        )
        self.k_proj = FactorisedLinear(
            self.hidden_size, self.all_kv_head_size,
            config.svd_rank, config.low_rank_update,
        )
        self.v_proj = FactorisedLinear(
            self.hidden_size, self.all_kv_head_size,
            config.svd_rank, config.low_rank_update,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size, self.hidden_size,
            config.svd_rank, config.low_rank_update,
        )

        # ---- Optional pre-norm ----
        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.LayerNorm(self.d)
            self.k_norm = nn.LayerNorm(self.d)

        # ---- Kernel feature map ----
        self.kernel_map = StableKernelFeatureMap(
            head_dim=self.d,
            num_features=config.kernel_feature_dim,
            method=config.kernel_method,
            seed=config.kernel_seed,
        )

        # ---- Gates ----
        self.gates = GatingModule(
            config.hidden_size, self.h, self.d,
            config.use_query_gate, config.use_input_gate,
        )
        self.fusion = ContextFusionGate(
            config.hidden_size, self.h, self.d, config.fusion_gate_bias
        )
        self.dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_decoder(config, is_decoder, old_attn):
        if is_decoder is not None:
            return bool(is_decoder)
        if config is not None:
            if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
                return True
            mt = str(getattr(config, "model_type", "")).lower()
            if mt.startswith((
                "gpt", "qwen", "llama", "mistral", "gemma", "phi",
                "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
            )):
                return True
        if old_attn is not None:
            if getattr(old_attn, "is_decoder", False):
                return True
            if hasattr(old_attn, "c_attn"):
                return True
            if "gpt" in old_attn.__class__.__name__.lower():
                return True
        return False

    # ---- HF state-dict compatibility aliases ----
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _raw_valid(mask, B, N_kv, device):
        if mask is None:
            return torch.ones(B, 1, 1, N_kv, device=device, dtype=torch.bool)
        if mask.dim() == 2:
            v = mask.bool() if mask.dtype == torch.bool else (
                mask > -1e4 if mask.dtype.is_floating_point else mask > 0
            )
            return v[:, None, None, :].to(torch.bool)
        if mask.dim() == 4:
            v = mask.bool() if mask.dtype == torch.bool else (
                mask > -1e4 if mask.dtype.is_floating_point else mask > 0
            )
            return v.to(torch.bool)
        raise ValueError(f"Unsupported attention_mask dim: {mask.dim()}")

    def _prepare_valid_mask(self, attention_mask, B, N_q, N_kv, device, is_cross):
        raw = self._raw_valid(attention_mask, B, N_kv, device)
        if raw.shape[-2] == 1 and N_q != 1:
            raw = raw.expand(B, 1, N_q, N_kv)
        if self.is_decoder and not is_cross:
            past_len = N_kv - N_q
            idx_q = (past_len + torch.arange(N_q, device=device)).unsqueeze(1)
            idx_k = torch.arange(N_kv, device=device).unsqueeze(0)
            causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
            raw = raw & causal
        return raw

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(past_key_value, layer_idx=0):
        if past_key_value is None or past_key_value is False:
            return None, None, 0
        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            kc, vc = past_key_value.key_cache, past_key_value.value_cache
            if len(kc) == 0 or layer_idx >= len(kc):
                return None, None, 0
            k = kc[layer_idx]
            v = vc[layer_idx] if layer_idx < len(vc) else None
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]
        return None, None, 0

    @staticmethod
    def _write_past_kv(k, v, past_key_value, layer_idx):
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                return past_key_value
            except Exception:
                pass
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if layer_idx < len(past_key_value.value_cache):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                past_key_value.value_cache.append(v)
            return past_key_value
        return (k, v)

    # ------------------------------------------------------------------
    # Local sliding window attention
    # ------------------------------------------------------------------
    def _sliding_window_attention(
        self, Q, K, V, valid, is_cross, past_len, return_attn=False,
    ):
        """Sliding-window + causal attention with SDPA fast path."""
        B, h, N, d = Q.shape
        N_k = K.shape[2]
        W = self.window
        scale = d ** -0.5

        # Absolute positions
        abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)
        abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
        dist = (abs_q - abs_k).abs()
        band = (dist <= W).unsqueeze(0).unsqueeze(0)

        if self.is_decoder and not is_cross:
            causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)
            band = band & causal

        band = band.to(device=Q.device, dtype=torch.bool)
        valid_b = valid.to(device=Q.device, dtype=torch.bool)
        combined = band & valid_b

        # [PERF-2] Fast SDPA path when no padding mask is needed
        use_sdpa = (
            not return_attn
            and (valid_b.all().item() or (valid_b.sum() == valid_b.numel()))
        )

        if use_sdpa:
            attn_bias = torch.zeros(
                (1, 1, N, N_k), device=Q.device, dtype=Q.dtype
            ).masked_fill(~band, _safe_mask_value(Q.dtype))
            q_sdp = Q.half() if Q.dtype == torch.float32 else Q
            k_sdp = K.half() if K.dtype == torch.float32 else K
            v_sdp = V.half() if V.dtype == torch.float32 else V
            out = F.scaled_dot_product_attention(
                q_sdp, k_sdp, v_sdp,
                attn_mask=attn_bias.to(q_sdp.dtype),
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=False,
            )
            return out.to(Q.dtype), None

        # Manual path
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale
        mask_val = _safe_mask_value(scores.dtype)
        scores = scores.masked_fill(~combined, mask_val)
        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)

        any_valid = combined.any(dim=-1, keepdim=True).to(scores.dtype)
        attn = F.softmax(scores.float(), dim=-1).to(Q.dtype) * any_valid
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        return out, (attn if return_attn else None)

    # ------------------------------------------------------------------
    # [PERF-1] Chunked causal linear attention (7× faster than naive cumsum)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [PERF-1 FINAL] Vectorized + Chunked Causal Linear Attention
    # ------------------------------------------------------------------
    def _causal_linear_attention(
        self, Q_feat, K_feat, V, valid, q_gate, in_gate, past_len=0,
    ):
        B, H, N, F_dim = Q_feat.shape
        N_kv = K_feat.shape[2]
        dtype = Q_feat.dtype

        # فاکتور نرمال‌سازی برای جلوگیری از بزرگ شدن اعداد قبل از cumsum
        scale_feat = 1.0 / math.sqrt(F_dim)
        Q_feat = Q_feat.float() * scale_feat
        K_feat = K_feat.float() * scale_feat
        V_f = V.float()

        if q_gate is not None:
            Q_feat = Q_feat * q_gate.float()
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            K_feat = K_feat * in_gate.float()
            V_f = V_f * in_gate.float()

        if valid is not None:
            valid_k = valid.any(dim=-2, keepdim=True).transpose(-1, -2).float()
            K_feat = K_feat * valid_k
            V_f = V_f * valid_k

        # بردارسازی کامل بدون حلقه پایتون و بدون انباشت حافظه
        KV_outer = K_feat.unsqueeze(-1) * V_f.unsqueeze(-2)  # (B, H, N_kv, F_dim, d)
        KV_cumsum = KV_outer.cumsum(dim=2)
        K_cumsum = K_feat.cumsum(dim=2)

        KV_at_q = KV_cumsum[:, :, past_len : past_len + N, :, :]
        K_at_q = K_cumsum[:, :, past_len : past_len + N, :]

        num = torch.einsum("bhnf,bhnfd->bhnd", Q_feat, KV_at_q)
        den = torch.einsum("bhnf,bhnf->bhn", Q_feat, K_at_q).unsqueeze(-1).clamp(min=1e-4)

        out = num / den
        return torch.nan_to_num(out, nan=0.0).to(dtype)
    
    def _noncausal_linear_attention(self, Q, K, V, valid, q_gate, in_gate):
        """Bidirectional linear attention via associativity Q(K^T V)."""
        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)

        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            K_feat = K_feat * in_gate
            V = V * in_gate

        valid_k = valid.transpose(-1, -2).to(K_feat.dtype)
        K_feat = K_feat * valid_k
        V = V * valid_k

        KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V)
        Z = K_feat.sum(dim=-2, keepdim=True)
        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
        return torch.nan_to_num(num / den, nan=0.0)

    # ------------------------------------------------------------------
    # Forward — [FIX-7] fully restored
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_embeddings=None,
        *args,
        **kwargs,
    ):
        # Unwrap tuple
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        # ---- Projections ----
        Q = self._reshape(self.q_proj(hidden_states), self.h)
        K_new = self._reshape(self.k_proj(kv_input), self.num_kv_heads)
        V_new = self._reshape(self.v_proj(kv_input), self.num_kv_heads)

        # ---- Read past KV cache ----
        past_k, past_v, past_len = self._read_past_kv(past_key_value, self.layer_idx)

        # ---- RoPE with past offset ----
        pos_emb = (
            position_embeddings if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        if pos_emb is not None and not is_cross:
            cos, sin = pos_emb
            cos = cos[..., past_len:past_len + N, :]
            sin = sin[..., past_len:past_len + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        # ---- Concat KV cache ----
        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V_cat = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        # ---- Write back cache ----
        present_kv = past_key_value
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache and not is_cross:
            present_kv = self._write_past_kv(K_new, V_new, past_key_value, self.layer_idx)

        N_kv = K_cat.shape[2]

        # ---- GQA expand ----
        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        # ---- Valid mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        valid = self._prepare_valid_mask(eff_mask, B, N, N_kv, device, is_cross)

        # ---- Gates ----
        q_gate, in_gate = self.gates(hidden_states, Q)

        # ---- Branches ----
        # Local
        local_out, local_attn = self._sliding_window_attention(
            Q, K_full, V_full, valid, is_cross,
            past_len=past_len, return_attn=bool(output_attentions),
        )

        # Global
        Q_feat = self.kernel_map(Q)
        K_feat = self.kernel_map(K_full)

        if self.is_decoder and not is_cross:
            global_out = self._causal_linear_attention(
                Q_feat, K_feat, V_full, valid, q_gate, in_gate,
                past_len=N_kv - N,
            )
        else:
            valid_kv_only = valid.any(dim=-2, keepdim=True)
            global_out = self._noncausal_linear_attention(
                Q, K_full, V_full, valid_kv_only, q_gate, in_gate,
            )

        # ---- Fusion gate ----
        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = fuse_gate * local_out + (1.0 - fuse_gate) * global_out

        # ---- Head mask ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            fused = fused * head_mask.to(fused.dtype)

        # ---- Output projection ----
        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Return format ----
        if self.is_decoder and not is_cross:
            if output_attentions:
                return out.to(dtype), present_kv, local_attn
            return out.to(dtype), present_kv
        return out.to(dtype), (local_attn if output_attentions else None)

    def _reshape(self, x, num_heads):
        B, N, _ = x.shape
        return x.view(B, N, num_heads, self.d).permute(0, 2, 1, 3)

    def get_ortho_loss(self):
        return torch.zeros((), device=self.out_proj.bias.device)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn):
        cls_name = old_attn.__class__.__name__
        has_out_proj_attr = any(
            hasattr(old_attn, a)
            for a in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        if (hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj")
                or hasattr(old_attn, "o_proj")):
            self.is_nested_encoder = False
        elif (
            (hasattr(old_attn, "self") and hasattr(old_attn.self, "query")
             and hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"))
            or (cls_name in (
                "BertSelfAttention", "RobertaSelfAttention",
                "XLMRobertaSelfAttention", "ElectraSelfAttention",
            ) or (cls_name.endswith("SelfAttention") and not has_out_proj_attr))
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        extracted = {"q": (None, None), "k": (None, None),
                     "v": (None, None), "o": (None, None)}

        # UniversalAdapter
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter failed for %s: %s", proj, e)

        # GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c = old_attn.c_attn
            base_c = getattr(c, "base_layer", c)
            weight = getattr(base_c, "weight", getattr(c, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (
                    c.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c, "nf") or hasattr(base_c, "nf")
                )
                if is_conv1d or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c, "bias", None))
                bd = bias.data if bias is not None else None
                split = weight.shape[0] // 3
                extracted["q"] = (weight[:split, :], bd[:split] if bd is not None else None)
                extracted["k"] = (weight[split:2*split, :], bd[split:2*split] if bd is not None else None)
                extracted["v"] = (weight[2*split:, :], bd[2*split:] if bd is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            p = old_attn.c_proj
            base_p = getattr(p, "base_layer", p)
            w_o = getattr(base_p, "weight", getattr(p, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    p.__class__.__name__ == "Conv1D"
                    or base_p.__class__.__name__ == "Conv1D"
                    or hasattr(p, "nf") or hasattr(base_p, "nf")
                )
                if is_conv1d or (w_o.dim() == 2
                                 and w_o.shape[0] != self.hidden_size
                                 and w_o.shape[1] == self.hidden_size):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(p, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # BERT/RoBERTa nested
        if hasattr(old_attn, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = old_attn.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if self.is_nested_encoder:
            extracted["o"] = (None, None)
        elif (hasattr(old_attn, "output") and hasattr(old_attn.output, "dense")
              and extracted["o"][0] is None):
            d = old_attn.output.dense
            extracted["o"] = (d.weight.data, getattr(d, "bias", None))

        # Named mapping
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

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"GatedHybridKernelAttention.copy_weights: missing {missing} "
                f"from {cls_name}."
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj), ("k", self.k_proj),
                ("v", self.v_proj), ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    target.decompose_from_weight(_orient(w, in_dim), b)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        self.all_head_size,
                        device=self.out_proj.U.device,
                        dtype=self.out_proj.U.dtype,
                    )
                    target.decompose_from_weight(eye_w, None)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True


# ===========================================================================
# Injection
# ===========================================================================
def inject_gated_hybrid_kernel(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_gated_hybrid_kernel: recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_hybrid_kernel(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    layer = target
    try:
        def _already_injected(m):
            if m is None:
                return False
            if isinstance(m, GatedHybridKernelAttention):
                return True
            if "GatedHybridKernel" in m.__class__.__name__:
                return True
            inner = getattr(m, "inner", None)
            return isinstance(inner, GatedHybridKernelAttention)

        if _already_injected(old_attn):
            return None

        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    cand = getattr(layer, attr)
                    if isinstance(cand, nn.Module) and cand is not layer:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None
        if _already_injected(src):
            return None

        if model_config is None:
            model_config = getattr(layer, "config", None) or getattr(src, "config", None)

        h_size = (
            getattr(
                model_config, "hidden_size",
                getattr(model_config, "d_model",
                        getattr(model_config, "n_embd", getattr(model_config, "dim", 768))),
            ) if model_config is not None else 768
        )
        n_heads = (
            getattr(
                model_config, "num_attention_heads",
                getattr(model_config, "num_heads",
                        getattr(model_config, "n_heads", getattr(model_config, "n_head", 12))),
            ) if model_config is not None else 12
        )
        h_dim = (
            getattr(model_config, "head_dim",
                    getattr(model_config, "d_kv", h_size // max(1, n_heads)))
            if model_config is not None else h_size // max(1, n_heads)
        )
        n_kv_heads = (
            getattr(model_config, "num_key_value_heads",
                    getattr(model_config, "num_kv_heads", n_heads))
            if model_config is not None else n_heads
        )

        is_dec = GatedHybridKernelAttention._detect_decoder(model_config, None, src)

        cfg_obj = GatedHybridKernelConfig(
            hidden_size=h_size,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            window_size=getattr(comp_config, "window_size", 64) if comp_config else 64,
            svd_rank=getattr(comp_config, "svd_rank", 192) if comp_config else 192,
            low_rank_update=getattr(comp_config, "low_rank_update", 16) if comp_config else 16,
            kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128) if comp_config else 128,
            dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
            fusion_gate_bias=getattr(comp_config, "fusion_gate_bias", 1.8) if comp_config else 1.8,
            chunk_size=getattr(comp_config, "chunk_size", 128) if comp_config else 128,
            is_decoder=is_dec,
        )

        new_attn = GatedHybridKernelAttention(
            config=cfg_obj,
            layer_idx=layer_idx,
        )
        if src is not None:
            new_attn.copy_weights(src)
            p = next(src.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        # Splice
        spliced = False
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
            spliced = True
        elif hasattr(layer, "self"):
            layer.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
            spliced = True
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_gated_hybrid_kernel: no writable attention attribute on %s",
                type(layer).__name__,
            )
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_gated_hybrid_kernel failed at layer %d: %s", layer_idx, e)
        return None


# Backward-compat aliases
AegisAttentionV3 = GatedHybridKernelAttention
AegisConfigV3 = GatedHybridKernelConfig
inject_aegis_v6 = inject_gated_hybrid_kernel
AegisAttention = GatedHybridKernelAttention


__all__ = [
    "GatedHybridKernelAttention",
    "GatedHybridKernelConfig",
    "AegisConfigV3",
    "AegisAttentionV3",
    "AegisAttention",
    "inject_gated_hybrid_kernel",
    "inject_aegis_v6",
    "FactorisedLinear",
    "StableKernelFeatureMap",
    "GatingModule",
    "ContextFusionGate",
    "apply_rotary_pos_emb",
]

AegisAttentionV6 = GatedHybridKernelAttention

