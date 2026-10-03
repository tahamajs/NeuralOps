"""
AEGIS-Attention V6 (Final-Final Corrected).

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.attention_utils import repeat_kv
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin):
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
            f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
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
        q = (
            torch.cat([q_rot_embed, q_pass], dim=-1)
            if q_pass.numel() > 0
            else q_rot_embed
        )
        k = (
            torch.cat([k_rot_embed, k_pass], dim=-1)
            if k_pass.numel() > 0
            else k_rot_embed
        )
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
    return q, k


# ===========================================================================
# Config
# ===========================================================================


class AegisConfigV3:
    def __init__(self, **kwargs: Any):
        self.hidden_size = kwargs.get("hidden_size")
        self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)
        self.head_dim = kwargs.get("head_dim")
        if (
            self.head_dim is None
            and self.hidden_size is not None
            and self.num_heads is not None
        ):
            self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.window_size = kwargs.get("window_size", 64)
        default_svd = (
            min(384, self.hidden_size) if self.hidden_size is not None else 384
        )
        self.svd_rank = kwargs.get("svd_rank", default_svd)
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
        self.kernel_clamp_max = kwargs.get("kernel_clamp_max", 8.0)
        self.kernel_use_float64 = kwargs.get("kernel_use_float64", False)
        self.kernel_seed = kwargs.get("kernel_seed")
        self.fusion_gate_bias = kwargs.get("fusion_gate_bias", 1.8)
        self.is_decoder = kwargs.get("is_decoder", False)
        self.chunk_size = kwargs.get("chunk_size", 64)


# ===========================================================================
# Factorised projection
# ===========================================================================


class FactorisedLinear(nn.Module):
    """
    SVD-factorized linear layer with trainable LoRA adapters.

    Naming convention matches PEFT and framework detection:
      * lora_A: (lora_rank, in_features)   -- down projection, trainable
      * lora_B: (out_features, lora_rank)  -- up projection, trainable

    Base SVD tensors (U, S, V, bias) are stored as frozen Parameters so that
    they appear in `model.parameters()` and are counted by the framework's
    parameter counter (which uses `model.parameters()`, not buffers).

    Forward: y = (x @ V @ diag(S) @ U^T) + (x @ lora_A^T @ lora_B^T) + bias
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        lora_rank: int,
        pretrained_weight: torch.Tensor | None = None,
        pretrained_bias: torch.Tensor | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.svd_rank = min(svd_rank, in_features, out_features)
        self.lora_rank = lora_rank

        device = (
            pretrained_weight.device
            if pretrained_weight is not None
            else torch.device("cpu")
        )
        dtype = (
            pretrained_weight.dtype if pretrained_weight is not None else torch.float32
        )

        # FIX 3: store SVD tensors as frozen Parameters so they are counted
        # by model.parameters() (framework uses .parameters(), not buffers).
        self.U = nn.Parameter(
            torch.zeros(out_features, self.svd_rank, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.S = nn.Parameter(
            torch.ones(self.svd_rank, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.V = nn.Parameter(
            torch.zeros(in_features, self.svd_rank, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, device=device, dtype=dtype),
            requires_grad=False,
        )

        # FIX 2: rename A -> lora_B, B -> lora_A with PEFT shapes.
        # Old: A (out, r), B (r, in).  New: lora_A (r, in), lora_B (out, r).
        if lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.zeros(lora_rank, in_features, device=device, dtype=dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, lora_rank, device=device, dtype=dtype)
            )
            # Same init as before: kaiming on the (r, in) tensor, zeros on (out, r).
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if pretrained_weight is not None:
            self.decompose_from_weight(pretrained_weight, pretrained_bias)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()

            # Shape alignment (unchanged from original)
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

            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.zero_()
            self.U.data[:, :r].copy_(U[:, :r].to(self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(self.S.dtype))
            self.V.data[:, :r].copy_(Vt[:r, :].t().contiguous().to(self.V.dtype))

            if bias is not None:
                n = min(self.bias.numel(), bias.numel())
                self.bias.data[:n].copy_(bias[:n].to(self.bias.dtype))

    def freeze_base_weights(self) -> None:
        # Base SVD weights are already frozen via requires_grad=False on construction.
        # LoRA tensors are trainable by default.
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    @property
    def base_weight(self) -> torch.Tensor:
        return (
            (self.U.float() * self.S.float().unsqueeze(0))
            @ self.V.float().t().contiguous()
        ).to(self.U.dtype)

    @property
    def weight(self) -> torch.Tensor:
        svd_w = (
            self.U.float() * self.S.float().unsqueeze(0)
        ) @ self.V.float().t().contiguous()
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_w = (
                self.lora_B.float() @ self.lora_A.float()
            )  # (out,r) @ (r,in) = (out,in)
            return (svd_w + lora_w).to(self.lora_B.dtype)
        return svd_w.to(self.U.dtype)

    # Backward-compat aliases for code that still refers to A / B.
    @property
    def A(self) -> nn.Parameter | None:
        return self.lora_B

    @property
    def B(self) -> nn.Parameter | None:
        return self.lora_A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()

        # SVD branch: x @ V @ diag(S) @ U^T + bias
        xV = torch.matmul(x_f, self.V.float())
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)
        out = torch.matmul(xV_S, self.U.float().t().contiguous()) + self.bias.float()

        # LoRA branch: x @ lora_A^T @ lora_B^T
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            xa = torch.matmul(x_f, self.lora_A.float().t().contiguous())  # (B, N, r)
            out = out + torch.matmul(xa, self.lora_B.float().t().contiguous())

        return out.to(orig_dtype)


# ===========================================================================
# Kernel feature map
# ===========================================================================


class KernelFeatureConfig:
    def __init__(
        self,
        head_dim,
        num_features,
        method="orthogonal",
        clamp_min=-20.0,
        clamp_max=8.0,
        use_float64_exp=False,
        eps=1e-5,
        scaling=None,
        seed=None,
    ):
        self.head_dim = head_dim
        self.num_features = num_features
        self.method = method
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_float64_exp = use_float64_exp
        self.eps = eps
        self.scaling = scaling
        self.seed = seed


class StableKernelFeatureMap(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.head_dim = config.head_dim
        self.m = config.num_features
        self.scale = (
            config.scaling
            if config.scaling is not None
            else 1.0 / math.sqrt(self.m + config.eps)
        )
        self.register_buffer("proj", self._build_projection())

    def _build_projection(self):
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        d, m = self.head_dim, self.m
        if self.cfg.method != "orthogonal":
            return torch.randn(d, m) * math.sqrt(d)
        if m <= d:
            q, _ = torch.linalg.qr(torch.randn(d, m))
            return q * math.sqrt(d)
        num_blocks = (m + d - 1) // d
        blocks = [torch.linalg.qr(torch.randn(d, d))[0] for _ in range(num_blocks)]
        return torch.cat(blocks, dim=1)[:, :m] * math.sqrt(d)

    def forward(self, x):
        """FAVOR+ feature map with GLOBAL max subtraction for FP16 stability."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            phi = torch.matmul(x_f, self.proj.float())
            norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True) / 2.0
            raw = phi - norm_sq

            # FIX: Global max (scalar) subtraction to avoid FP16 overflow.
            # This scalar cancels in the softmax normalization (safe).
            max_val = raw.detach().max()

            exp_in = torch.clamp(
                raw - max_val,
                min=self.cfg.clamp_min,
                max=self.cfg.clamp_max,
            )
            return (self.scale * torch.exp(exp_in)).to(x.dtype)


# ===========================================================================
# Gates
# ===========================================================================


class GatingModule(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, use_query, use_input):
        super().__init__()
        self.query_gate = nn.Linear(head_dim, 1, bias=False) if use_query else None
        if use_input:
            self.input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.input_gate.bias, 1.8)
        else:
            self.input_gate = None

    def forward(self, hidden_states, Q):
        q_gate = (
            torch.sigmoid(self.query_gate(Q)) if self.query_gate is not None else None
        )
        in_gate = None
        if self.input_gate is not None:
            in_gate = (
                torch.sigmoid(self.input_gate(hidden_states))
                .permute(0, 2, 1)
                .unsqueeze(-1)
            )
        return q_gate, in_gate


class ContextFusionGate(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, init_bias=1.8):
        super().__init__()
        self.hidden_proj = (
            nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            if hidden_size != num_heads * head_dim
            else None
        )
        self.proj = nn.Linear(3 * head_dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, init_bias)

    def forward(self, hidden_states, local_out, global_out):
        B, h, N, d = local_out.shape
        h_exp = (
            self.hidden_proj(hidden_states)
            if self.hidden_proj is not None
            else hidden_states
        )
        h_per_head = h_exp.view(B, N, h, d).permute(0, 2, 1, 3)
        return torch.sigmoid(
            self.proj(torch.cat([h_per_head, local_out, global_out], dim=-1))
        )


# ===========================================================================
# Main attention
# ===========================================================================


@ATTENTION_REGISTRY.register("aegis_v6", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_v6_attn", supports_causal=True, supports_gqa=True)
class AegisAttentionV3(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config=None,
        pretrained_qkv=None,
        pretrained_attn=None,
        is_decoder=None,
        **kwargs,
    ):
        if not isinstance(config, AegisConfigV3):
            h_size = getattr(
                config,
                "hidden_size",
                getattr(
                    config,
                    "d_model",
                    getattr(config, "n_embd", getattr(config, "dim", None)),
                ),
            )
            if h_size is None and pretrained_attn is not None:
                for obj in [
                    pretrained_attn,
                    getattr(pretrained_attn, "self", None),
                    getattr(pretrained_attn, "output", None),
                ]:
                    if (
                        obj is not None
                        and hasattr(obj, "dense")
                        and hasattr(obj.dense, "out_features")
                    ):
                        h_size = obj.dense.out_features
                        break
            if h_size is None:
                h_size = kwargs.get("hidden_size", 768)
            n_heads = getattr(
                config,
                "num_attention_heads",
                getattr(
                    config,
                    "num_heads",
                    getattr(config, "n_heads", getattr(config, "n_head", None)),
                ),
            ) or kwargs.get("num_heads", 12)
            h_dim = getattr(
                config, "head_dim", getattr(config, "d_kv", None)
            ) or kwargs.get("head_dim", h_size // max(1, n_heads))
            n_kv_heads = getattr(
                config, "num_key_value_heads", getattr(config, "num_kv_heads", None)
            ) or kwargs.get("num_key_value_heads", n_heads)
            dec = self._detect_decoder(config, is_decoder, pretrained_attn)
            config = AegisConfigV3(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                window_size=getattr(config, "window_size", 64),
                svd_rank=getattr(
                    config, "svd_rank", min(128, h_size) if h_size else 128
                ),
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
        self.is_decoder = getattr(config, "is_decoder", False)
        self.is_nested_encoder = False
        self.chunk_size = config.chunk_size

        self.q_proj = FactorisedLinear(
            self.hidden_size,
            self.all_head_size,
            config.svd_rank,
            config.low_rank_update,
        )
        self.k_proj = FactorisedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            config.svd_rank,
            config.low_rank_update,
        )
        self.v_proj = FactorisedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            config.svd_rank,
            config.low_rank_update,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size,
            self.hidden_size,
            config.svd_rank,
            config.low_rank_update,
        )

        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.LayerNorm(self.d)
            self.k_norm = nn.LayerNorm(self.d)

        self.kernel_map = StableKernelFeatureMap(
            KernelFeatureConfig(
                head_dim=self.d,
                num_features=config.kernel_feature_dim,
                method=config.kernel_method,
                clamp_min=config.kernel_clamp_min,
                clamp_max=config.kernel_clamp_max,
                use_float64_exp=config.kernel_use_float64,
                seed=config.kernel_seed,
            )
        )

        self.gates = GatingModule(
            config.hidden_size,
            self.h,
            self.d,
            config.use_query_gate,
            config.use_input_gate,
        )
        self.fusion = ContextFusionGate(
            config.hidden_size, self.h, self.d, config.fusion_gate_bias
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @staticmethod
    def _detect_decoder(config, is_decoder, old_attn):
        if is_decoder is not None:
            return bool(is_decoder)
        if config is not None:
            if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
                return True
            mt = str(getattr(config, "model_type", "")).lower()
            if mt.startswith(
                (
                    "gpt",
                    "qwen",
                    "llama",
                    "mistral",
                    "gemma",
                    "phi",
                    "opt",
                    "bloom",
                    "falcon",
                    "olmo",
                    "mpt",
                    "stablelm",
                )
            ):
                return True
        if old_attn is not None:
            if getattr(old_attn, "is_decoder", False):
                return True
            if hasattr(old_attn, "c_attn"):
                return True
            if "gpt" in old_attn.__class__.__name__.lower():
                return True
        return False

    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _raw_valid(mask, B, N_kv, device):
        if mask is None:
            return torch.ones(B, 1, 1, N_kv, device=device, dtype=torch.bool)
        if mask.dim() == 2:
            if mask.dtype == torch.bool:
                v = mask
            elif mask.dtype.is_floating_point:
                v = mask > -1e4
            else:
                v = mask > 0
            return v[:, None, None, :].to(torch.bool)
        if mask.dim() == 4:
            if mask.dtype == torch.bool:
                v = mask
            elif mask.dtype.is_floating_point:
                v = mask > -1e4
            else:
                v = mask > 0
            return v.to(torch.bool)
        raise ValueError(f"Unsupported attention_mask dim: {mask.dim()}")

    def _prepare_valid_mask(
        self,
        attention_mask: torch.Tensor | None,
        B: int,
        N_q: int,
        N_kv: int,
        device: torch.device,
        is_cross: bool,
    ) -> torch.Tensor:
        """
        Build a boolean validity mask of shape (B, 1, N_q, N_kv).
        True = key is allowed for the corresponding query.

        Causal masking is applied ONLY when:
        - the module is a decoder, AND
        - this is self-attention (not cross-attention).

        Handles three decoding regimes correctly via absolute positions:
        * prefill:       N_q == N_kv   (full sequence, causal mask diag-aligned)
        * single decode: N_q == 1      (single query sees all past keys)
        * multi decode:  N_q > 1, N_q < N_kv (chunked decode with past cache)
        In all three cases the query at absolute position `past_len + i` can attend
        to keys at absolute positions `<= past_len + i`.
        """
        # 1) Raw validity from padding (before any causal logic)
        raw = self._raw_valid(attention_mask, B, N_kv, device)  # (B, 1, 1 or N_q, N_kv)

        # 2) Broadcast padding along query dim if needed
        if raw.shape[-2] == 1 and N_q != 1:
            raw = raw.expand(B, 1, N_q, N_kv)

        # 3) Causal mask (decoder self-attention only)
        if self.is_decoder and not is_cross:
            # Sanity check: shapes must line up
            if raw.shape[-2] != N_q or raw.shape[-1] != N_kv:
                raise ValueError(
                    f"_prepare_valid_mask: shape mismatch: raw={tuple(raw.shape)} "
                    f"expected (B, 1, {N_q}, {N_kv})"
                )

            # Absolute positions: query i has absolute index (past_len + i).
            # past_len = N_kv - N_q is exact for both prefill and decode-with-cache.
            past_len = N_kv - N_q
            idx_q = (past_len + torch.arange(N_q, device=device)).unsqueeze(
                1
            )  # (N_q, 1)
            idx_k = torch.arange(N_kv, device=device).unsqueeze(0)  # (1, N_kv)

            # Causal: query q can attend to key k iff abs_k <= abs_q
            causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)  # (1, 1, N_q, N_kv)
            raw = raw & causal

        return raw

    # ------------------------------------------------------------------
    # Sliding window — FIX-S1, FIX-S2
    # ------------------------------------------------------------------
    def _sliding_window_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        valid: torch.Tensor,
        is_cross: bool,
        return_attn: bool = False,
    ):
        """
        Sliding-window + causal attention.

        Args:
            Q:       (B, h, N, d)
            K:       (B, h, N_kv, d)
            V:       (B, h, N_kv, d)
            valid:   (B, 1, N, N_kv) boolean — padding + causal pre-computed
                    by `_prepare_valid_mask`. This function additionally restricts
                    attention to a local band of width `self.window`.
            is_cross: whether this call is cross-attention (skip causal band).
            return_attn: if True, also return the attention matrix.

        Returns:
            (out, attn) if return_attn else (out, None)
        """
        B, h, N, d = Q.shape
        N_k = K.shape[2]
        W = self.window
        scale = d**-0.5

        # 1) Raw scores (full N x N_kv matrix; for production replace with a
        #    FlashAttention / Triton kernel to actually get O(N*W) memory).
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale

        # 2) Absolute positions
        #    past_len is exact for prefill (0), single decode (N_kv-1),
        #    and multi decode (N_kv-N).
        past_len = (N_k - N) if (self.is_decoder and not is_cross and N_k >= N) else 0
        abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)  # (N, 1)
        abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)  # (1, N_k)

        # 3) Local band: |abs_q - abs_k| <= W
        dist = (abs_q - abs_k).abs()  # (N, N_k)
        band = (dist <= W).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N_k)

        # 4) Causal band: query at abs_q can attend to key at abs_k iff abs_k <= abs_q.
        #    Applied for ALL decoder self-attention cases (prefill, single decode,
        #    multi-token decode). It degenerates to "all keys allowed" when N == 1
        #    because abs_q = N_k-1 >= every abs_k.
        if self.is_decoder and not is_cross:
            causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N_k)
            band = band & causal

        # 5) Apply band + caller-provided valid mask (padding + causal).
        mask_val = -1e4
        scores = scores.masked_fill(~band, mask_val)
        scores = scores.masked_fill(~valid, mask_val)
        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)

        # 6) Safe softmax: rows with no valid key → zero output (not uniform).
        any_valid = valid.any(dim=-1, keepdim=True).to(scores.dtype)  # (B, 1, N, 1)
        attn = F.softmax(scores.float(), dim=-1).to(Q.dtype) * any_valid
        attn = F.dropout(attn, p=self.cfg.dropout, training=self.training)

        # 7) Weighted sum
        out = torch.matmul(attn, V)

        if return_attn:
            return out, attn
        return out, None

    # ------------------------------------------------------------------
    # Non-causal linear attention (encoder)
    # ------------------------------------------------------------------
    def _global_linear_attention(self, Q, K, V, valid_kv_only, q_gate, in_gate):
        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)
        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            K_feat = K_feat * in_gate
            V = V * in_gate
        valid_k = valid_kv_only.transpose(-1, -2).to(K_feat.dtype)
        K_feat = K_feat * valid_k
        V = V * valid_k
        KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V)
        Z = K_feat.sum(dim=-2, keepdim=True)
        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
        return torch.nan_to_num(num / den, nan=0.0)

        # ------------------------------------------------------------------
        # Causal linear attention  — FIX-C3, FIX-C4
        # ------------------------------------------------------------------

    def _causal_linear_attention(
        self, Q, K, V, valid, q_gate, in_gate, chunk_size: int | None = None
    ):
        """
        Chunked causal linear attention.

        Three cases:
        * N == N_k (training / prefill): full chunked cumsum with causal structure.
        * N == 1 (single-token decode with cache): exact, since the single query
            attends to all keys in the cache (causal mask trivially holds).
        * N > 1 and N != N_k (multi-token decode): FALLBACK to sliding-window
            attention over the full key history. This avoids the current
            non-causal approximation which is known to be inaccurate.
        """
        B, h, N, d = Q.shape
        N_k = K.shape[2]
        if chunk_size is None:
            chunk_size = self.chunk_size

        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)

        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            K_feat = K_feat * in_gate
            V = V * in_gate

        valid_k = valid.any(dim=-2, keepdim=True).transpose(-1, -2).to(K_feat.dtype)
        K_feat = K_feat * valid_k
        V = V * valid_k

        # ---- Case: decode with cache (N != N_k) ----
        if N_k != N:
            # FIX: For single-token decode (N==1), non-causal over full history is EXACT.
            # For multi-token decode, fall back to exact sliding-window attention
            # rather than the inaccurate non-causal approximation.
            if N == 1:
                KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V)
                Z = K_feat.sum(dim=-2, keepdim=True)
                num = torch.matmul(Q_feat, KV)
                den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(
                    min=1e-6
                )
                return torch.nan_to_num(num / den, nan=0.0)
            else:
                # Multi-token decode: use exact sliding-window attention
                logger.warning(
                    "_causal_linear_attention: multi-token decode (N=%d, N_k=%d). "
                    "Falling back to exact sliding-window attention.",
                    N,
                    N_k,
                )
                _fallback_out, _ = self._sliding_window_attention(
                    Q, K, V, valid, False, return_attn=False
                )
                return _fallback_out

        # ---- Case: N == N_k (training / full prefill) ----
        M = Q_feat.shape[-1]
        state_kv = torch.zeros(B, h, M, d, device=Q.device, dtype=Q_feat.dtype)
        state_k = torch.zeros(B, h, M, device=Q.device, dtype=Q_feat.dtype)
        outs = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Qc = Q_feat[:, :, start:end]
            Kc = K_feat[:, :, start:end]
            Vc = V[:, :, start:end]

            # FIX-C3: correct outer product orientation
            KV_local = (Kc.unsqueeze(-1) * Vc.unsqueeze(-2)).cumsum(dim=2)
            K_local = Kc.cumsum(dim=2)

            KV_total = KV_local + state_kv.unsqueeze(2)
            K_total = K_local + state_k.unsqueeze(2)

            num = torch.einsum("bhnm,bhnmd->bhnd", Qc, KV_total)
            den = torch.einsum("bhnm,bhnm->bhn", Qc, K_total).clamp(min=1e-6)
            outs.append(num / den.unsqueeze(-1))

            state_kv = KV_total[:, :, -1]
            state_k = K_total[:, :, -1]

        out = torch.cat(outs, dim=2)
        return torch.nan_to_num(out, nan=0.0)

    # ------------------------------------------------------------------
    # KV-cache
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(past_key_value, layer_idx=0):
        if past_key_value is None:
            return None, None
        if hasattr(past_key_value, "key_cache") and hasattr(
            past_key_value, "value_cache"
        ):
            keys = past_key_value.key_cache
            vals = past_key_value.value_cache
            if len(keys) == 0:
                return None, None
            idx = min(layer_idx, len(keys) - 1)
            return keys[idx], vals[idx]
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            return past_key_value[0], past_key_value[1]
        return None, None

    @staticmethod
    def _past_len(past_k):
        return 0 if past_k is None else past_k.shape[-2]

    # ------------------------------------------------------------------
    # Forward
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
        layer_idx=0,
        *args,
        **kwargs,
    ):
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states

        Q = self._reshape(self.q_proj(hidden_states), self.h)
        K_new = self._reshape(self.k_proj(kv_input), self.num_kv_heads)
        V_new = self._reshape(self.v_proj(kv_input), self.num_kv_heads)

        # RoPE with past offset
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_k_probe, _ = self._read_past_kv(past_key_value, layer_idx)
            past_len = self._past_len(past_k_probe)
            cos = cos[..., past_len : past_len + N, :]
            sin = sin[..., past_len : past_len + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        past_k, past_v = self._read_past_kv(past_key_value, layer_idx)
        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        if (
            past_key_value is not None
            and hasattr(past_key_value, "update")
            and not is_cross
        ):
            past_key_value.update(K_new, V_new, layer_idx)
        else:
            pass

        N_kv = K_cat.shape[2]
        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        valid = self._prepare_valid_mask(eff_mask, B, N, N_kv, device, is_cross)

        q_gate, in_gate = self.gates(hidden_states, Q)

        if self.is_decoder:
            local_out, local_attn = self._sliding_window_attention(
                Q, K_full, V_full, valid, is_cross, return_attn=bool(output_attentions)
            )
            global_out = self._causal_linear_attention(
                Q, K_full, V_full, valid, q_gate, in_gate
            )
            fuse_gate = self.fusion(hidden_states, local_out, global_out)
            fused = fuse_gate * local_out + (1.0 - fuse_gate) * global_out
        else:
            local_out, local_attn = self._sliding_window_attention(
                Q, K_full, V_full, valid, is_cross, return_attn=bool(output_attentions)
            )
            valid_kv_only = valid.any(dim=-2, keepdim=True)
            global_out = self._global_linear_attention(
                Q, K_full, V_full, valid_kv_only, q_gate, in_gate
            )
            fuse_gate = self.fusion(hidden_states, local_out, global_out)
            fused = fuse_gate * local_out + (1.0 - fuse_gate) * global_out

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            fused = fused * head_mask.to(fused.dtype)

        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        out = fused if self.is_nested_encoder else self.out_proj(fused)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_ret = local_attn if output_attentions else None
        # Always return a plain 2-tuple so `out, weights = attn(...)` unpacking works
        # for both encoder (legacy_bert) and decoder paths.
        return out, attn_ret

    def _reshape(self, x, num_heads):
        B, N, _ = x.shape
        return x.view(B, N, num_heads, self.d).permute(0, 2, 1, 3)

    def get_ortho_loss(self):
        return torch.zeros((), device=self.out_proj.bias.device)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn):
        from framework.core.adapters import UniversalAdapter

        cls_name = old_attn.__class__.__name__
        has_out_proj_attr = any(
            hasattr(old_attn, a)
            for a in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        if (
            hasattr(old_attn, "out_lin")
            or hasattr(old_attn, "c_proj")
            or hasattr(old_attn, "o_proj")
        ):
            self.is_nested_encoder = False
        elif (
            hasattr(old_attn, "self")
            and hasattr(old_attn.self, "query")
            and hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
        ) or (
            cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
            )
            or "RobertaSelfAttention" in str(type(old_attn))
            or "BertSelfAttention" in str(type(old_attn))
            or (cls_name.endswith("SelfAttention") and not has_out_proj_attr)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug(f"UniversalAdapter failed for {proj}: {e}")

        if hasattr(old_attn, "c_attn") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            c_attn_mod = old_attn.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                )
                if is_conv1d or (
                    weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size
                ):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bd = bias.data if bias is not None else None
                split = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split, :],
                    bd[:split] if bd is not None else None,
                )
                extracted["k"] = (
                    weight[split : 2 * split, :],
                    bd[split : 2 * split] if bd is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split :, :],
                    bd[2 * split :] if bd is not None else None,
                )

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = old_attn.c_proj
            base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_p, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    (c_proj_mod.__class__.__name__ == "Conv1D")
                    or (base_p.__class__.__name__ == "Conv1D")
                    or hasattr(c_proj_mod, "nf")
                    or hasattr(base_p, "nf")
                )
                if is_conv1d or (
                    w_o.dim() == 2
                    and w_o.shape[0] != self.hidden_size
                    and w_o.shape[1] == self.hidden_size
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        if hasattr(old_attn, "self") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            s = old_attn.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if self.is_nested_encoder:
            extracted["o"] = (None, None)
        elif (
            hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

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
                            extracted[proj] = (
                                layer.weight.data,
                                getattr(layer, "bias", None),
                            )
                            break

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisAttentionV3.copy_weights: could not extract mandatory projections "
                f"{missing} from {old_attn.__class__.__name__}. "
                f"Refusing to install randomly-initialised weights."
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    target.decompose_from_weight(_orient(w, in_dim), b)
                    # Warn if weight norm is suspiciously near-zero
                    norm = float(torch.linalg.norm(w.float()).item())
                    if norm < 1e-4:
                        logger.warning(
                            "AegisAttentionV3.copy_weights: extracted weight for '%s' has "
                            "near-zero Frobenius norm (%.2e) — weights may be uninitialised.",
                            proj_key,
                            norm,
                        )
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        self.all_head_size,
                        device=self.out_proj.U.device,
                        dtype=self.out_proj.U.dtype,
                    )
                    target.decompose_from_weight(eye_w, None)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif (
            hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower()
        ):
            self.is_decoder = True


# ===========================================================================
# Injection
# ===========================================================================


@ATTENTION_REGISTRY.register("aegis_v6_inject")
@ATTENTION_REGISTRY.register("inject_aegis_v6")
def inject_aegis_v6(
    layer=None,
    model=None,
    model_config=None,
    comp_config=None,
    layer_idx=0,
    total_layers=12,
    old_attn=None,
    _depth=0,
    **kwargs,
):
    if _depth > 8:
        logger.warning("inject_aegis_v6: recursion depth exceeded.")
        return None
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(
        target.layers, (torch.nn.ModuleList, list)
    ):
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
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        if model_config is None:
            if any(
                hasattr(layer, a) for a in ("hidden_size", "d_model", "n_embd", "dim")
            ):
                model_config = layer
            elif hasattr(layer, "attention"):
                old_attn = layer.attention

        h_size = (
            getattr(
                model_config,
                "hidden_size",
                getattr(
                    model_config,
                    "d_model",
                    getattr(model_config, "n_embd", getattr(model_config, "dim", 768)),
                ),
            )
            if model_config is not None
            else 768
        )
        n_heads = (
            getattr(
                model_config,
                "num_attention_heads",
                getattr(
                    model_config,
                    "num_heads",
                    getattr(
                        model_config, "n_heads", getattr(model_config, "n_head", 12)
                    ),
                ),
            )
            if model_config is not None
            else 12
        )
        h_dim = (
            getattr(
                model_config,
                "head_dim",
                getattr(model_config, "d_kv", h_size // max(1, n_heads)),
            )
            if model_config is not None
            else h_size // max(1, n_heads)
        )
        n_kv_heads = (
            getattr(
                model_config,
                "num_key_value_heads",
                getattr(model_config, "num_kv_heads", n_heads),
            )
            if model_config is not None
            else n_heads
        )

        src_attn = (
            old_attn
            if old_attn is not None
            else getattr(
                layer, "attention", getattr(layer, "self", getattr(layer, "attn", None))
            )
        )
        is_dec = AegisAttentionV3._detect_decoder(model_config, None, src_attn)

        cfg = AegisConfigV3(
            hidden_size=h_size,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            window_size=getattr(comp_config, "window_size", 64),
            svd_rank=getattr(comp_config, "svd_rank", 192),
            low_rank_update=getattr(comp_config, "low_rank_update", 16),
            kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128),
            dropout=getattr(comp_config, "dropout", 0.1),
            fusion_gate_bias=getattr(comp_config, "fusion_gate_bias", 1.8),
            chunk_size=getattr(comp_config, "chunk_size", 64),
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


AegisAttentionV6 = AegisAttentionV3
