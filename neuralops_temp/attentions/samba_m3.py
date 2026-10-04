"""
SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Block (Hardened).

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.attentions.simple_ssm import SimpleSSM
except ImportError:
    try:
        from simple_ssm import SimpleSSM
    except ImportError:
        SimpleSSM = None

logger = logging.getLogger(__name__)


# =========================================================================
# Helpers
# =========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    # Slice along sequence axis if needed
    if cos.shape[-2] != q.shape[-2]:
        if cos.shape[-2] >= q.shape[-2]:
            cos = cos.narrow(-2, 0, q.shape[-2])
            sin = sin.narrow(-2, 0, q.shape[-2])
        else:
            raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} < q {q.shape[-2]}")

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Align K length with cos
    cos_k = cos
    sin_k = sin
    if cos.shape[-2] != k.shape[-2]:
        if cos.shape[-2] >= k.shape[-2]:
            cos_k = cos.narrow(-2, cos.shape[-2] - k.shape[-2], k.shape[-2])
            sin_k = sin.narrow(-2, sin.shape[-2] - k.shape[-2], k.shape[-2])

    cos_q = cos[..., :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., :rotary_dim].to(q.device, q.dtype)
    cos_k = cos_k[..., :rotary_dim].to(k.device, k.dtype)
    sin_k = sin_k[..., :rotary_dim].to(k.device, k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    return q, k


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.freeze_base = freeze_base
        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SM3-FIX-01: cast weight/bias to input dtype (fixes Half/Float mismatch)
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        b = None
        if self.bias is not None:
            b = self.bias if self.bias.dtype == x.dtype else self.bias.to(x.dtype)
        out = F.linear(x, w, b)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A if self.lora_A.dtype == x.dtype else self.lora_A.to(x.dtype)
            bb = self.lora_B if self.lora_B.dtype == x.dtype else self.lora_B.to(x.dtype)
            lora_out = F.linear(F.linear(x, a), bb) * self.scaling
            out = out + lora_out
        return out

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True


class SpectralMamba(nn.Module):
    def __init__(self, d_model: int = 768, config: Any | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", d_model))
        elif hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model = int(d_model)
        if SimpleSSM is not None:
            try:
                self.mamba = SimpleSSM(config=config, d_model=self.d_model, **kwargs)
            except TypeError:
                try:
                    self.mamba = SimpleSSM(d_model=self.d_model, config=config)
                except TypeError:
                    self.mamba = SimpleSSM(self.d_model)
        else:
            self.mamba = nn.Identity()
        # SM3-FIX-12: bounded per-channel gate (0..1)
        self.dt_scale = nn.Parameter(torch.full((self.d_model,), 0.5))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.mamba(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        # SM3-FIX-02: cast gate to out.dtype
        gate = torch.sigmoid(self.dt_scale).to(out.dtype)
        gate = gate.view(1, 1, -1) if out.dim() == 3 else gate
        out = out * gate
        return out, None


@ATTENTION_REGISTRY.register("samba_m3", supports_causal=True, supports_gqa=True)
class SAMBAM3Block(nn.Module):
    r"""SAMBA-M3: Local Sliding-Window Attention + Spectral Mamba."""

    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        d_model: int | None = None,
        n_heads: int | None = None,
        window: int = 128,
        rank: int | None = None,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(config, int):
            d_model = config
            config = None

        # ---- d_model / n_heads ----
        if d_model is None:
            if config is not None:
                d_model = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", 768))
            elif pretrained_attn is not None:
                for attr in ("q_proj", "q_lin", "query"):
                    m = getattr(pretrained_attn, attr, None)
                    if m is not None and hasattr(m, "in_features"):
                        d_model = m.in_features
                        break
                if d_model is None:
                    d_model = getattr(pretrained_attn, "hidden_size", 768)
            else:
                d_model = kwargs.get("hidden_size", 768)

        if n_heads is None:
            n_heads = _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "n_heads", kwargs.get("num_attention_heads", 12)),
            )

        window = int(_cfg_get(config, "window_size", window))

        self.d_model: int = int(d_model)
        self.n_heads: int = int(n_heads)

        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        self.head_dim: int = self.d_model // self.n_heads
        self.window: int = window
        self.lora_rank: int = int(kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank)))
        self.lora_alpha: float = float(
            kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))
        )

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.n_heads
        self.num_key_value_groups: int = max(1, self.n_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.n_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # ---- SM3-FIX-08: decoder detection (override wins) ----
        if is_decoder is not None:
            is_dec = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            is_dec = bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "is_causal", False)
                or kwargs.get("is_causal", False)
            )
            decoder_prefixes = (
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
            if any(model_type.startswith(p) for p in decoder_prefixes):
                is_dec = True
            arch = getattr(config, "architectures", None) if config is not None else None
            if arch and any("GPT" in str(a) or "CausalLM" in str(a) for a in arch):
                is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = self.is_decoder
        self.is_nested_encoder: bool = False

        # ---- Projections ----
        self.q_lin = LoRALinear(
            self.d_model,
            self.q_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_lin = LoRALinear(
            self.d_model,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_lin = LoRALinear(
            self.d_model,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_lin = LoRALinear(
            self.q_head_size,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        # ---- Mamba branch ----
        self.mamba = SpectralMamba(
            self.d_model, config=config, is_decoder=self.is_decoder, **kwargs
        )

        # ---- SM3-FIX-04: alpha always fp32, cast in forward ----
        self.alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.gate_fn = nn.Sigmoid()

        dropout_p = float(
            _cfg_get(config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---- aliases ----
    @property
    def query(self):
        return self.q_lin

    @property
    def key(self):
        return self.k_lin

    @property
    def value(self):
        return self.v_lin

    @property
    def q_proj(self):
        return self.q_lin

    @property
    def k_proj(self):
        return self.k_lin

    @property
    def v_proj(self):
        return self.v_lin

    def get_ortho_loss(self) -> float:
        return 0.0

    def freeze_base_weights(self) -> None:
        for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            m.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            m.unfreeze_base_weights()

    # ------------------------------------------------------------------
    # Weight transfer (SM3-FIX-09)
    # ------------------------------------------------------------------
    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(pretrained_attn, "__class__", type(pretrained_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(pretrained_attn, "query")
                and not hasattr(pretrained_attn, "dense")
                and not hasattr(pretrained_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(pretrained_attn, "self") and hasattr(pretrained_attn, "output"):
                self.is_nested_encoder = True
                src = pretrained_attn.self
                if hasattr(pretrained_attn.output, "dense"):
                    w_o = pretrained_attn.output.dense.weight
                    b_o = pretrained_attn.output.dense.bias
            else:
                src = pretrained_attn

            # --- GPT-2 fused ---
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                p_cls = getattr(src.c_attn, "__class__", type(src.c_attn)).__name__
                if (
                    "Conv1D" in p_cls
                    or hasattr(src.c_attn, "nf")
                    or (c_attn_w.shape[0] == self.d_model and c_attn_w.shape[1] != self.d_model)
                ):
                    c_attn_w = c_attn_w.t().contiguous()
                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size
                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]
                out_w = src.c_proj.weight.detach()
                p_proj_cls = getattr(src.c_proj, "__class__", type(src.c_proj)).__name__
                if (
                    "Conv1D" in p_proj_cls
                    or hasattr(src.c_proj, "nf")
                    or (out_w.shape[0] == self.d_model and out_w.shape[1] != self.d_model)
                ):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # --- BERT/RoBERTa nested ---
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # --- Qwen / LLaMA ---
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = (
                    src.q_proj.bias.detach()
                    if getattr(src.q_proj, "bias", None) is not None
                    else None
                )
                w_k = src.k_proj.weight.detach()
                b_k = (
                    src.k_proj.bias.detach()
                    if getattr(src.k_proj, "bias", None) is not None
                    else None
                )
                w_v = src.v_proj.weight.detach()
                b_v = (
                    src.v_proj.bias.detach()
                    if getattr(src.v_proj, "bias", None) is not None
                    else None
                )
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = (
                        src.o_proj.bias.detach()
                        if getattr(src.o_proj, "bias", None) is not None
                        else None
                    )

            # --- Flat (DistilBERT) ---
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # --- Fallback ---
            if w_q is None:
                try:
                    adapter = UniversalAdapter()
                    for proj, target in (
                        ("q", self.q_lin),
                        ("k", self.k_lin),
                        ("v", self.v_lin),
                        ("o", self.out_lin),
                    ):
                        w, b = adapter.get_weight_bias(src, proj)
                        if w is None:
                            continue
                        if proj == "q":
                            w_q, b_q = w, b
                        elif proj == "k":
                            w_k, b_k = w, b
                        elif proj == "v":
                            w_v, b_v = w, b
                        else:
                            w_o, b_o = w, b
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # --- Copy into targets (SM3-FIX-09: cast + freeze) ---
            for target_mod, w_src, b_src, name in (
                (self.q_lin, w_q, b_q, "q_lin"),
                (self.k_lin, w_k, b_k, "k_lin"),
                (self.v_lin, w_v, b_v, "v_lin"),
                (self.out_lin, w_o, b_o, "out_lin"),
            ):
                if w_src is None:
                    continue
                tw = target_mod.weight
                w_cast = w_src.to(dtype=tw.dtype, device=tw.device)
                # orient if transposed
                if w_cast.shape != tw.shape and w_cast.t().contiguous().shape == tw.shape:
                    w_cast = w_cast.t().contiguous()
                min_r = min(tw.shape[0], w_cast.shape[0])
                min_c = min(tw.shape[1], w_cast.shape[1])
                tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])
                if b_src is not None and target_mod.bias is not None:
                    tb = target_mod.bias
                    b_cast = b_src.to(dtype=tb.dtype, device=tb.device)
                    mb = min(tb.numel(), b_cast.numel())
                    tb.data[:mb].copy_(b_cast[:mb])
                fnorm = float(torch.linalg.norm(tw.data.float()))
                if fnorm < 1e-4:
                    logger.warning("SAMBAM3Block: %s norm too small (%.2e)", name, fnorm)
                target_mod.freeze_base_weights()

    def _chunked_sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        past_len: int = 0,
    ) -> torch.Tensor:
        B, H, L, d = q.shape
        scale = 1.0 / math.sqrt(d)
        chunk = 64
        outputs = []

        for start in range(0, L, chunk):
            end = min(start + chunk, L)
            q_c = q[:, :, start:end]
            C_q = end - start

            # Absolute query positions (offset by past_len)
            q_abs_start = past_len + start
            q_abs_end = past_len + end

            # Key range: absolute positions in [max(0, q_abs_start - window), q_abs_end + window) for bidir
            if self.is_decoder:
                k_abs_start = max(0, q_abs_start - self.window)
                k_abs_end = q_abs_end
            else:
                k_abs_start = max(0, q_abs_start - self.window)
                k_abs_end = min(past_len + L, q_abs_end + self.window)

            # Slice keys in the *cache coordinates* (0..past_len+L)
            k_c = k[:, :, k_abs_start:k_abs_end]
            v_c = v[:, :, k_abs_start:k_abs_end]

            scores = torch.matmul(q_c.float(), k_c.transpose(-2, -1).contiguous().float()) * scale

            q_idx = torch.arange(q_abs_start, q_abs_end, device=q.device).view(1, 1, C_q, 1)
            k_idx = torch.arange(k_abs_start, k_abs_end, device=q.device).view(1, 1, 1, -1)
            dist = (q_idx - k_idx).abs()
            valid = dist <= self.window
            if self.is_decoder:
                valid = valid & (k_idx <= q_idx)

            scores = scores.masked_fill(~valid, -1e4)

            # SM3-FIX-07: head_mask dim==3 handled uniformly
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    # (B, N_total)
                    pad = (attention_mask[:, k_abs_start:k_abs_end] == 0).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(pad, -1e4)
                elif attention_mask.dim() == 3:
                    scores = scores + attention_mask[:, start:end, k_abs_start:k_abs_end].unsqueeze(
                        1
                    ).to(scores.dtype)
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask[:, :, start:end, k_abs_start:k_abs_end].to(
                        scores.dtype
                    )

            probs = F.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs.to(v_c.dtype))

            if head_mask is not None:
                hm = head_mask
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    # (B, H, Lk) — align to current key slice and unsqueeze
                    hm = hm[:, :, k_abs_start:k_abs_end].unsqueeze(2)
                hm = hm.to(probs.dtype)
                probs = probs * hm

            out_c = torch.matmul(probs, v_c)
            outputs.append(out_c)

        return torch.cat(outputs, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("SAMBAM3Block does not support cross-attention")

        B, L, D = hidden_states.shape
        target_dtype = hidden_states.dtype

        # ---- SM3-FIX-05: read cache ----
        past_k = past_v = None
        past_len = 0
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                past_len = past_k.shape[-2]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                past_k = past_key_value.key_cache[0]
                past_v = past_key_value.value_cache[0]
                past_len = past_k.shape[-2]

        # ---- QKV ----
        q = (
            self.q_lin(hidden_states)
            .view(B, L, self.n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_lin(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_lin(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- SM3-FIX-06: RoPE with offset ----
        pos_emb = position_embeddings or kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            if cos.shape[-2] >= past_len + L:
                cos = cos.narrow(-2, past_len, L)
                sin = sin.narrow(-2, past_len, L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- Append to KV cache ----
        if past_k is not None:
            k = torch.cat([past_k.to(k.dtype), k], dim=-2)
            v = torch.cat([past_v.to(v.dtype), v], dim=-2)

        present_key_value = (k, v)

        # ---- GQA expand ----
        if self.num_kv_heads != self.n_heads:
            k_exp = repeat_kv(k, self.num_key_value_groups)
            v_exp = repeat_kv(v, self.num_key_value_groups)
        else:
            k_exp, v_exp = k, v

        # ---- Attention ----
        attn_out = self._chunked_sliding_window_attention(
            q,
            k_exp,
            v_exp,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_len=past_len,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        attn_out = attn_out.to(target_dtype)

        # ---- Mamba branch ----
        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask, **kwargs)
        mamba_out = mamba_out.to(target_dtype)

        # ---- Fusion (SM3-FIX-04: alpha cast) ----
        alpha = torch.sigmoid(self.alpha).to(target_dtype)

        if self.is_nested_encoder:
            out = alpha * attn_out + (1.0 - alpha) * mamba_out
        else:
            attn_proj = self.out_lin(attn_out).to(target_dtype)
            out = alpha * attn_proj + (1.0 - alpha) * mamba_out

        out = torch.nan_to_num(out.to(target_dtype), nan=0.0)

        # ---- SM3-FIX-11: no silent pad; assert shape ----
        if out.shape[-1] != D:
            raise RuntimeError(f"SAMBAM3Block: output dim {out.shape[-1]} != expected {D}")

        return out, (present_key_value if output_attentions is False else (out, present_key_value))


@ATTENTION_REGISTRY.register("samba_m3_inject")
@ATTENTION_REGISTRY.register("inject_samba_m3")
def inject_samba_m3(
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
        logger.warning("inject_samba_m3: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_samba_m3(
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

    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break
    if src is None:
        return None

    # Idempotency
    if isinstance(src, SAMBAM3Block) or "SAMBAM3" in src.__class__.__name__:
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        child = getattr(target, attr, None)
        if isinstance(child, SAMBAM3Block):
            return None

    if cfg is None and hasattr(target, "config"):
        cfg = target.config
    if cfg is None and hasattr(src, "config"):
        cfg = src.config

    d_model = 768
    n_heads = 12
    if src is not None:
        d_model = getattr(
            src,
            "hidden_size",
            getattr(src, "embed_dim", getattr(src, "all_head_size", 768)),
        )
        for attr in ("q_proj", "q_lin", "query"):
            m = getattr(src, attr, None)
            if m is not None and hasattr(m, "in_features"):
                d_model = m.in_features
                break
        n_heads = getattr(
            src,
            "num_heads",
            getattr(src, "num_attention_heads", getattr(src, "n_heads", 12)),
        )
    if cfg is not None:
        d_model = _cfg_get(cfg, "hidden_size", _cfg_get(cfg, "dim", d_model))
        n_heads = _cfg_get(cfg, "num_attention_heads", _cfg_get(cfg, "n_heads", n_heads))

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 32.0))
    is_decoder_arg = kwargs.pop("is_decoder", None)

    kwargs.pop("d_model", None)
    kwargs.pop("n_heads", None)
    kwargs.pop("num_heads", None)

    try:
        new_block = SAMBAM3Block(
            config=cfg,
            d_model=d_model,
            n_heads=n_heads,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            pretrained_attn=src,
            is_decoder=is_decoder_arg,
            **kwargs,
        )
    except Exception as e:
        logger.error("SAMBAM3Block construction failed: %s", e)
        return None

    ref = next(src.parameters(), None)
    if ref is not None:
        new_block = new_block.to(device=ref.device, dtype=ref.dtype)

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_block.is_nested_encoder = True
            target.attention.self = new_block
        else:
            new_block.is_nested_encoder = False
            target.attention = new_block
        spliced = True
    elif hasattr(target, "self_attn"):
        new_block.is_nested_encoder = False
        target.self_attn = new_block
        spliced = True
    elif hasattr(target, "attn"):
        new_block.is_nested_encoder = False
        target.attn = new_block
        spliced = True
    elif hasattr(target, "self"):
        new_block.is_nested_encoder = True
        target.self = new_block
        spliced = True

    if not spliced:
        logger.error(
            "inject_samba_m3: no writable attention attribute on %s",
            type(target).__name__,
        )
        return None

    return new_block


__all__ = [
    "SAMBAM3Block",
    "SpectralMamba",
    "LoRALinear",
    "inject_samba_m3",
    "apply_rotary_pos_emb",
]
