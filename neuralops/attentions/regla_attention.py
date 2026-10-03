"""Refined Gated Linear Attention (ReGLA) – Production-Hardened (v2)."""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

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
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank) if lora_rank and lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(
                torch.empty((out_features, self.lora_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        w = self.weight.to(target_dtype) if self.weight.dtype != target_dtype else self.weight
        b = (
            self.bias.to(target_dtype)
            if (self.bias is not None and self.bias.dtype != target_dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(target_dtype) if self.lora_A.dtype != target_dtype else self.lora_A
            bb = self.lora_B.to(target_dtype) if self.lora_B.dtype != target_dtype else self.lora_B
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out.to(target_dtype)


# ===========================================================================
# ReGLAAttention
# ===========================================================================
@ATTENTION_REGISTRY.register("regla_class")
@ATTENTION_REGISTRY.register("regla_attention_class")
class ReGLAAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        kernel_size: int = 3,
        pretrained_attn: nn.Module | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        is_decoder: bool | None = None,
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        comp_config = getattr(config, "compression_config", None) if config else None
        src = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")

        # -------- hidden_size --------
        dim = hidden_size
        if dim is None:
            dim = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.dim = int(dim)

        # -------- num_heads --------
        n_heads = num_heads
        if n_heads is None:
            n_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        if n_heads is None and src is not None:
            n_heads = getattr(
                src,
                "num_heads",
                getattr(src, "num_attention_heads", getattr(src, "n_head", None)),
            )
        if n_heads is None:
            n_heads = max(1, self.dim // 64)
        self.num_heads = max(1, int(n_heads))

        # -------- head_dim --------
        cfg_hd = _cfg_get(config, "head_dim", None)
        if cfg_hd is not None:
            self.head_dim = int(cfg_hd)
        elif src is not None:
            self.head_dim = None
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(src, attr):
                    qw = getattr(src, attr)
                    w = getattr(qw, "weight", None)
                    if w is not None:
                        self.head_dim = int(w.shape[0] // self.num_heads)
                        break
            if self.head_dim is None:
                self.head_dim = self.dim // self.num_heads
        else:
            self.head_dim = self.dim // self.num_heads

        if self.dim % self.num_heads != 0:
            self.head_dim = self.dim // self.num_heads

        # -------- num_kv_heads --------
        self.num_kv_heads = int(
            _cfg_get(
                config,
                "num_key_value_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
        )
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.kernel_size = int(kernel_size)
        self.chunk_size = max(1, int(chunk_size))

        self.conv_scale = nn.Parameter(torch.zeros(1))

        # -------- is_decoder --------
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or bool(_cfg_get(config, "causal", False))
                or any(
                    mt.startswith(p)
                    for p in (
                        "gpt",
                        "qwen",
                        "llama",
                        "mistral",
                        "gemma",
                        "phi",
                        "opt",
                        "bloom",
                        "falcon",
                    )
                )
                or (src is not None and hasattr(src, "c_attn"))
                or bool(kwargs.get("is_causal", False))
            )
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # -------- LoRA config --------
        self.lora_rank = int(
            _cfg_get(
                config,
                "lora_rank",
                _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 8)),
            )
        )
        self.lora_alpha = float(
            _cfg_get(
                config,
                "lora_alpha",
                _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        )
        self.freeze_base = self.lora_rank > 0

        # -------- device/dtype --------
        target_dtype = kwargs.get("dtype")
        target_device = kwargs.get("device")
        if target_dtype is None and src is not None:
            p = next(src.parameters(), None)
            if p is not None:
                target_dtype = p.dtype
                if target_device is None:
                    target_device = p.device
        if target_dtype is None:
            target_dtype = torch.float32

        # -------- projections --------
        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.gate_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            device=target_device,
            dtype=target_dtype,
        )

        # Aliases
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj

        # Conv1d
        self.conv = nn.Conv1d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            groups=self.dim,
            bias=True,
            device=target_device,
            dtype=target_dtype,
        )

        with torch.no_grad():
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
            if self.gate_proj.bias is not None:
                self.gate_proj.bias.zero_()

        if src is not None:
            self.copy_weights(src)

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        def _unwrap_linear(layer: Any) -> Any:
            return getattr(layer, "base_layer", layer)

        # GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = _unwrap_linear(src.c_attn)
            c_w = c_attn_mod.weight.data.clone() if hasattr(c_attn_mod, "weight") else None
            if c_w is not None:
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(c_attn_mod, "bias", None)
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = _unwrap_linear(src.c_proj)
                o_w = c_proj_mod.weight.data.clone() if hasattr(c_proj_mod, "weight") else None
                if o_w is not None:
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(c_proj_mod, "bias", None)
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # RoBERTa nested (outer container)
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (
                            w.data.clone(),
                            b.data.clone() if b is not None else None,
                        )
            except Exception as e:
                logger.debug("UniversalAdapter failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"ReGLAAttention.copy_weights: missing {missing_after} from {cls_name}."
            )

        with torch.no_grad():
            for proj_key, target_mod in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            target_mod.bias.zero_()
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                if (
                    w.shape != target_mod.weight.shape
                    and target_mod.weight.shape == w.t().contiguous().shape
                ):
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        device=target_mod.weight.device, dtype=target_mod.weight.dtype
                    )
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    target_mod.bias.data.zero_()

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "ReGLAAttention: weight for '%s' has near-zero norm (%.2e)",
                        proj_key,
                        w_norm,
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("ReGLAAttention does not support cross-attention")

        x = hidden_states
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype

        q = (
            self.q_proj(x)
            .view(B, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            self.k_proj(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            self.v_proj(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        g = (
            self.gate_proj(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # RoPE with past_len offset
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len : past_len + L, :]
            sin = sin[..., past_len : past_len + L, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)
            g = repeat_kv(g, self.kv_repeat)

        # Non-negative kernel activation & gating
        q_act = F.elu(q.float()) + 1.0
        k_act = F.elu(k.float()) + 1.0
        g_sig = torch.sigmoid(g.float())

        # Valid mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = (
                    (attention_mask > 0)
                    if not attention_mask.dtype.is_floating_point
                    else (attention_mask > -1e4)
                )
            elif attention_mask.dim() == 4:
                if attention_mask.dtype.is_floating_point:
                    valid_mask = (attention_mask > -1e4).any(dim=-2).squeeze(1)
                else:
                    valid_mask = attention_mask.bool().any(dim=-2).squeeze(1)
            else:
                valid_mask = torch.ones(B, L, device=device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(B, L, device=device, dtype=torch.bool)

        mask_4d = valid_mask[:, None, :, None].to(q_act.dtype)
        k_gated = k_act * g_sig * mask_4d
        v_gated = v.float() * g_sig * mask_4d

        is_causal = self.is_decoder or kwargs.get("is_causal", False)

        if is_causal:
            chunk = self.chunk_size
            S = torch.zeros(
                B,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=device,
                dtype=q_act.dtype,
            )
            Z = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=q_act.dtype)
            out_list = []

            for start in range(0, L, chunk):
                end = min(start + chunk, L)
                q_c = q_act[:, :, start:end]
                k_c = k_gated[:, :, start:end]
                v_c = v_gated[:, :, start:end]

                intra_kv = torch.cumsum(k_c.unsqueeze(-1) * v_c.unsqueeze(-2), dim=2)
                num_intra = (q_c.unsqueeze(-2) @ intra_kv).squeeze(-2)
                num_inter = (q_c.unsqueeze(-2) @ S.unsqueeze(2)).squeeze(-2)
                num = num_intra + num_inter

                z_intra = torch.cumsum(k_c, dim=2)
                den = (q_c * (z_intra + Z.unsqueeze(2))).sum(-1, keepdim=True).clamp(min=1e-4)
                out_list.append(num / den)

                S = S + (k_c.unsqueeze(-1) * v_c.unsqueeze(-2)).sum(dim=2)
                Z = Z + k_c.sum(dim=2)

            out = torch.cat(out_list, dim=2)
        else:
            kv = torch.matmul(k_gated.transpose(-2, -1).contiguous(), v_gated)
            z = k_gated.sum(dim=2, keepdim=True)
            num = torch.matmul(q_act, kv)
            den = (q_act * z).sum(-1, keepdim=True).clamp(min=1e-4)
            out = num / den

        out = out * mask_4d

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            out = out * hm.to(out.dtype)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size).to(dtype)

        # Depthwise 1D conv branch
        x_in = (x * valid_mask.unsqueeze(-1).to(dtype)).permute(0, 2, 1)
        conv_w = self.conv.weight.to(dtype)
        conv_b = self.conv.bias.to(dtype) if self.conv.bias is not None else None
        if is_causal:
            x_pad = F.pad(x_in, (self.kernel_size - 1, 0))
            loc = F.conv1d(
                x_pad,
                conv_w,
                conv_b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            ).permute(0, 2, 1)
        else:
            p = self.kernel_size // 2
            x_pad = F.pad(x_in, (p, p))
            loc = F.conv1d(
                x_pad,
                conv_w,
                conv_b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            ).permute(0, 2, 1)

        if loc.shape[-1] != self.dim:
            raise RuntimeError(
                f"ReGLAAttention: conv output dim {loc.shape[-1]} != hidden dim {self.dim}"
            )

        fused = (out.to(dtype) + self.conv_scale.to(dtype) * loc.to(dtype)).to(dtype)
        fused = fused * valid_mask.unsqueeze(-1).to(dtype)

        if self.is_nested_encoder:
            output = fused
        else:
            output = self.out_proj(fused)

        if output.shape[-1] != D:
            raise RuntimeError(f"ReGLAAttention: output dim {output.shape[-1]} != input {D}")

        return output.to(dtype), None


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("regla")
@ATTENTION_REGISTRY.register("regla_attention")
@ATTENTION_REGISTRY.register("regla_inject")
@ATTENTION_REGISTRY.register("inject_regla")
def inject_regla(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_regla: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
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
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_regla(
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

    # Single layer injection
    try:
        src_attn = old_attn
        if src_attn is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        # RE-18: pass the OUTER container (cand) so copy_weights
                        # can extract output projection for nested models.
                        src_attn = cand
                        break
        if src_attn is None:
            src_attn = target

        if isinstance(src_attn, ReGLAAttention) or "ReGLA" in src_attn.__class__.__name__:
            return None

        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        hidden_size = (
            kwargs.get("hidden_size")
            or getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or (
                src_attn.q_proj.in_features
                if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
                else None
            )
            or (
                src_attn.query.in_features
                if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
                else None
            )
            or (
                src_attn.q_lin.in_features
                if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
                else None
            )
            or (target.in_features if hasattr(target, "in_features") else None)
            or _cfg_get(cfg, "hidden_size", 768)
        )
        num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(src_attn, "num_heads", None)
            or getattr(src_attn, "num_attention_heads", None)
            or _cfg_get(cfg, "num_attention_heads", 12)
        )

        # RE-19: propagate kernel_size, chunk_size, rank from comp_config
        attn_kwargs = {}
        if comp_config is not None:
            for key in ("kernel_size", "chunk_size", "rank"):
                val = getattr(comp_config, key, None)
                if val is not None:
                    attn_kwargs[key] = val

        kwargs.pop("hidden_size", None)
        kwargs.pop("num_heads", None)
        kwargs.pop("d_model", None)
        new_attn = ReGLAAttention(
            config=cfg,
            pretrained_attn=src_attn if src_attn is not target else None,
            hidden_size=hidden_size,
            num_heads=num_heads,
            **attn_kwargs,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        # RE-20: ensure out_proj is Identity when attaching to nested encoder
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                if not isinstance(new_attn.out_proj, nn.Identity):
                    new_attn.out_proj = nn.Identity()
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            if not isinstance(new_attn.out_proj, nn.Identity):
                new_attn.out_proj = nn.Identity()
            target.self = new_attn

        return new_attn
    except Exception as e:
        logger.exception("inject_regla failed at layer %d: %s", layer_idx, e)
        return None


inject_regla_attention = inject_regla


__all__ = [
    "ReGLAAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_regla",
    "inject_regla_attention",
]
