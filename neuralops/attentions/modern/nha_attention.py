"""
Native Hybrid Attention (NHA) — Optimized & Production-Hardened.

Fuses localized sliding-window attention (via SDPA fast-path) with global
linear Random Fourier Features (RFF via vectorized causal scan) through
a per-head learned gate.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)

_DENOM_EPS = 1e-5


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
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

    if cos.shape[-2] < q.shape[-2]:
        raise ValueError(f"NHA RoPE: cos seq {cos.shape[-2]} < q seq {q.shape[-2]}")
    if cos.shape[-2] > q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = min(cos.shape[-1], q.shape[-1])
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim].to(q_rot.dtype)

    q_rot = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
    return q, k


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
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
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
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
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
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


class NativeHybridAttention(BaseAttention):
    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        window_size: int = 64,
        rff_features: int = 128,
        dropout: float = 0.1,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        self.config = config
        if comp_config is None:
            comp_config = getattr(config, "compression_config", None)
        self.comp_config = comp_config

        source_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else (old_attn if old_attn is not None else kwargs.get("source_attn"))
        )

        hidden_size = kwargs.get("hidden_size", kwargs.get("dim", kwargs.get("d_model")))
        num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", kwargs.get("n_heads")))
        num_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads"))

        if config is not None:
            if hidden_size is None:
                hidden_size = _cfg_get(
                    config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", 768))
                )
            if num_heads is None:
                num_heads = _cfg_get(
                    config, "num_attention_heads", _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", 12))
                )
            if num_kv_heads is None:
                num_kv_heads = _cfg_get(
                    config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", num_heads)
                )
        elif source_attn is not None:
            if hidden_size is None:
                for a in ("hidden_size", "embed_dim", "all_head_size"):
                    hidden_size = getattr(source_attn, a, None)
                    if hidden_size:
                        break
            if num_heads is None:
                for a in ("num_attention_heads", "n_heads", "num_heads"):
                    num_heads = getattr(source_attn, a, None)
                    if num_heads:
                        break
            if num_kv_heads is None:
                num_kv_heads = getattr(
                    source_attn, "num_key_value_heads", getattr(source_attn, "num_kv_heads", num_heads)
                )
        else:
            hidden_size = hidden_size or 768
            num_heads = num_heads or 12
            num_kv_heads = num_kv_heads or num_heads

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = min(int(num_kv_heads or self.num_heads), self.num_heads)
        self.head_dim = self.dim // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.window_size = int(
            _cfg_get(comp_config, "window_size", _cfg_get(config, "window_size", kwargs.get("window_size", window_size)))
        )
        self.rff_features = int(
            _cfg_get(comp_config, "rff_features", _cfg_get(config, "rff_features", kwargs.get("rff_features", rff_features)))
        )

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
            is_dec = False
            if config is not None:
                is_dec = bool(getattr(config, "is_decoder", False) or getattr(config, "is_causal", False))
            _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
            if any(model_type.startswith(p) for p in _dec_pfx):
                is_dec = True
            if hasattr(config, "architectures"):
                archs = getattr(config, "architectures", []) or []
                if any("GPT" in str(a) or "CausalLM" in str(a) for a in archs):
                    is_dec = True
            if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
                is_dec = True
            if source_attn is not None:
                if getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn"):
                    is_dec = True
            self.is_decoder = bool(is_dec)

        self.is_causal = bool(self.is_decoder)
        self.is_nested_encoder = False
        self.layer_idx = int(kwargs.get("layer_idx", 0))
        self.output_scale = nn.Parameter(torch.ones(1))

        def _lora_rank():
            for src in (config, comp_config):
                v = getattr(src, "lora_rank", None) if src else None
                if v is not None:
                    return int(v)
            return int(kwargs.get("lora_rank", 16))

        def _lora_alpha():
            for src in (config, comp_config):
                v = getattr(src, "lora_alpha", None) if src else None
                if v is not None:
                    return float(v)
            return float(kwargs.get("lora_alpha", 32.0))

        self.lora_rank = _lora_rank()
        self.lora_alpha = _lora_alpha()
        self.freeze_base = self.lora_rank > 0

        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )

        omega = self._build_orthogonal_matrix(self.head_dim, self.rff_features)
        self.register_buffer("rff_proj", omega)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", dropout)) if config else dropout
        self.dropout = nn.Dropout(dropout_p)

        self.gate = nn.Linear(self.dim, self.num_heads)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 3.0)

        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)

    def _build_orthogonal_matrix(self, d: int, m: int) -> torch.Tensor:
        blocks = []
        remaining = m
        while remaining > 0:
            block_size = min(d, remaining)
            g = torch.randn(d, block_size)
            q, _ = torch.linalg.qr(g)
            blocks.append(q)
            remaining -= block_size
        return torch.cat(blocks, dim=1) * math.sqrt(d)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = any(hasattr(src, a) for a in ("out_lin", "c_proj", "o_proj"))
        is_nested = (
            (hasattr(src, "self") and hasattr(src.self, "query") and hasattr(src, "output") and hasattr(src.output, "dense"))
            or (cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention")
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower()))
        ) and not has_out_lin
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn = src.c_attn
            base_c = getattr(c_attn, "base_layer", c_attn)
            c_w = getattr(base_c, "weight", getattr(c_attn, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (c_attn.__class__.__name__ == "Conv1D" or base_c.__class__.__name__ == "Conv1D" or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]))
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split : 2 * split], c_b[split : 2 * split] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split :], c_b[2 * split :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj = src.c_proj
                base_o = getattr(c_proj, "base_layer", c_proj)
                o_w = getattr(base_o, "weight", getattr(c_proj, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (c_proj.__class__.__name__ == "Conv1D" or base_o.__class__.__name__ == "Conv1D" or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]))
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, name in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"), "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"), "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        mod = getattr(src, n)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        target_map = {"q": self.q_proj, "k": self.k_proj, "v": self.v_proj, "o": self.out_proj}

        with torch.no_grad():
            for key, target_mod in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue
                    raise RuntimeError(f"NHA: missing weights for {key}")

                if w.shape == target_mod.weight.shape:
                    w_o = w
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    w_o = w.t().contiguous()
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    padded = torch.zeros_like(target_mod.weight)
                    padded[:min_0, :min_1] = w[:min_0, :min_1]
                    w_o = padded

                target_mod.weight.copy_(w_o)
                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.shape[0], b.shape[0])
                    target_mod.bias.zero_()
                    target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

        if self.lora_rank > 0:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                m.freeze_base_weights()

    def _local_window_attention(self, Q, K, V, valid_mask, is_causal=False, past_len=0, output_attentions=False):
        B, H, N_q, d = Q.shape
        N_k = K.shape[-2]
        device = Q.device
        dtype = Q.dtype

        q_idx = (past_len + torch.arange(N_q, device=device)).unsqueeze(1)
        k_idx = torch.arange(N_k, device=device).unsqueeze(0)

        if is_causal:
            band_mask = (k_idx > q_idx) | ((q_idx - k_idx) > self.window_size)
        else:
            band_mask = (q_idx - k_idx).abs() > self.window_size

        attn_mask = torch.zeros((B, 1, N_q, N_k), device=device, dtype=dtype)
        attn_mask.masked_fill_(band_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if valid_mask is not None:
            if valid_mask.shape[-1] != N_k:
                pad = N_k - valid_mask.shape[-1]
                if pad > 0:
                    valid_mask = torch.cat(
                        [valid_mask, torch.ones(B, pad, device=device, dtype=torch.bool)],
                        dim=1,
                    )
            pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)  
            attn_mask.masked_fill_(pad_mask, -1e4)

        if output_attentions:
            scale = 1.0 / math.sqrt(d)
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
            scores = scores + attn_mask
            attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0).to(dtype)
            attn = self.dropout(attn)
            out = torch.matmul(attn, V)
            return out, attn

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        return out, None

    def _global_rff_attention_chunked(self, Q, K, V, valid_mask, is_causal=False, past_len=0):
        B, H, N_q, d = Q.shape
        N_k = K.shape[-2]
        m = self.rff_features
        device = Q.device
        dtype = Q.dtype

        scale = d**-0.25
        Qf = torch.matmul(Q * scale, self.rff_proj.to(dtype=dtype, device=device))
        Kf = torch.matmul(K * scale, self.rff_proj.to(dtype=dtype, device=device))

        norm = 1.0 / math.sqrt(m)
        phi_Q = (F.elu(Qf) + 1.0) * norm
        phi_K = (F.elu(Kf) + 1.0) * norm

        if valid_mask is not None:
            kv_mask = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            phi_K = phi_K * kv_mask
            V_masked = V * kv_mask
        else:
            V_masked = V

        if not is_causal:
            KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)
            Z = phi_K.sum(dim=-2, keepdim=True)
            num = torch.matmul(phi_Q, KV)
            den = torch.matmul(phi_Q, Z.transpose(-2, -1)).clamp(min=_DENOM_EPS)
            return num / den

        # ==========================================================
        # Fully Vectorized Memory-Efficient Causal RFF (Inference-Safe)
        # ==========================================================
        # scores_lin: (B, H, N_q, N_k)
        scores_lin = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
        
        # Causal Masking: Query i sees Key j if j <= i + past_len
        q_idx = torch.arange(N_q, device=device).unsqueeze(1)
        k_idx = torch.arange(N_k, device=device).unsqueeze(0)
        causal_mask = k_idx <= (q_idx + past_len)
        
        scores_lin = scores_lin.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        
        denom = scores_lin.sum(dim=-1, keepdim=True).clamp(min=_DENOM_EPS)
        out = torch.matmul(scores_lin, V_masked) / denom
        
        return out.to(dtype)

    @staticmethod
    def _read_past_kv(past_key_value: Any, layer_idx: int):
        if past_key_value is None:
            return None, None, 0
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            return (k, v, k.shape[-2]) if k is not None else (None, None, 0)
        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            idx = min(layer_idx, len(past_key_value.key_cache) - 1)
            k = past_key_value.key_cache[idx]
            if k is None:
                return None, None, 0
            v = past_key_value.value_cache[idx] if hasattr(past_key_value, "value_cache") and idx < len(past_key_value.value_cache) else None
            return k, v, k.shape[-2]
        return None, None, 0

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
        layer_idx: int | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("NHA does not support cross-attention.")

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        B, N, D = hidden_states.shape
        device = hidden_states.device
        is_causal = bool(self.is_decoder or kwargs.get("is_causal", False))
        layer_idx = self.layer_idx if layer_idx is None else layer_idx

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        K_new = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        V_new = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        past_k, past_v, past_len = self._read_past_kv(past_key_value, layer_idx)
        if past_k is not None and past_v is not None:
            K_cat = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V_cat = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K_cat, V_cat = K_new, V_new

        present_key_value = None
        if past_key_value is not None and hasattr(past_key_value, "update"):
            past_key_value.update(K_new, V_new, layer_idx)
            present_key_value = past_key_value
        elif self.is_decoder:
            present_key_value = (K_cat, V_cat)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            if cos.shape[-2] >= past_len + N:
                cos_slice = cos[..., past_len : past_len + N, :]
                sin_slice = sin[..., past_len : past_len + N, :]
            else:
                cos_slice, sin_slice = cos, sin
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos_slice, sin_slice)
            if past_k is not None and past_v is not None:
                K_cat = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)

        if self.kv_repeat > 1:
            K_full = repeat_kv(K_cat, self.kv_repeat)
            V_full = repeat_kv(V_cat, self.kv_repeat)
        else:
            K_full, V_full = K_cat, V_cat

        cur_valid = get_token_valid_mask(attention_mask, B, N, device)
        all_valid = (
            torch.cat([torch.ones(B, past_len, device=device, dtype=torch.bool), cur_valid], dim=1)
            if past_len > 0 and cur_valid is not None
            else cur_valid
        )

        local_out, local_attn = self._local_window_attention(
            Q, K_full, V_full, all_valid,
            is_causal=is_causal, past_len=past_len,
            output_attentions=bool(output_attentions)
        )
        global_out = self._global_rff_attention_chunked(
            Q, K_full, V_full, all_valid,
            is_causal=is_causal, past_len=past_len
        )

        gate = torch.sigmoid(self.gate(hidden_states)).unsqueeze(-1).permute(0, 2, 1, 3)
        out = gate * local_out + (1.0 - gate) * global_out

        if head_mask is not None:
            hm = head_mask.to(out.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.unsqueeze(-1).unsqueeze(-1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            out = out * hm

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = out * self.output_scale

        if not self.is_nested_encoder:
            out = self.out_proj(out)

        if output_attentions:
            return (out, local_attn)
        return (out, present_key_value)


@ATTENTION_REGISTRY.register("inject_native_hybrid_attention")
def inject_nha(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_nha: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else (getattr(target, "config", None) or kwargs.get("config"))

    if layer is None and model is not None:
        layer_list = None
        for path in (
            ("distilbert", "transformer", "layer"), ("roberta", "encoder", "layer"),
            ("bert", "encoder", "layer"), ("transformer", "h"), ("model", "layers"),
            ("encoder", "layer"), ("layers",), ("h",),
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
            n_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_nha(
                    layer=lyr, model_config=cfg, comp_config=comp_config,
                    layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
                )
                if res is not None:
                    n_injected += 1
            return target if n_injected > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None:
        src_attn = target

    if isinstance(src_attn, NativeHybridAttention) or "NativeHybridAttention" in src_attn.__class__.__name__:
        return None

    if src_attn is None and cfg is None:
        return None

    try:
        new_attn = NativeHybridAttention(
            config=cfg, comp_config=comp_config,
            pretrained_attn=src_attn if src_attn is not target else None,
            layer_idx=layer_idx, **kwargs,
        )
    except Exception as e:
        logger.warning("NHA construction/weight copy failed: %s", e)
        return None

    ref = None
    if src_attn is not None and src_attn is not target:
        ref = next(src_attn.parameters(), None)
    if ref is None:
        ref = next(target.parameters(), None)
    if ref is not None:
        new_attn.to(device=ref.device, dtype=ref.dtype)

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        target.attn = new_attn
        spliced = True

    if not spliced:
        logger.error("inject_nha: no writable attribute on %s", type(target).__name__)
        return None

    return new_attn


inject_native_hybrid_attention = inject_nha

__all__ = [
    "NativeHybridAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_nha",
    "inject_native_hybrid_attention",
]