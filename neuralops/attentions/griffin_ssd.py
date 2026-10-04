"""
Griffin SSD: Hybrid Local Sliding-Window Attention and Global State Space Duality (SSD) Kernel.

Combines bidirectional/causal local sliding-window self-attention with multi-head
State Space Duality linear recurrence, LoRA residual adapters, and multi-architecture pretrained weight transfer.
Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.gated_delta_net import LoRALinear
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (
            (module.__class__.__name__ == "Conv1D")
            or (base.__class__.__name__ == "Conv1D")
            or (w.ndim == 2 and w.shape[0] < w.shape[1])
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2 and cos.shape[0] == q.shape[-2]:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
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

    rotary_dim_even = 2 * (min(rotary_dim, head_dim) // 2)
    if rotary_dim_even == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim_even], q[..., rotary_dim_even:]
    k_rot, k_pass = k[..., :rotary_dim_even], k[..., rotary_dim_even:]
    cos_cut = cos[..., :rotary_dim_even].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim_even].to(q_rot.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed

    return q, k


@ATTENTION_REGISTRY.register("griffin_ssd")
@ATTENTION_REGISTRY.register("griffin_ssd_block")
class GriffinSSDBlock(BaseAttention):
    r"""Griffin Hybrid Block combining local window attention with global SSD linear attention."""

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        window_size: int = 64,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        def _get(k: str, default: Any = None) -> Any:
            for obj in (
                comp_config,
                config,
                getattr(config, "compression_config", None),
            ):
                if obj is not None:
                    if isinstance(obj, dict) and k in obj and obj[k] is not None:
                        return obj[k]
                    elif hasattr(obj, k):
                        val = getattr(obj, k)
                        if val is not None:
                            return val
            return kwargs.get(k, default)

        # Resolve d_model
        if d_model is None:
            d_model = _get("hidden_size", _get("d_model", _get("n_embd", _get("dim", None))))
        if d_model is None and old_attn is not None:
            d_model = (
                getattr(old_attn, "hidden_size", None)
                or getattr(old_attn, "embed_dim", None)
                or getattr(old_attn, "all_head_size", None)
            )
            if d_model is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(old_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        d_model = m.in_features
                        break
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        # Resolve num_heads
        if num_heads is None:
            num_heads = (
                _get("num_attention_heads", None)
                or _get("num_heads", None)
                or _get("n_heads", None)
                or _get("n_head", None)
            )
        if num_heads is None and old_attn is not None:
            num_heads = getattr(
                old_attn, "num_heads", getattr(old_attn, "num_attention_heads", None)
            )
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim: int = int(d_model)
        self.num_heads: int = int(num_heads)

        resolved_head_dim = _get("head_dim", None)
        if resolved_head_dim is None and old_attn is not None:
            resolved_head_dim = getattr(old_attn, "head_dim", None)
            if resolved_head_dim is None:
                for q_cand in ("q_proj", "q_lin", "query", "W_q"):
                    m = getattr(old_attn, q_cand, None)
                    if m is not None and hasattr(m, "out_features"):
                        resolved_head_dim = m.out_features // max(1, self.num_heads)
                        break
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim: int = int(resolved_head_dim)

        kv_heads = _get("num_key_value_heads", None)
        if kv_heads is None and old_attn is not None:
            kv_heads = getattr(
                old_attn,
                "num_key_value_heads",
                getattr(old_attn, "num_kv_heads", getattr(old_attn, "n_kv_heads", None)),
            )
            if (
                kv_heads is None
                and hasattr(old_attn, "k_proj")
                and hasattr(old_attn.k_proj, "out_features")
                and self.head_dim > 0
            ):
                kv_heads = old_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads: int = int(kv_heads if kv_heads is not None else self.num_heads)

        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.window_size: int = int(_get("window_size", kwargs.get("window_size", window_size)))

        # Decoder / causal detection
        model_type = str(_get("model_type", "")).lower()
        is_dec = _get(
            "is_decoder",
            _get("is_causal", kwargs.get("is_causal", kwargs.get("is_decoder", False))),
        )
        if (
            any(
                model_type.startswith(pfx)
                for pfx in (
                    "gpt",
                    "qwen",
                    "llama",
                    "mistral",
                    "bloom",
                    "opt",
                    "falcon",
                    "codegen",
                )
            )
            or hasattr(config, "architectures")
            and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])
            )
        ):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)

        # Nested encoder detection
        is_nested = False
        if old_attn is not None:
            src_cls = old_attn.__class__.__name__
            if (
                src_cls
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())
                or (hasattr(old_attn, "self") and hasattr(old_attn, "output"))
            ):
                is_nested = True
        if (
            not is_nested
            and model_type in ["roberta", "bert", "xlm-roberta"]
            and not self.is_decoder
        ):
            is_nested = True
        self.is_nested_encoder: bool = bool(is_nested)

        # LoRA setup
        self.lora_rank: int = int(_get("lora_rank", kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha: float = float(_get("lora_alpha", kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base: bool = self.lora_rank > 0

        # Multi-head projections
        self.W_q = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_k = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_v = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.ssd_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Learnable decay gate for global SSD branch (0.0 -> sigmoid(0) = 0.5)
        self.decay_param = nn.Parameter(torch.zeros(self.num_heads, 1, 1))

        self.rotary_emb = getattr(old_attn, "rotary_emb", None) if old_attn is not None else None
        self.comp_config = comp_config

        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights across multiple model families."""
        if old_attn is None:
            return

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        try:
            adapter = UniversalAdapter()
            for proj in ["q", "k", "v", "o"]:
                w, b = adapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
        except Exception:
            pass

        # 2. GPT-2 Fused c_attn
        if hasattr(old_attn, "c_attn") and extracted["q"][0] is None:
            c_w, c_b = _unpack_weight_bias(old_attn.c_attn)
            if c_w is not None:
                if c_w.dim() == 2 and c_w.shape[1] == 3 * self.dim:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
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

            if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
                o_w, o_b = _unpack_weight_bias(old_attn.c_proj)
                if o_w is not None:
                    if o_w.dim() == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim:
                        o_w = o_w.t().contiguous()
                    extracted["o"] = (o_w, o_b)

        # 3. RoBERTa / BERT Nested
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            self.is_nested_encoder = True
            s = old_attn.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                extracted[proj_key] = _unpack_weight_bias(mod)
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                extracted["o"] = _unpack_weight_bias(old_attn.output.dense)

        # 4. Flat projections (DistilBERT, Qwen, LLaMA)
        else:
            candidates = {
                "q": ["q_proj", "q_lin", "query"],
                "k": ["k_proj", "k_lin", "key"],
                "v": ["v_proj", "v_lin", "value"],
                "o": ["out_proj", "out_lin", "o_proj", "dense"],
            }
            for proj_key, names in candidates.items():
                if extracted[proj_key][0] is None:
                    for name in names:
                        if hasattr(old_attn, name):
                            mod = getattr(old_attn, name)
                            cw, cb = _unpack_weight_bias(mod)
                            if cw is not None:
                                extracted[proj_key] = (cw, cb)
                                break

        # Copy into modules
        target_map = {
            "q": self.W_q,
            "k": self.W_k,
            "v": self.W_v,
            "o": self.out_proj,
        }
        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is not None:
                    if w.shape == target_mod.weight.shape:
                        target_mod.weight.copy_(w)
                    elif w.t().contiguous().shape == target_mod.weight.shape:
                        target_mod.weight.copy_(w.t().contiguous())
                    elif (
                        w.shape[0] < target_mod.weight.shape[0]
                        and target_mod.weight.shape[0] % w.shape[0] == 0
                    ):
                        repeats = target_mod.weight.shape[0] // w.shape[0]
                        target_mod.weight.copy_(w.repeat(repeats, 1))
                    else:
                        min_0 = min(target_mod.weight.shape[0], w.shape[0])
                        min_1 = min(target_mod.weight.shape[1], w.shape[1])
                        target_mod.weight.zero_()
                        target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                    if b is not None and getattr(target_mod, "bias", None) is not None:
                        if b.shape == target_mod.bias.shape:
                            target_mod.bias.copy_(b)
                        elif (
                            b.shape[0] < target_mod.bias.shape[0]
                            and target_mod.bias.shape[0] % b.shape[0] == 0
                        ):
                            repeats = target_mod.bias.shape[0] // b.shape[0]
                            target_mod.bias.copy_(b.repeat(repeats))

            # Initialize ssd_proj from q
            if extracted["q"][0] is not None:
                w_q = extracted["q"][0]
                if w_q.shape == self.ssd_proj.weight.shape:
                    self.ssd_proj.weight.copy_(w_q)
                elif w_q.t().contiguous().shape == self.ssd_proj.weight.shape:
                    self.ssd_proj.weight.copy_(w_q.t().contiguous())

        if self.freeze_base:
            for mod in [self.W_q, self.W_k, self.W_v, self.ssd_proj, self.out_proj]:
                mod.freeze_base_weights()

    def _ssd_scan_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """State-Space Duality (SSD) matrix scan supporting forward causal and bidirectional mode."""
        B, H, L, d = q.shape
        target_dtype = q.dtype

        decay_typed = decay.to(target_dtype)
        log_decay = torch.log(decay_typed.clamp(min=1e-4, max=1.0 - 1e-4))  # [1, H, 1, 1]
        pos = torch.arange(L, device=q.device, dtype=target_dtype)
        pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [L, L]

        attn_scores = (q @ k.transpose(-1, -2).contiguous()).to(target_dtype)  # [B, H, L, L]

        if mask is not None:
            mask_typed = (
                mask.to(target_dtype) if mask.dim() == 2 else mask[:, 0, 0, :].to(target_dtype)
            )
            mask_2d = mask_typed.unsqueeze(1).unsqueeze(-1) * mask_typed.unsqueeze(1).unsqueeze(-2)
        else:
            mask_2d = None

        if self.is_decoder:
            # Forward causal decay matrix
            tril_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=target_dtype))
            D_causal = (
                torch.exp(
                    (pos_diff.unsqueeze(0).unsqueeze(0) * log_decay).clamp(min=-40.0, max=0.0)
                ).to(target_dtype)
                * tril_mask
            )
            if mask_2d is not None:
                D_causal = D_causal * mask_2d
            D_causal = D_causal.to(target_dtype)
            v_typed = v.to(target_dtype)
            y_fwd = (attn_scores * D_causal) @ v_typed  # [B, H, L, d]
            return y_fwd.to(target_dtype), None
        else:
            # Symmetric distance decay for bidirectional encoders: D_sym(i, j) = exp(|i - j| * log_decay)
            dist = torch.abs(pos_diff).unsqueeze(0).unsqueeze(0)
            D_sym = torch.exp((dist * log_decay).clamp(min=-40.0, max=0.0)).to(target_dtype)
            if mask_2d is not None:
                D_sym = D_sym * mask_2d
            D_sym = D_sym.to(target_dtype)
            v_typed = v.to(target_dtype)
            y_sym = (attn_scores * D_sym) @ v_typed  # [B, H, L, d]
            return y_sym.to(target_dtype), None

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for Griffin SSD attention."""
        if encoder_hidden_states is not None:
            return hidden_states, None

        mask = kwargs.get("mask")
        if attention_mask is None and mask is not None:
            attention_mask = mask

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        target_dtype = hidden_states.dtype
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        # 1. Multi-head Projections
        q = self.W_q(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous().to(target_dtype)
        k = (
            self.W_k(hidden_states)
            .view(B, L, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
            .to(target_dtype)
        )
        v = (
            self.W_v(hidden_states)
            .view(B, L, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
            .to(target_dtype)
        )
        q_ssd = (
            self.ssd_proj(hidden_states)
            .view(B, L, H, d)
            .transpose(1, 2)
            .contiguous()
            .to(target_dtype)
        )

        # Apply RoPE if provided or probe from rotary_emb
        pos_emb = (
            position_embeddings
            if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        pos_ids = kwargs.get("position_ids")
        if pos_emb is not None:
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
                cos, sin = pos_emb[0], pos_emb[1]
            else:
                cos, sin = pos_emb, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)
        elif self.rotary_emb is not None:
            rotary_mod = self.rotary_emb
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            cos, sin = None, None
            try:
                cos, sin = rotary_mod(q, seq_len=L)
            except Exception:
                try:
                    cos, sin = rotary_mod(q, pos_ids)
                except Exception:
                    try:
                        cos, sin = rotary_mod(q)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # GQA repeat
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        # 2. Local Sliding Window Attention
        scores = torch.matmul(q, k.transpose(-1, -2).contiguous()) / math.sqrt(d)  # [B, H, L, L]
        idx = torch.arange(L, device=hidden_states.device)
        dist = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
        window_mask = dist > self.window_size

        mask_val = -10000.0
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), mask_val)

        if self.is_decoder:
            causal_mask = torch.triu(
                torch.ones(L, L, device=hidden_states.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_val)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
                pad_mask = (attention_mask == 0).view(B, 1, 1, L)
                scores = scores.masked_fill(pad_mask, mask_val)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        local_attn = F.softmax(scores, dim=-1).to(target_dtype)
        local_attn = torch.nan_to_num(local_attn, nan=0.0)
        local_out = torch.matmul(local_attn, v).to(target_dtype)  # [B, H, L, d]

        # 3. Global SSD Recurrence
        q_norm = F.normalize(q_ssd, p=2, dim=-1).to(target_dtype)
        k_norm = (F.normalize(k, p=2, dim=-1) / (d**0.5)).to(target_dtype)
        decay = torch.sigmoid(self.decay_param.to(target_dtype)).view(1, H, 1, 1)

        ssd_fwd, ssd_bwd = self._ssd_scan_parallel(q_norm, k_norm, v, decay, mask_bool)

        if self.is_decoder or ssd_bwd is None:
            global_out = ssd_fwd.to(target_dtype)
        else:
            global_out = (0.5 * (ssd_fwd + ssd_bwd)).to(target_dtype)

        # 4. Fused Output
        fused = (
            (local_out + global_out)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, self.all_head_size)
            .to(target_dtype)
        )
        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused).to(target_dtype)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out.to(target_dtype), None


@ATTENTION_REGISTRY.register("griffin_ssd_inject")
@ATTENTION_REGISTRY.register("griffin_ssd")
def inject_griffin_ssd(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Any | None:
    r"""Injection hook for GriffinSSDBlock supporting single layers and whole models."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_griffin_ssd(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # 2. Single layer injection
    curr_attn = old_attn
    if curr_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                curr_attn = target.attention.self
            else:
                curr_attn = target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
        elif hasattr(target, "attn"):
            curr_attn = target.attn
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn

    if curr_attn is None and target is not None and hasattr(target, "forward"):
        curr_attn = target

    if curr_attn is None and cfg is None:
        return None

    try:
        new_layer = GriffinSSDBlock(
            config=cfg,
            old_attn=curr_attn,
            lora_rank=lora_rank,
            **kwargs,
        )
    except Exception as ex:
        logger.warning("Failed to instantiate GriffinSSDBlock: %s", ex)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_layer.is_nested_encoder = True
                target.attention.self = new_layer
            else:
                new_layer.is_nested_encoder = False
                target.attention = new_layer
        elif hasattr(target, "self"):
            new_layer.is_nested_encoder = True
            target.self = new_layer
        elif hasattr(target, "attn"):
            new_layer.is_nested_encoder = False
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            new_layer.is_nested_encoder = False
            target.self_attn = new_layer

    return new_layer
