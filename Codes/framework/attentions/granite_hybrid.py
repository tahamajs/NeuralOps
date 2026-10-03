"""Granite Hybrid Attention-SSM Layer with MoE Feed-Forward.

Combines Mamba-2 SSM recurrent layers with dense Multihead Attention (MHA) and
Top-k Mixture-of-Experts (MoE) feed-forward, with LoRA adapters and pretrained weight transfer.
Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.gated_delta_net import LoRALinear
from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

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
        if (module.__class__.__name__ == "Conv1D") or (base.__class__.__name__ == "Conv1D") or (w.ndim == 2 and w.shape[0] < w.shape[1]):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Expands grouped key/value heads to match query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


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


class Mamba2(nn.Module):
    r"""Simplified Mamba-2 State Space Model (SSM) block."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.expand: int = expand
        inner_dim: int = d_model * expand

        self.in_proj: nn.Linear = nn.Linear(d_model, inner_dim * 2)
        self.conv1d: nn.Conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv - 1
        )
        self.dt_proj: nn.Linear = nn.Linear(inner_dim, inner_dim)
        self.A_log: nn.Parameter = nn.Parameter(
            torch.log(torch.rand(inner_dim, d_state) * 0.1)
        )
        self.D: nn.Parameter = nn.Parameter(torch.ones(inner_dim))
        self.out_proj: nn.Linear = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        B, L, D = x.shape
        inner_dim = self.in_proj.out_features // 2
        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2).contiguous())[..., :L].transpose(1, 2).contiguous()
        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0)).to(dtype=x.dtype)

        dA_exp_arg = torch.clamp(torch.einsum("bli,in->blin", dt, A), min=-20.0, max=0.0)
        dA = torch.exp(dA_exp_arg)
        dBX = torch.einsum("bli,blin->blin", x_ssm, dt.unsqueeze(-1))

        # Numerically stable recurrence bounded without exponential overflow
        h_t = torch.zeros(B, inner_dim, self.d_state, device=x.device, dtype=x.dtype)
        h_list = []
        for t in range(L):
            h_t = dA[:, t] * h_t + dBX[:, t]
            h_t = torch.clamp(h_t, min=-1e4, max=1e4)
            h_list.append(h_t)
        h = torch.stack(h_list, dim=1)

        out = (h * self.D.view(1, 1, inner_dim, 1)).sum(dim=-1) * F.silu(z)
        return self.out_proj(out)


class MixtureOfExperts(nn.Module):
    r"""Top-$k$ Token-Choice Mixture-of-Experts (MoE) Layer."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.output_scale = nn.Parameter(torch.ones(1))
        self.dim: int = dim
        self.num_experts: int = num_experts
        self.top_k = top_k
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.router: nn.Linear = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        B, L, D = x.shape
        scores = self.router(x)  # (B, L, E)
        k = min(self.top_k, scores.shape[-1])
        weights, indices = torch.topk(scores, k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (indices == e).any(dim=-1)  # (B, L)
            if mask.any():
                selected_weights = (weights * (indices == e).float()).sum(
                    dim=-1, keepdim=True
                )
                out += (
                    mask.unsqueeze(-1)
                    * selected_weights
                    * self.experts[e](x * mask.unsqueeze(-1))
                )
        return out


@ATTENTION_REGISTRY.register("granite_hybrid")
@ATTENTION_REGISTRY.register("granite_hybrid_block")
class GraniteHybridBlock(BaseAttention):
    r"""Granite Hybrid Attention-SSM Layer with MoE Feed-Forward and LoRA Adaptation."""

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        ratio: int = 2,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        def _get(k: str, default: Any = None) -> Any:
            for obj in (config, comp_config, getattr(config, "compression_config", None)):
                if obj is not None and hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
            return kwargs.get(k, default)

        if dim is None:
            dim = _get("hidden_size", _get("d_model", _get("n_embd", _get("dim", None))))
        if dim is None and src_attn is not None:
            dim = getattr(src_attn, "hidden_size", getattr(src_attn, "embed_dim", None))
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
        dim = dim or kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        if num_heads is None:
            num_heads = _get("num_attention_heads", _get("num_heads", None))
        if num_heads is None and src_attn is not None:
            num_heads = getattr(src_attn, "num_heads", getattr(src_attn, "num_attention_heads", None))
        num_heads = num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 8))

        self.dim: int = int(dim)
        self.num_heads: int = int(num_heads)

        resolved_head_dim = _get("head_dim", None)
        if resolved_head_dim is None and src_attn is not None:
            resolved_head_dim = getattr(src_attn, "head_dim", None)
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim: int = int(resolved_head_dim)

        kv_heads = _get("num_key_value_heads", None)
        if kv_heads is None and src_attn is not None:
            kv_heads = getattr(src_attn, "num_key_value_heads", getattr(src_attn, "num_kv_heads", getattr(src_attn, "n_kv_heads", None)))
            if kv_heads is None and hasattr(src_attn, "k_proj") and hasattr(src_attn.k_proj, "out_features") and self.head_dim > 0:
                kv_heads = src_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads: int = int(kv_heads if kv_heads is not None else self.num_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Decoder detection
        model_type = str(_get("model_type", "")).lower()
        is_dec = _get("is_decoder", _get("is_causal", kwargs.get("is_causal", kwargs.get("is_decoder", False))))
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        # LoRA setup
        self.lora_rank: int = int(_get("lora_rank", kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha: float = float(_get("lora_alpha", kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base: bool = (self.lora_rank > 0)

        self.mamba: Mamba2 = Mamba2(d_model=self.dim)

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        num_exp = int(_get("num_experts", kwargs.get("num_experts", num_experts)))
        tk = int(_get("top_k", kwargs.get("top_k", top_k)))
        rat = int(_get("ratio", kwargs.get("ratio", ratio)))
        self.moe: MixtureOfExperts = MixtureOfExperts(self.dim, self.dim * 4, num_exp, tk)
        self.alpha = nn.Parameter(torch.tensor(3.0))
        self.moe_scale = nn.Parameter(torch.zeros(1))
        self.is_nested_encoder = False
        self.output_scale = nn.Parameter(torch.ones(1))

        self.rotary_emb = getattr(src_attn, "rotary_emb", None) if src_attn is not None else None
        self.comp_config = comp_config

        if src_attn is not None:
            self.copy_weights(src_attn)

    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        try:
            adapter = UniversalAdapter()
            for p in ["q", "k", "v", "o"]:
                w, b = adapter.get_weight_bias(old_attn, p)
                if w is not None:
                    extracted[p] = (w, b)
        except Exception as e:
                print(f'copy_weights failed: {e}')
                logger.warning(f"copy_weights failed: {e}")

        if hasattr(old_attn, "c_attn") and extracted["q"][0] is None:
            c_w, c_b = _unpack_weight_bias(old_attn.c_attn)
            if c_w is not None:
                if c_w.dim() == 2 and c_w.shape[1] == 3 * self.dim:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                extracted["q"] = (c_w[:split_size], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
                o_w, o_b = _unpack_weight_bias(old_attn.c_proj)
                if o_w is not None:
                    if o_w.dim() == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim:
                        o_w = o_w.t().contiguous()
                    extracted["o"] = (o_w, o_b)

        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                extracted[proj_key] = _unpack_weight_bias(getattr(s, sub_name, None))
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                extracted["o"] = _unpack_weight_bias(old_attn.output.dense)

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
                            cw, cb = _unpack_weight_bias(getattr(old_attn, name))
                            if cw is not None:
                                extracted[proj_key] = (cw, cb)
                                break

        target_map = {"q": self.q_proj, "k": self.k_proj, "v": self.v_proj, "o": self.out_proj}
        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is not None:
                    if w.shape == target_mod.weight.shape:
                        target_mod.weight.copy_(w)
                    elif w.t().contiguous().shape == target_mod.weight.shape:
                        target_mod.weight.copy_(w.t().contiguous())
                    elif w.shape[0] < target_mod.weight.shape[0] and target_mod.weight.shape[0] % w.shape[0] == 0:
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
                        elif b.shape[0] < target_mod.bias.shape[0] and target_mod.bias.shape[0] % b.shape[0] == 0:
                            repeats = target_mod.bias.shape[0] // b.shape[0]
                            target_mod.bias.copy_(b.repeat(repeats))

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

    def _attn_forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L_q, _ = q_in.shape
        L_k = k_in.shape[1]
        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        q = self.q_proj(q_in).view(B, L_q, H, Dh).transpose(1, 2).contiguous()
        k = self.k_proj(k_in).view(B, L_k, H_kv, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(v_in).view(B, L_k, H_kv, Dh).transpose(1, 2).contiguous()

        # RoPE application
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        pos_ids = kwargs.get("position_ids")
        if pos_emb is not None:
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
                cos, sin = pos_emb[0], pos_emb[1]
                q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        elif self.rotary_emb is not None:
            rotary_mod = self.rotary_emb
            if pos_ids is None:
                pos_ids = torch.arange(L_q, device=q_in.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(q, seq_len=L_k)
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

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and L_q == L_k:
            causal_mask = torch.triu(torch.full((L_q, L_k), -1e4, device=q_in.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).view(B, 1, 1, L_k)
                scores = scores.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.all_head_size)
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)
        return out, attn_weights

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
        if encoder_hidden_states is not None:
            return hidden_states, None

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        attn_out, attn_weights = self._attn_forward(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        mamba_out = self.mamba(hidden_states)
        gate = torch.sigmoid(self.alpha)
        x = gate * attn_out + (1.0 - gate) * mamba_out
        x = x + self.moe_scale * self.moe(x)

        if x.shape[-1] != hidden_states.shape[-1]:
            if x.shape[-1] < hidden_states.shape[-1]:
                x = F.pad(x, (0, hidden_states.shape[-1] - x.shape[-1]))
            else:
                x = x[..., : hidden_states.shape[-1]]

        return x, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("granite_hybrid_inject")
@ATTENTION_REGISTRY.register("inject_granite_hybrid")
def inject_granite_hybrid(
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
    r"""Injection hook for GraniteHybridBlock supporting single layers and whole models."""
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
                res = inject_granite_hybrid(
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
        new_layer = GraniteHybridBlock(
            config=cfg,
            old_attn=curr_attn,
            lora_rank=lora_rank,
            **kwargs,
        )
    except Exception as ex:
        logger.warning("Failed to instantiate GraniteHybridBlock: %s", ex)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_layer
                new_layer.is_nested_encoder = True
            else:
                target.attention = new_layer
        elif hasattr(target, "self"):
            target.self = new_layer
            new_layer.is_nested_encoder = True
        elif hasattr(target, "attn"):
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer

    return new_layer
