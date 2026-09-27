"""
SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Block (Hardened).
Supports:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
- True chunked local sliding-window attention (O(L · W))
- No double-residual or double LayerNorm
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.attentions.simple_ssm import SimpleSSM
except ImportError:
    try:
        from simple_ssm import SimpleSSM
    except ImportError:
        SimpleSSM = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
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
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)

    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

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

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


class SpectralMamba(nn.Module):
    def __init__(self, d_model: int = 768, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", d_model))
        elif hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if SimpleSSM is not None:
            self.mamba = SimpleSSM(config=config, d_model=d_model, **kwargs)
        else:
            self.mamba = nn.Identity()
        self.dt_scale = nn.Parameter(torch.full((d_model,), 5.0))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.mamba(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        gate = torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        out = out * gate
        return out, None


@ATTENTION_REGISTRY.register("samba_m3", supports_causal=True, supports_gqa=True)
class SAMBAM3Block(nn.Module):
    r"""SAMBA-M3: Combination of Local Attention and Spectral Mamba."""

    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        window: int = 128,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(config, int):
            d_model = config
            config = None
        if d_model is None:
            if config is not None:
                d_model = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)
        if n_heads is None:
            n_heads = _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "n_heads", kwargs.get("num_attention_heads", 12)),
            )
        window = _cfg_get(config, "window_size", window)

        self.d_model: int = int(d_model)
        self.n_heads: int = int(n_heads)

        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        self.head_dim: int = self.d_model // self.n_heads
        self.window: int = window
        self.lora_rank: int = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha: float = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.n_heads
        self.num_key_value_groups: int = max(1, self.n_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.n_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder / Causal detection
        model_type = str(_cfg_get(config, "model_type", "")).lower()
        is_dec = _cfg_get(config, "is_decoder", _cfg_get(config, "is_causal", kwargs.get("is_causal", False)))
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if is_decoder is not None:
            is_dec = is_decoder
        self.is_decoder: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

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

        self.mamba = SpectralMamba(self.d_model, config=config, is_decoder=self.is_decoder, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # Start at equal mix
        dropout_p = _cfg_get(config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> LoRALinear:
        return self.q_lin

    @property
    def key(self) -> LoRALinear:
        return self.k_lin

    @property
    def value(self) -> LoRALinear:
        return self.v_lin

    @property
    def q_proj(self) -> LoRALinear:
        return self.q_lin

    @property
    def k_proj(self) -> LoRALinear:
        return self.k_lin

    @property
    def v_proj(self) -> LoRALinear:
        return self.v_lin

    def get_ortho_loss(self) -> float:
        return 0.0

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

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

            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                p_cls = getattr(src.c_attn, "__class__", type(src.c_attn)).__name__
                if ("Conv1D" in p_cls) or hasattr(src.c_attn, "nf") or (c_attn_w.shape[0] == self.d_model and c_attn_w.shape[1] != self.d_model):
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
                if ("Conv1D" in p_proj_cls) or hasattr(src.c_proj, "nf") or (out_w.shape[0] == self.d_model and out_w.shape[1] != self.d_model):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

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

            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

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

            for target_mod, w_src, b_src, name in [
                (self.q_lin, w_q, b_q, "q_lin"),
                (self.k_lin, w_k, b_k, "k_lin"),
                (self.v_lin, w_v, b_v, "v_lin"),
            ]:
                if w_src is not None:
                    tw = target_mod.weight
                    w_cast = w_src.to(tw.dtype)
                    min_r = min(tw.shape[0], w_cast.shape[0])
                    min_c = min(tw.shape[1], w_cast.shape[1])
                    tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])
                    if b_src is not None and target_mod.bias is not None:
                        tb = target_mod.bias
                        b_cast = b_src.to(tb.dtype)
                        min_b = min(tb.numel(), b_cast.numel())
                        tb.data[:min_b].copy_(b_cast[:min_b])
                    fnorm = float(torch.linalg.norm(tw.data.float()))
                    if fnorm < 1e-4:
                        logger.warning(f"SAMBAM3Block: {name} norm too small ({fnorm:.2e})")

            if w_o is not None:
                tw_o = self.out_lin.weight
                w_o_cast = w_o.to(tw_o.dtype)
                min_r = min(tw_o.shape[0], w_o_cast.shape[0])
                min_c = min(tw_o.shape[1], w_o_cast.shape[1])
                tw_o.data[:min_r, :min_c].copy_(w_o_cast[:min_r, :min_c])
                if b_o is not None and self.out_lin.bias is not None:
                    tb_o = self.out_lin.bias
                    b_o_cast = b_o.to(tb_o.dtype)
                    min_b = min(tb_o.numel(), b_o_cast.numel())
                    tb_o.data[:min_b].copy_(b_o_cast[:min_b])
                fnorm_o = float(torch.linalg.norm(tw_o.data.float()))
                if fnorm_o < 1e-4:
                    logger.warning(f"SAMBAM3Block: out_lin norm too small ({fnorm_o:.2e})")

    def _chunked_sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, L, d = q.shape
        scale = 1.0 / math.sqrt(d)
        chunk = 64
        outputs = []

        for start in range(0, L, chunk):
            end = min(start + chunk, L)
            q_c = q[:, :, start:end]  # [B, H, C_q, d]
            C_q = end - start

            k_start = max(0, start - self.window)
            k_end = end if self.is_decoder else min(L, end + self.window)
            k_c = k[:, :, k_start:k_end]  # [B, H, C_k, d]
            v_c = v[:, :, k_start:k_end]  # [B, H, C_k, d]

            scores = torch.matmul(q_c.float(), k_c.transpose(-2, -1).contiguous().float()) * scale

            q_idx = torch.arange(start, end, device=q.device).view(1, 1, C_q, 1)
            k_idx = torch.arange(k_start, k_end, device=q.device).view(1, 1, 1, -1)
            dist = (q_idx - k_idx).abs()
            valid = dist <= self.window
            if self.is_decoder:
                valid = valid & (k_idx <= q_idx)

            scores = scores.masked_fill(~valid, -1e4)

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    pad = (attention_mask[:, k_start:k_end] == 0).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(pad, -1e4)
                elif attention_mask.dim() == 3:
                    scores = scores + attention_mask[:, start:end, k_start:k_end].unsqueeze(1).to(scores.dtype)
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask[:, :, start:end, k_start:k_end].to(scores.dtype)

            probs = F.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs.to(v.dtype))

            if head_mask is not None:
                if head_mask.dim() == 1:
                    hm = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    hm = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
                elif head_mask.dim() == 3:
                    hm = head_mask.unsqueeze(-1).to(probs.dtype)
                else:
                    hm = head_mask
                probs = probs * hm

            out_c = torch.matmul(probs, v_c)
            outputs.append(out_c)

        return torch.cat(outputs, dim=2)

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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("SAMBAM3Block does not support cross-attention")

        B, L, D = hidden_states.shape

        q = self.q_lin(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads != self.n_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        attn_out = self._chunked_sliding_window_attention(
            q, k, v, attention_mask=attention_mask, head_mask=head_mask
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.q_head_size)

        target_dtype = hidden_states.dtype
        attn_out = attn_out.to(target_dtype)
        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask, **kwargs)
        mamba_out = mamba_out.to(target_dtype)
        alpha = torch.sigmoid(self.alpha).to(target_dtype)

        # Remove internal double residual and norm
        if self.is_nested_encoder:
            out = alpha * attn_out + (1.0 - alpha) * mamba_out
        else:
            attn_proj = self.out_lin(attn_out).to(target_dtype)
            out = alpha * attn_proj + (1.0 - alpha) * mamba_out

        out = torch.nan_to_num(out.to(target_dtype), nan=0.0)

        if attention_mask is not None and attention_mask.dim() == 2:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)

        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1])) if out.shape[-1] < D else out[..., :D]

        return (out.to(target_dtype), None)


@ATTENTION_REGISTRY.register("samba_m3_inject")
@ATTENTION_REGISTRY.register("inject_samba_m3")
def inject_samba_m3(
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
            res = inject_samba_m3(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    """Safe injection hook for SAMBAM3Block."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        d_model = 768
        n_heads = 12
        if src is not None:
            d_model = getattr(src, "hidden_size", getattr(src, "embed_dim", getattr(src, "all_head_size", 768)))
            if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                d_model = src.q_proj.in_features
            elif hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                d_model = src.q_lin.in_features
            n_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", 12))

        if model_config is not None:
            d_model = _cfg_get(model_config, "hidden_size", _cfg_get(model_config, "dim", d_model))
            n_heads = _cfg_get(model_config, "num_attention_heads", n_heads)

        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
        lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 32.0))

        new_block = SAMBAM3Block(
            config=model_config,
            d_model=d_model,
            n_heads=n_heads,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_block
            else:
                layer.attention = new_block
            return new_block
        layer.attention = new_block
        return new_block
    except Exception as e:
        logger.error("inject_samba_m3 failed at layer %d: %s", layer_idx, e)
        return None
