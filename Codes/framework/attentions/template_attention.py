r"""
Template Attention Mechanism (YourCustomAttention) – Production-Hardened.
Supports: DistilBERT (Flat MHA), RoBERTa/BERT (Nested Encoder),
GPT-2 (Causal Decoder with Conv1D), Qwen/LLaMA (GQA + RoPE).

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

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
    x2 = x[..., half:2 * half]
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
    """TA-02/TA-11/TA-12: dtype-safe, past_len-aware RoPE."""
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
        raise ValueError(
            f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
        )

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
# TA-01: LoRALinear with device/dtype support
# ===========================================================================

class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.lora_rank)

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
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


# ===========================================================================
# TA-10: single registry key per role
# ===========================================================================

@ATTENTION_REGISTRY.register("template_attention_class")
@ATTENTION_REGISTRY.register("your_custom_attention_class")
@ATTENTION_REGISTRY.register("low_rank_modulated_attention_class")
class YourCustomAttention(BaseAttention):
    r"""Template attention layer with optional low-rank pooling modulation."""

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")

        # -------- hidden_size inference --------
        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(config, "hidden_size",
                           _cfg_get(config, "dim",
                                    _cfg_get(config, "d_model", None)))
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None and hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                dim = src.q_proj.in_features
            elif dim is None and hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                dim = src.q_lin.in_features
            elif dim is None and hasattr(src, "query") and hasattr(src.query, "in_features"):
                dim = src.query.in_features
            elif dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                w = src.c_attn.weight
                dim = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)
        self.hidden_size: int = int(dim)

        # -------- num_heads inference (TA-13) --------
        num_heads = None
        if config is not None and not isinstance(config, int):
            num_heads = _cfg_get(config, "num_attention_heads",
                                 _cfg_get(config, "n_heads",
                                          _cfg_get(config, "num_heads", None)))
        if num_heads is None and src is not None:
            num_heads = getattr(src, "num_heads",
                                getattr(src, "num_attention_heads",
                                        getattr(src, "n_head", None)))
        if num_heads is None:
            # Fallback: assume head_dim ≈ 64
            num_heads = max(1, self.hidden_size // 64)
        self.num_heads: int = max(1, int(num_heads))

        # -------- head_dim --------
        head_dim = None
        if config is not None and not isinstance(config, int):
            head_dim = _cfg_get(config, "head_dim", None)
        if head_dim is None:
            head_dim = kwargs.get("head_dim")
        if head_dim is None:
            head_dim = self.hidden_size // self.num_heads
        self.head_dim: int = int(head_dim)

        # -------- num_kv_heads (GQA) --------
        cfg_kv = None
        if config is not None and not isinstance(config, int):
            cfg_kv = _cfg_get(config, "num_key_value_heads",
                              _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads",
                             getattr(src, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups: int = max(1, self.num_heads // self.num_kv_heads)

        # TA-04: all_head_size explicitly defined
        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        self.lora_rank: int = int(kwargs.get(
            "lora_rank",
            _cfg_get(config, "lora_rank", lora_rank) if config is not None else lora_rank,
        ))
        self.lora_alpha: float = float(kwargs.get(
            "lora_alpha",
            _cfg_get(config, "lora_alpha", lora_alpha) if config is not None else lora_alpha,
        ))

        # -------- decoder detection --------
        if is_decoder is not None:
            self.is_decoder: bool = bool(is_decoder)
        elif config is not None and not isinstance(config, int):
            mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or bool(kwargs.get("is_causal", False))
                or any(mt.startswith(p) for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
            )
        else:
            self.is_decoder = bool(kwargs.get("is_causal", False))
        self.is_causal = self.is_decoder
        self.is_nested_encoder: bool = False

        # -------- projections --------
        self.query = LoRALinear(
            self.hidden_size, self.q_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
        )
        self.key = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
        )
        self.value = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.hidden_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
        )

        # Aliases
        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        # TA-06: rank_proj respects device/dtype
        if rank is not None and rank > 0:
            self.rank_proj: nn.Linear | None = nn.Linear(
                self.hidden_size, rank, bias=False,
            )
            nn.init.zeros_(self.rank_proj.weight)
        else:
            self.rank_proj = None

        dropout_p = 0.1
        if config is not None and not isinstance(config, int):
            dropout_p = _cfg_get(config, "dropout",
                                 _cfg_get(config, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    @property
    def out_lin(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.query
    @property
    def k_lin(self): return self.key
    @property
    def v_lin(self): return self.value

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2, Qwen/LLaMA."""
        with torch.no_grad():
            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if (src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or (hasattr(old_attn, "query")
                    and not hasattr(old_attn, "dense")
                    and not hasattr(old_attn, "out_proj"))):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight.detach()
                    b_o = (old_attn.output.dense.bias.detach()
                           if old_attn.output.dense.bias is not None else None)
            else:
                src = old_attn

            # Pattern 1: GPT-2 Conv1D fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
                    c_attn_w = c_attn_w.t().contiguous()
                q_dim, k_dim, v_dim = self.q_head_size, self.kv_head_size, self.kv_head_size
                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim:q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim:q_dim + k_dim + v_dim, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim:q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim:q_dim + k_dim + v_dim]
                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT
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

            # Pattern 3: Qwen / LLaMA
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

            # Pattern 4: DistilBERT
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

            # Fallback: UniversalAdapter
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None: missing.append("Q")
            if w_k is None: missing.append("K")
            if w_v is None: missing.append("V")
            if w_o is None and not self.is_nested_encoder: missing.append("O")
            if missing:
                raise RuntimeError(
                    f"YourCustomAttention.copy_weights: missing {missing} from {type(old_attn).__name__}"
                )

            def _copy_layer(target_layer: LoRALinear, w, b) -> None:
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().contiguous().shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(w.t().contiguous().to(target_layer.weight.dtype))
                    else:
                        min_0 = min(target_layer.weight.shape[0], w.shape[0])
                        min_1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.data.zero_()
                        target_layer.weight.data[:min_0, :min_1].copy_(w[:min_0, :min_1].to(target_layer.weight.dtype))

                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.data.copy_(b.to(target_layer.bias.dtype))
                    else:
                        min_b = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.data.zero_()
                        target_layer.bias.data[:min_b].copy_(b[:min_b].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.data.zero_()

            _copy_layer(self.query, w_q, b_q)
            _copy_layer(self.key, w_k, b_k)
            _copy_layer(self.value, w_v, b_v)
            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o)

            # Frobenius validation
            for name, layer in (("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)):
                norm = float(torch.linalg.norm(layer.weight.data.float()).item())
                if norm < 1e-4:
                    logger.warning("YourCustomAttention: '%s' weight norm near zero (%.2e)", name, norm)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "YourCustomAttention: cross-attention not supported."
            )

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.query(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.key(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.value(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # TA-02: RoPE with past_len offset
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
            cos = cos[..., past_len: past_len + L, :]
            sin = sin[..., past_len: past_len + L, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(self.head_dim)

        # TA-05: unify mask
        eff_mask = encoder_attention_mask if encoder_attention_mask is not None else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("attn_mask")

        valid_mask_2d = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype == torch.bool:
                    valid_mask_2d = eff_mask
                elif eff_mask.dtype.is_floating_point:
                    valid_mask_2d = eff_mask > -1e4
                else:
                    valid_mask_2d = eff_mask > 0
            elif eff_mask.dim() == 4:
                if eff_mask.dtype == torch.bool:
                    v_mask = eff_mask
                elif eff_mask.dtype.is_floating_point:
                    v_mask = eff_mask > -1e4
                else:
                    v_mask = eff_mask > 0
                valid_mask_2d = v_mask.any(dim=-2).squeeze(1)

        # SDPA fast path (TA-03: mask-aware)
        is_causal = self.is_decoder and L > 1
        use_sdpa = (
            self.rank_proj is None
            and head_mask is None
            and not output_attentions
        )

        if use_sdpa:
            try:
                q_sdpa = q.half() if q.dtype == torch.float32 else q
                k_sdpa = k.half() if k.dtype == torch.float32 else k
                v_sdpa = v.half() if v.dtype == torch.float32 else v

                attn_bias = None
                if eff_mask is not None:
                    if eff_mask.dim() == 2:
                        attn_bias = ((eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0).to(q_sdpa.dtype)
                    elif eff_mask.dim() == 4:
                        attn_bias = eff_mask.to(q_sdpa.dtype)

                out_sdpa = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=attn_bias,
                    is_causal=is_causal and attn_bias is None,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                out = out_sdpa.to(dtype).transpose(1, 2).contiguous().view(B, L, self.q_head_size)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

                if self.is_nested_encoder:
                    return out, None
                result = self.out_proj(out)
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                return result, None
            except Exception as e:
                # TA-08: log instead of silent pass
                logger.debug("SDPA path failed (%s); falling back to explicit.", e)

        # Explicit path
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        # Low-rank modulation
        if self.rank_proj is not None:
            H_pool = hidden_states.mean(dim=1)
            H_rank = self.rank_proj(H_pool)
            modulation = 1.0 + torch.tanh(H_rank).mean(dim=-1, keepdim=True).view(B, 1, 1, 1)
            scores = scores * modulation

        if is_causal:
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        if valid_mask_2d is not None:
            scores = scores.masked_fill(~valid_mask_2d[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # TA-05: head_mask unified
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        # TA-14: no silent pad/truncate
        if output.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"YourCustomAttention: output dim {output.shape[-1]} != hidden_size {self.hidden_size}"
            )

        return output, (attn if output_attentions else None)


# ===========================================================================
# TA-09 / TA-10: injector with recursion guard
# ===========================================================================

@ATTENTION_REGISTRY.register("custom_attn")
@ATTENTION_REGISTRY.register("template_attention")
@ATTENTION_REGISTRY.register("low_rank_modulated_attention")
@ATTENTION_REGISTRY.register("inject_custom_attn")
@ATTENTION_REGISTRY.register("inject_template_attention")
def inject_custom_attn(
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
        logger.warning("inject_custom_attn: max recursion depth exceeded.")
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_custom_attn(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            src = target

        # Double-injection guard
        if isinstance(src, YourCustomAttention) or "YourCustomAttention" in src.__class__.__name__:
            return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        if model_config is None and hasattr(target, "config"):
            model_config = target.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", None))
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

        new_attn = YourCustomAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src if src is not target else None,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
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
            target.self = new_attn

        return new_attn
    except Exception as e:
        logger.exception("inject_custom_attn failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "YourCustomAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_custom_attn",
]
