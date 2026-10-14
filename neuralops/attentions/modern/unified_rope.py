"""
Universal Rotary Position Embedding Attention (UnifiedRoPE).
"""
from neuralops.core.common import _cfg_get
from neuralops.core.common import _detect_decoder

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.amp_utils import maybe_half
from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)






# ═══════════════════════════════════════════════════════════════════
# UnifiedRoPE Attention
# ═══════════════════════════════════════════════════════════════════
@ATTENTION_REGISTRY.register(
    "unified_rope",
    supports_causal=True,
    supports_gqa=True,
    supports_rope=True,
)
class UnifiedRoPE(BaseAttention):
    r"""Universal Rotary Position Embedding Attention."""

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        max_seq_len: int = 4096,
        base: int = 10000,
        dropout: float = 0.1,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config=config, rank=rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")
        self.config = config

        # ---- hidden_size ----
        if dim is None:
            if config is not None and not isinstance(config, int):
                dim = _cfg_get(config, "hidden_size",
                              _cfg_get(config, "dim", _cfg_get(config, "d_model", None)))
            if dim is None and src is not None:
                dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
                if dim is None and hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                    dim = src.q_proj.in_features
                elif dim is None and hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                    dim = src.q_lin.in_features
                elif dim is None and hasattr(src, "query") and hasattr(src.query, "in_features"):
                    dim = src.query.in_features
                elif dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                    cw = src.c_attn.weight
                    dim = cw.shape[0] if cw.shape[0] != 3 * cw.shape[1] else cw.shape[1]
            if dim is None:
                dim = kwargs.get("hidden_size", 768)

        # ---- num_heads ----
        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = _cfg_get(
                    config, "num_attention_heads",
                    _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None))
                )
            if num_heads is None and src is not None:
                num_heads = getattr(
                    src, "num_heads",
                    getattr(src, "num_attention_heads", getattr(src, "n_head", None))
                )
            if num_heads is None:
                num_heads = 12

        self.hidden_size = int(dim)
        self.num_heads = max(1, int(num_heads))
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.lora_rank = int(kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank)))
        self.lora_alpha = float(kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha)))

        # ---- num_kv_heads ----
        cfg_kv = (
            _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
            if config is not None else None
        )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups = max(1, self.num_heads // self.num_kv_heads)

        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        # ---- FIX-1: decoder detection ----
        self.is_decoder = _detect_decoder(config, is_decoder, src)
        self.is_nested_encoder = False

        # ---- Projections ----
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and src is not None:
            p = next(src.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        self.q_proj = LoRALinear(
            self.hidden_size, self.q_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.hidden_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, dtype=target_dtype,
        )

        # ---- Aliases ----
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        # ---- Dropout ----
        dropout_p = (
            getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", dropout))
            if config else dropout
        )
        self.dropout = nn.Dropout(float(dropout_p))

        # ---- FIX-5: Chunked cache pre-allocation ----
        self.rotary_dim = 2 * (self.head_dim // 2)
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        else:
            inv_freq = torch.empty(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_max_len = 0
        self._build_cache(max_seq_len)

        # ---- copy weights ----
        if src is not None:
            self.copy_weights(src)

        if target_dtype is not None:
            self.to(target_dtype)

    # ------------------------------------------------------------------
    def _build_cache(self, seq_len: int) -> None:
        """Precompute cos/sin up to seq_len. FIX-5: allocate round-up to avoid rebuilds."""
        # Round up to next power of 2 (min 512) to reduce rebuild frequency
        new_len = max(512, 1 << max(0, (seq_len - 1).bit_length()))
        if new_len <= self._cache_max_len:
            return
        self._cache_max_len = new_len

        if self.rotary_dim == 0 or self.inv_freq.numel() == 0:
            self.register_buffer("cos_cached", torch.ones(new_len, 0), persistent=False)
            self.register_buffer("sin_cached", torch.zeros(new_len, 0), persistent=False)
            return

        t = torch.arange(new_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply_rope(
        self,
        x: torch.Tensor,
        seq_len: int,
        past_len: int = 0,
    ) -> torch.Tensor:
        """FIX-3: Apply RoPE with past_len offset for decode-time consistency.

        Args:
            x: (B, H, L, d)
            seq_len: current sequence length L
            past_len: number of cached tokens before this batch
        """
        if self.rotary_dim == 0 or self.cos_cached.shape[-1] == 0:
            return x

        total_len = past_len + seq_len
        if total_len > self._cache_max_len:
            self._build_cache(total_len)

        rotary_dim = min(self.rotary_dim, self.cos_cached.shape[-1], x.shape[-1])
        rotary_dim = 2 * (rotary_dim // 2)
        if rotary_dim == 0:
            return x

        # FIX-3: slice with past_len offset
        cos = self.cos_cached[past_len:total_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[past_len:total_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
        cos = cos.to(x.device, x.dtype)
        sin = sin.to(x.device, x.dtype)

        from neuralops.core.triton_kernels.fused_rope import (
            fused_unified_rope_offset_triton,
            fused_unified_rope_offset_pytorch,
        )
        try:
            return fused_unified_rope_offset_triton(x, cos, sin, rotary_dim)
        except Exception:
            return fused_unified_rope_offset_pytorch(x, cos, sin, rotary_dim)

    # ------------------------------------------------------------------
    # copy_weights (same as before, kept intact)
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        with torch.no_grad():
            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_w = src.c_attn.weight.detach()
                c_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_w.shape[0] == self.hidden_size and c_w.shape[1] != self.hidden_size:
                    c_w = c_w.t().contiguous()
                q_dim, k_dim, v_dim = self.q_head_size, self.kv_head_size, self.kv_head_size
                w_q = c_w[:q_dim, :]
                w_k = c_w[q_dim:q_dim + k_dim, :]
                w_v = c_w[q_dim + k_dim:q_dim + k_dim + v_dim, :]
                if c_b is not None:
                    b_q = c_b[:q_dim]
                    b_k = c_b[q_dim:q_dim + k_dim]
                    b_v = c_b[q_dim + k_dim:q_dim + k_dim + v_dim]
                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: BERT / RoBERTa
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
                elif w_o is None:
                    for attr in ("out_proj", "out_lin", "dense", "c_proj", "wo", "proj", "output"):
                        if hasattr(src, attr):
                            pl = getattr(src, attr)
                            if hasattr(pl, "dense"):
                                pl = pl.dense
                            if hasattr(pl, "weight"):
                                w_o = pl.weight.detach()
                                b_o = pl.bias.detach() if pl.bias is not None else None
                                break

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

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None or w_o is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None: w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None: w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None: w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None: w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Validation
            missing = []
            if w_q is None: missing.append("Q")
            if w_k is None: missing.append("K")
            if w_v is None: missing.append("V")
            if w_o is None and not self.is_nested_encoder: missing.append("Out")
            if missing:
                raise RuntimeError(
                    f"UnifiedRoPE: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            def _copy_layer(target_layer, w, b):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().contiguous().shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.t().contiguous().to(target_layer.weight.dtype))
                    else:
                        m0 = min(target_layer.weight.shape[0], w.shape[0])
                        m1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.zero_()
                        target_layer.weight[:m0, :m1].copy_(w[:m0, :m1].to(target_layer.weight.dtype))
                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.copy_(b.to(target_layer.bias.dtype))
                    else:
                        mb = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.zero_()
                        target_layer.bias[:mb].copy_(b[:mb].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.zero_()

            _copy_layer(self.q_proj, w_q, b_q)
            _copy_layer(self.k_proj, w_k, b_k)
            _copy_layer(self.v_proj, w_v, b_v)
            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o)

    # ------------------------------------------------------------------
    # FIX-1 + FIX-6: forward with KV-cache and proper return format
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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        B, L, _ = hidden_states.shape
        device = hidden_states.device
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]
        layer_idx = int(kwargs.get("layer_idx", 0))

        # ---- Projections ----
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k_new = self.k_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v_new = self.v_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # ---- FIX-1: Read past KV cache ----
        past_k = past_v = None
        past_len = 0
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]
            if past_k is not None:
                past_len = past_k.shape[-2]

        # ---- FIX-3: RoPE with past_len offset ----
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb
            # Caller is responsible for slicing; assume correct length
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
        else:
            q = self._apply_rope(q, L, past_len=0)          # q is always current positions
            k_new = self._apply_rope(k_new, N_kv_new, past_len=past_len)  # k continues from past

        # ---- FIX-1: Concatenate KV cache ----
        present_kv: Any | None = None
        if is_cross:
            K_cat, V_cat = k_new, v_new
            present_kv = past_key_value
        else:
            if past_k is not None and past_v is not None:
                K_cat = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
                V_cat = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
            else:
                K_cat, V_cat = k_new, v_new

            use_cache = kwargs.get("use_cache", False) or past_key_value is not None
            if use_cache:
                if past_key_value is not None and hasattr(past_key_value, "update"):
                    try:
                        past_key_value.update(k_new, v_new, layer_idx)
                        present_kv = past_key_value
                    except Exception:
                        present_kv = (K_cat, V_cat)
                else:
                    present_kv = (K_cat, V_cat)

        N_kv = K_cat.shape[-2]

        # ---- GQA expand ----
        if self.num_key_value_groups > 1:
            K_full = repeat_kv(K_cat, self.num_key_value_groups)
            V_full = repeat_kv(V_cat, self.num_key_value_groups)
        else:
            K_full, V_full = K_cat, V_cat

        # ---- FIX-4: SDPA fast-path with bool mask ----
        is_causal = bool(self.is_decoder and not is_cross and L > 1)
        attn_out: torch.Tensor | None = None
        attn_weights_out: torch.Tensor | None = None

        use_sdpa = (not output_attentions) and (head_mask is None)
        if use_sdpa:
            try:
                # Build bool mask
                bool_mask = None
                if is_causal:
                    # Bottom-right aligned causal mask
                    q_idx = torch.arange(L, device=device).unsqueeze(1) + (N_kv - L)
                    k_idx = torch.arange(N_kv, device=device).unsqueeze(0)
                    causal = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)  # (1, 1, L, N_kv)
                    bool_mask = causal
                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        pad = (attention_mask > 0).bool()
                        pad = pad[:, None, None, :]
                    elif attention_mask.dim() == 4:
                        pad = (attention_mask > 0).bool() if attention_mask.dtype != torch.bool else attention_mask
                    else:
                        pad = None
                    if pad is not None:
                        bool_mask = pad if bool_mask is None else (bool_mask & pad)

                q_s = maybe_half(q)
                k_s = maybe_half(K_full)
                v_s = maybe_half(V_full)

                out_sdp = F.scaled_dot_product_attention(
                    q_s, k_s, v_s,
                    attn_mask=bool_mask,
                    is_causal=False,  # Already handled in mask
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = out_sdp.to(q.dtype)
            except Exception as e:
                logger.debug("UnifiedRoPE SDPA failed, falling back: %s", e)
                attn_out = None

        # ---- Manual path (fallback or when output_attentions/head_mask) ----
        if attn_out is None:
            scores = torch.matmul(q.float(), K_full.float().transpose(-2, -1).contiguous()) * self.scale

            if is_causal:
                q_idx = torch.arange(L, device=device).unsqueeze(1) + (N_kv - L)
                k_idx = torch.arange(N_kv, device=device).unsqueeze(0)
                causal = (k_idx <= q_idx).unsqueeze(0).unsqueeze(1)
                scores = scores.masked_fill(~causal, -1e4)

            if attention_mask is not None:
                scores = apply_attention_mask(scores, attention_mask)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores, dim=-1).to(q.dtype)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask[:, :, :1].unsqueeze(-1)
                attn = attn * head_mask.to(attn.dtype)

            attn_out = torch.matmul(attn, V_full)
            attn_weights_out = attn if output_attentions else None

        # ---- Output projection ----
        context = attn_out.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context if context.shape[-1] == self.hidden_size else self.out_proj(context)
        else:
            output = self.out_proj(context)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        # FIX-6: Correct return format
        if self.is_decoder and not is_cross:
            if output_attentions:
                return output.to(hidden_states.dtype), present_kv, attn_weights_out
            return output.to(hidden_states.dtype), present_kv

        if output_attentions:
            return output.to(hidden_states.dtype), attn_weights_out
        return output.to(hidden_states.dtype), None


# ═══════════════════════════════════════════════════════════════════
# FIX-2: Injector registered in ATTENTION_REGISTRY
# ═══════════════════════════════════════════════════════════════════
def inject_unified_rope(
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
        logger.warning("inject_unified_rope: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
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
            res = inject_unified_rope(
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

    # ---- Single layer ----
    try:
        def _already_injected(m):
            if m is None:
                return False
            if isinstance(m, UnifiedRoPE):
                return True
            if "UnifiedRoPE" in m.__class__.__name__:
                return True
            inner = getattr(m, "inner", None)
            if isinstance(inner, UnifiedRoPE):
                return True
            wrapped = getattr(m, "_wrapped_attention", None)
            if isinstance(wrapped, UnifiedRoPE):
                return True
            return False

        if _already_injected(old_attn):
            return old_attn

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
        if _already_injected(src):
            return src

        if model_config is None:
            model_config = getattr(target, "config", None) or getattr(src, "config", None)

        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_attn = UnifiedRoPE(
            config=model_config,
            comp_config=comp_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        # Place on correct device/dtype
        try:
            ref = next(src.parameters(), None)
            if ref is not None:
                new_attn.to(device=ref.device, dtype=ref.dtype)
        except Exception:
            pass

        # Splice
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True

        if not spliced:
            logger.error("inject_unified_rope: no writable attention attribute on %s", type(target).__name__)
            return None

        return new_attn
    except Exception as e:
        logger.exception("inject_unified_rope failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "UnifiedRoPE",
    "LoRALinear",
    "inject_unified_rope",
    "apply_rotary_pos_emb",
]
