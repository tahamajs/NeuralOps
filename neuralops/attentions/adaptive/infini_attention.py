"""
Jamba-2 Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block.
"""

from __future__ import annotations
from neuralops.core.cache_utils import read_cache as _read_cache
from neuralops.core.common import _detect_return_format
from neuralops.core.common import _detect_decoder

from neuralops.core.attention_utils import (
    safe_mask_value as _safe_mask_value,
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.expert_choice_moe import ExpertChoiceMoE
from neuralops.attentions.simple_ssm import SimpleSSM
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)




# ===========================================================================
# SSM Sub-branch
# ===========================================================================
class SVDMambaSub(nn.Module):
    """
    SVD-Mamba sub-branch.

    [JAMBA-FIX-01] `out_proj` uses NORMAL init (not zeros).
    The ReZero scaling is handled OUTSIDE by `mamba_scale=0`.
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        d_model: int = 768,
        config: Any | None = None,
        rank: int | None = None,
        is_decoder: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.ssm = SimpleSSM(d_model=d_model, config=config, is_decoder=is_decoder)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # ---- [JAMBA-FIX-01] NORMAL init, NOT zeros ----
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        self._ssm_supports_state = self._ssm_accepts_past_state()

    def _ssm_accepts_past_state(self) -> bool:
        import inspect

        try:
            sig = inspect.signature(self.ssm.forward)
            return "past_state" in sig.parameters or "state" in sig.parameters
        except (ValueError, TypeError):
            return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_state: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any | None]:
        x_proj = self.in_proj(hidden_states)
        ssm_result: Any = None
        present_state: Any | None = None

        if self._ssm_supports_state:
            try:
                ssm_result = self.ssm(
                    x_proj, attention_mask=attention_mask, past_state=past_state
                )
            except TypeError:
                try:
                    ssm_result = self.ssm(
                        x_proj, attention_mask=attention_mask, state=past_state
                    )
                except Exception:
                    ssm_result = self.ssm(x_proj, attention_mask=attention_mask)
        else:
            ssm_result = self.ssm(x_proj, attention_mask=attention_mask)

        if isinstance(ssm_result, tuple):
            ssm_out = ssm_result[0]
            if len(ssm_result) > 1:
                present_state = ssm_result[1]
        else:
            ssm_out = ssm_result

        out = self.out_proj(ssm_out)
        return out, present_state


def _write_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    mamba_state: Any | None,
    past_key_value: Any,
    layer_idx: int,
) -> Any:
    """[JAMBA-FIX-07] Robust cache write across formats."""
    if past_key_value is not None and hasattr(past_key_value, "key_cache"):
        if layer_idx < len(past_key_value.key_cache):
            past_key_value.key_cache[layer_idx] = k
            past_key_value.value_cache[layer_idx] = v
        else:
            past_key_value.key_cache.append(k)
            past_key_value.value_cache.append(v)
        if not hasattr(past_key_value, "mamba_state"):
            past_key_value.mamba_state = []
        if layer_idx < len(past_key_value.mamba_state):
            past_key_value.mamba_state[layer_idx] = mamba_state
        else:
            past_key_value.mamba_state.append(mamba_state)
        return past_key_value

    if mamba_state is not None:
        return (k, v, mamba_state)
    return (k, v)


# ===========================================================================
# Jamba-2 Block
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "jamba",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class Jamba2Block(BaseAttention):
    """
    Jamba-2 hybrid block:
        * Pre-LN Mamba branch (ReZero-scaled)
        * Multi-Head Attention branch
        * Pre-LN MoE branch (ReZero-scaled)

    Returns the SUM of all three deltas; the outer block performs the residual add.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        d_model: int | None = None,
        n_heads: int | None = None,
        n_experts: int = 4,
        top_k: int = 2,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        self.config = config

        # ---- Dimensions ----
        d_model = (
            kwargs.get("d_model")
            or kwargs.get("hidden_size")
            or kwargs.get("dim")
            or d_model
        )
        n_heads = (
            kwargs.get("n_heads")
            or kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or n_heads
        )

        if d_model is None and config is not None:
            d_model = getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "dim", 768)),
            )
        if d_model is None and source_attn is not None:
            for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                layer = getattr(source_attn, attr, None)
                if layer is not None and hasattr(layer, "out_features"):
                    d_model = layer.out_features
                    break
        if d_model is None:
            d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if n_heads is None and config is not None:
            n_heads = getattr(
                config,
                "num_attention_heads",
                getattr(config, "n_heads", 12),
            )
        if n_heads is None and source_attn is not None:
            n_heads = getattr(
                source_attn,
                "num_attention_heads",
                getattr(source_attn, "n_heads", 12),
            )
        if n_heads is None:
            n_heads = 12

        if config is not None:
            n_experts = int(
                getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            )
            top_k = int(getattr(config, "top_k", top_k))

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model
        self.n_heads = int(n_heads)
        self.num_heads = self.n_heads

        # ---- GQA ----
        num_kv_heads = None
        if config is not None:
            num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                getattr(config, "num_kv_heads", None),
            )
        if num_kv_heads is None:
            num_kv_heads = kwargs.get(
                "num_kv_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
        self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))

        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        # ---- Decoder detection ----
        self.is_decoder = _detect_decoder(
            config, kwargs.get("is_decoder"), source_attn, kwargs
        )
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self._return_format: str = "llama"

        # ---- LoRA config ----
        if config is not None:
            self.lora_rank = int(
                getattr(
                    config,
                    "lora_rank",
                    getattr(
                        comp_config, "lora_rank", kwargs.get("lora_rank", 16)
                    ),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config,
                    "lora_alpha",
                    getattr(
                        comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)
                    ),
                )
            )
        elif comp_config is not None:
            self.lora_rank = int(
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            )
            self.lora_alpha = float(
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
            )
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = self.lora_rank > 0

        self.source_attn = source_attn
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None)
            if source_attn is not None
            else None
        )

        # ---- Projections ----
        self.q_lin = LoRALinear(
            self.d_model,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_lin = LoRALinear(
            self.d_model,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_lin = LoRALinear(
            self.d_model,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_lin = LoRALinear(
            self.all_head_size,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin

        # ---- Mamba sub-branch ----
        self.mamba = SVDMambaSub(
            self.d_model, config=config, rank=rank, is_decoder=self.is_decoder
        )
        # ---- MoE sub-branch ----
        try:
            self.moe = ExpertChoiceMoE(
                self.d_model, n_experts=n_experts, capacity_factor=1.0
            )
        except Exception as e:
            logger.warning("Jamba2: MoE construction failed (%s); disabling.", e)
            self.moe = None

        # ---- Pre-LayerNorms ----
        self.norm1 = nn.LayerNorm(self.d_model)  # Mamba input norm
        self.norm3 = nn.LayerNorm(self.d_model)  # MoE input norm

        # ---- [JAMBA-FIX-01] ReZero with correct init ----
        # Only the SCALE parameters are zero.
        # out_proj inside mamba/moe use normal init so gradients flow.
        self.mamba_scale = nn.Parameter(torch.zeros(1))
        self.moe_scale = nn.Parameter(torch.zeros(1))

        dropout_p = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1)
                if config
                else 0.1,
            )
        )
        self.dropout = nn.Dropout(dropout_p)

        self._aux_loss_sum: float = 0.0
        self._aux_loss_count: int = 0

        if source_attn is not None:
            self.copy_weights(source_attn)
        self._return_format = _detect_return_format(
            source_attn, self.is_nested_encoder
        )

    # ------------------------------------------------------------------
    # Aux loss bookkeeping
    # ------------------------------------------------------------------
    @property
    def aux_loss(self) -> float:
        if self._aux_loss_count == 0:
            return 0.0
        return self._aux_loss_sum / self._aux_loss_count

    def reset_aux_loss(self) -> None:
        self._aux_loss_sum = 0.0
        self._aux_loss_count = 0

    def freeze_base_weights(self) -> None:
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(
            hasattr(src, a)
            for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj")
        )
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
            hasattr(src, "query") and not has_out
        ):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # ---- GPT-2 fused c_attn ----
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                if (
                    c.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c, "nf")
                    or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])
                ):
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                cb = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = cb.data.clone() if cb is not None else None
                extracted["q"] = (
                    cw[:split],
                    cb[:split] if cb is not None else None,
                )
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb[split : 2 * split] if cb is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb[2 * split :] if cb is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o = src.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    if (
                        o.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or hasattr(o, "nf")
                        or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                    ):
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (
                        ow,
                        bo.data.clone() if bo is not None else None,
                    )

        # ---- BERT / RoBERTa nested ----
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (
                        m.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # ---- Flat projections ----
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            w_data = m.weight.data.clone()
                            if (
                                m.__class__.__name__ == "Conv1D"
                                or hasattr(m, "nf")
                                or (
                                    key == "o"
                                    and w_data.ndim == 2
                                    and w_data.shape[0] < w_data.shape[1]
                                )
                            ):
                                w_data = w_data.t().contiguous()
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                w_data,
                                b.data.clone() if b is not None else None,
                            )
                            break

        # ---- Fallback ----
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
                            w.clone(),
                            b.clone() if b is not None else None,
                        )
            except Exception as e:
                logger.debug("Jamba2 UniversalAdapter fallback: %s", e)

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
        }

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target.weight.shape
                    and w.t().contiguous().shape == target.weight.shape
                ):
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        target.weight.device, target.weight.dtype
                    )
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                if self.freeze_base:
                    target.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb
        if self.freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _norm_dtype(self, norm: nn.LayerNorm, fallback: torch.dtype) -> torch.dtype:
        return norm.weight.dtype if getattr(norm, "weight", None) is not None else fallback

    def _format_output(
        self,
        out: torch.Tensor,
        attn_weights: torch.Tensor | None,
        present: Any | None,
    ) -> tuple:
        fmt = self._return_format
        if fmt == "bert":
            return (out, attn_weights) if attn_weights is not None else (out,)
        if fmt == "gpt2":
            return (
                (out, present, attn_weights)
                if attn_weights is not None
                else (out, present)
            )
        return (
            (out, attn_weights, present)
            if attn_weights is not None
            else (out, present)
        )

    def _normalize_head_mask(
        self, head_mask: torch.Tensor, attn: torch.Tensor
    ) -> torch.Tensor:
        hm = head_mask.to(attn.dtype)
        if hm.dim() == 1:
            hm = hm.view(1, self.num_heads, 1, 1)
        elif hm.dim() == 2:
            if hm.shape[-1] == self.num_heads:
                hm = hm.view(hm.shape[0], self.num_heads, 1, 1)
            else:
                hm = hm.view(1, hm.shape[-1], 1, 1)
        elif hm.dim() == 3:
            hm = hm.unsqueeze(2)
        return attn * hm

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
        *args: Any,
        **kwargs: Any,
    ):
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, _ = hidden_states.shape
        layer_idx = int(kwargs.get("layer_idx", 0))
        position_embeddings = kwargs.get("position_embeddings")
        target_dtype = hidden_states.dtype

        past_k, past_v, past_len, past_mamba = _read_cache(past_key_value, layer_idx)

        # =========================================================
        # Branch 1: Mamba (Pre-LN + ReZero)
        # =========================================================
        mamba_out = 0
        present_mamba = None
        if self.mamba is not None:
            n1d = self._norm_dtype(self.norm1, target_dtype)
            x_norm1 = self.norm1(hidden_states.to(n1d)).to(target_dtype)
            m_out, present_mamba = self.mamba(
                x_norm1, attention_mask=attention_mask, past_state=past_mamba
            )
            # [JAMBA-FIX-01] ReZero scaling by mamba_scale (=0 initially).
            # out_proj is normally initialized inside SVDMambaSub so
            # gradient on mamba_scale is m_out (nonzero) → learning starts.
            mamba_out = m_out.to(target_dtype) * self.mamba_scale

        # =========================================================
        # Branch 2: Multi-Head Attention
        # =========================================================
        q = (
            self.q_lin(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
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

        # ---- [JAMBA-FIX-05] RoPE applied only to NEW tokens ----
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.shape[-2] >= past_len + L:
                cos = cos.narrow(-2, past_len, L)
                sin = sin.narrow(-2, past_len, L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif getattr(self, "rotary_emb", None) is not None:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = (
                    torch.arange(past_len, past_len + L, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            cos = sin = None
            for call in (
                lambda: self.rotary_emb(k, seq_len=L),
                lambda: self.rotary_emb(k, pos_ids),
                lambda: self.rotary_emb(pos_ids),
            ):
                try:
                    cos, sin = call()
                    break
                except Exception:
                    continue
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # ---- Concat with past KV ----
        if past_k is not None:
            k = torch.cat([past_k.to(k.dtype), k], dim=-2)
            v = torch.cat([past_v.to(v.dtype), v], dim=-2)

        present_key_value = _write_cache(k, v, present_mamba, past_key_value, layer_idx)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k_exp = repeat_kv(k, self.kv_repeat)
            v_exp = repeat_kv(v, self.kv_repeat)
        else:
            k_exp, v_exp = k, v

        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = bool(self.is_decoder and L > 1)
        mask_val = _safe_mask_value(target_dtype)

        # ---- [JAMBA-FIX-02] SDPA can now work with attention_mask ----
        # ---- [JAMBA-FIX-03] is_causal and attn_mask are mutually exclusive ----
        use_sdpa = (head_mask is None) and (not output_attentions)

        attn_weights: torch.Tensor | None = None
        attn_out: torch.Tensor | None = None

        if use_sdpa:
            # Build an explicit additive mask when needed
            attn_bias: torch.Tensor | None = None

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    valid = get_token_valid_mask(
                        attention_mask, B, k_exp.shape[-2], hidden_states.device
                    )
                    if valid is not None and not valid.all():
                        attn_bias = torch.zeros(
                            (B, 1, L, k_exp.shape[-2]),
                            device=q.device,
                            dtype=q.dtype,
                        ).masked_fill(
                            ~valid.unsqueeze(1).unsqueeze(2), mask_val
                        )
                elif attention_mask.dim() == 3:
                    attn_bias = attention_mask.unsqueeze(1).to(q.dtype)
                elif attention_mask.dim() == 4:
                    if attention_mask.dtype == torch.bool:
                        attn_bias = torch.zeros_like(
                            attention_mask, dtype=q.dtype
                        ).masked_fill(~attention_mask, mask_val)
                    else:
                        attn_bias = attention_mask.to(q.dtype)

            try:
                _qs = q.to(target_dtype)
                _ks = k_exp.to(target_dtype)
                _vs = v_exp.to(target_dtype)

                # Combine causal into attn_bias when present
                if is_causal:
                    if attn_bias is None:
                        # Pure causal → use is_causal=True (fastest)
                        _sdpa_out = F.scaled_dot_product_attention(
                            _qs,
                            _ks,
                            _vs,
                            is_causal=True,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
                    else:
                        # Add causal to the existing bias
                        kv_len = k_exp.shape[-2]
                        causal_bool = torch.tril(
                            torch.ones(
                                L,
                                kv_len,
                                device=q.device,
                                dtype=torch.bool,
                            ),
                            diagonal=past_len,
                        )
                        attn_bias = attn_bias.masked_fill(
                            ~causal_bool.unsqueeze(0).unsqueeze(0), mask_val
                        )
                        _sdpa_out = F.scaled_dot_product_attention(
                            _qs,
                            _ks,
                            _vs,
                            attn_mask=attn_bias,
                            is_causal=False,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
                else:
                    if attn_bias is not None:
                        _sdpa_out = F.scaled_dot_product_attention(
                            _qs,
                            _ks,
                            _vs,
                            attn_mask=attn_bias,
                            is_causal=False,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )
                    else:
                        _sdpa_out = F.scaled_dot_product_attention(
                            _qs,
                            _ks,
                            _vs,
                            dropout_p=self.dropout.p if self.training else 0.0,
                        )

                attn_out = (
                    _sdpa_out.to(target_dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.all_head_size)
                )
            except Exception as e:
                logger.debug("Jamba2 SDPA path failed (%s); falling back.", e)
                attn_out = None

        # ---- Manual path ----
        if attn_out is None:
            scores = (
                torch.matmul(q.float(), k_exp.float().transpose(-2, -1).contiguous())
                * scale
            )

            # [JAMBA-FIX-04] Causal with diagonal=past_len (KV-cache safe)
            if is_causal:
                kv_len = k_exp.shape[-2]
                causal = torch.tril(
                    torch.ones(L, kv_len, device=hidden_states.device, dtype=torch.bool),
                    diagonal=past_len,
                )
                scores = scores.masked_fill(
                    ~causal.unsqueeze(0).unsqueeze(0), mask_val
                )

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    valid = get_token_valid_mask(
                        attention_mask, B, k_exp.shape[-2], hidden_states.device
                    )
                    if valid is not None:
                        scores = scores.masked_fill(
                            ~valid.unsqueeze(1).unsqueeze(2), mask_val
                        )
                elif attention_mask.dim() == 3:
                    scores = scores + attention_mask.unsqueeze(1).float()
                elif attention_mask.dim() == 4:
                    if attention_mask.dtype == torch.bool:
                        scores = scores.masked_fill(~attention_mask, mask_val)
                    else:
                        scores = scores + attention_mask.float()

            # ---- Safe softmax ----
            m = scores.amax(dim=-1, keepdim=True)
            m_safe = torch.where(m < -1e3, torch.zeros_like(m), m)
            exp_s = torch.exp((scores - m_safe).clamp(max=0.0))
            probs = (
                exp_s / exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            ).to(v_exp.dtype)

            probs = self.dropout(probs)
            if head_mask is not None:
                probs = self._normalize_head_mask(head_mask, probs)

            attn_out = (
                torch.matmul(probs, v_exp)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.all_head_size)
            )
            attn_weights = probs if output_attentions else None

        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        attn_delta = attn_out if self.is_nested_encoder else self.out_lin(attn_out)

        # =========================================================
        # Branch 3: MoE (Pre-LN + ReZero)
        # =========================================================
        moe_out = 0
        if self.moe is not None:
            n3d = self._norm_dtype(self.norm3, target_dtype)
            x_norm3 = self.norm3(hidden_states.to(n3d)).to(target_dtype)
            moe_res = self.moe(x_norm3)
            if isinstance(moe_res, (tuple, list)):
                moe_res = moe_res[0]
            moe_out = moe_res.to(target_dtype) * self.moe_scale  # ReZero
            self._accumulate_aux_loss()

        # ---- [JAMBA-FIX-06] Parallel deltas only; outer block adds residual ----
        final_delta = mamba_out + attn_delta + moe_out

        return self._format_output(final_delta, attn_weights, present_key_value)

    def _accumulate_aux_loss(self) -> None:
        aux = getattr(self.moe, "aux_loss", 0.0)
        v = float(aux.item()) if torch.is_tensor(aux) else float(aux)
        self._aux_loss_sum += v
        self._aux_loss_count += 1

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p: Any) -> int:
            return (
                p.numel() * p.element_size()
                if isinstance(p, (nn.Parameter, torch.Tensor))
                else 0
            )

        out: dict[str, int] = {}
        for name, m in (
            ("q_lin", self.q_lin),
            ("k_lin", self.k_lin),
            ("v_lin", self.v_lin),
            ("out_lin", self.out_lin),
            ("mamba", self.mamba),
            ("moe", self.moe),
        ):
            out[name] = sum(_sz(p) for p in m.parameters()) if m is not None else 0
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
def inject_jamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Jamba2Block | nn.Module | None:
    # [JAMBA-FIX-09] Recursion guard
    if _depth > 8:
        logger.warning("inject_jamba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
            res = inject_jamba(
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

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    # [JAMBA-FIX-09] Idempotency guard
    if isinstance(src_attn, Jamba2Block) or "Jamba2" in src_attn.__class__.__name__:
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        if isinstance(getattr(target, attr, None), Jamba2Block):
            return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or getattr(getattr(src_attn, "q_proj", None), "in_features", None)
        or getattr(getattr(src_attn, "q_lin", None), "in_features", None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        # Filter duplicated kwargs
        for k in (
            "d_model", "hidden_size", "num_heads", "num_attention_heads", "old_attn"
        ):
            kwargs.pop(k, None)

        new_block = Jamba2Block(
            d_model=hidden_size,
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn if src_attn is not target else None,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_block.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.exception("Jamba2Block injection failed: %s", e)
        return None

    # ---- Splice ----
    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
            spliced = True
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
            spliced = True
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
            spliced = True
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
            spliced = True

        if not spliced:
            logger.error(
                "inject_jamba: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        new_block._return_format = _detect_return_format(
            src_attn, new_block.is_nested_encoder
        )

    return new_block


__all__ = [
    "Jamba2Block",
    "SVDMambaSub",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_jamba",
]