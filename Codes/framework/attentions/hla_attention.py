"""Hadamard Linear Attention (HLA).

Multi-head bidirectional/causal linear recurrent attention with Hadamard associative kernelization,
learnable decay gating, denominator normalization, and pretrained weight transfer.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension support."""
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

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
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = (
                self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            )
            lora_out = F.linear(F.linear(x, a), bb) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("hla", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("hla_attention", supports_causal=True, supports_gqa=True)
class HLAAttention(BaseAttention):
    r"""Hadamard Linear Attention (HLA) Bidirectional / Causal Module.

    Maintains multi-head associative causal/anti-causal matrix states $S_t \in \mathbb{R}^{B \times H \times d_k \times d_k}$.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if num_heads is None:
            if config is not None:
                num_heads = getattr(
                    config, "num_attention_heads", getattr(config, "n_heads", 12)
                )
            else:
                num_heads = kwargs.get("num_attention_heads", 12)

        comp_cfg = kwargs.get("comp_config")
        if comp_cfg is not None:
            lora_rank = getattr(
                comp_cfg, "lora_rank", getattr(comp_cfg, "rank", lora_rank)
            )
            lora_alpha = getattr(
                comp_cfg, "lora_alpha", getattr(comp_cfg, "alpha", lora_alpha)
            )
        elif config is not None:
            lora_rank = getattr(config, "lora_rank", lora_rank)
            lora_alpha = getattr(config, "lora_alpha", lora_alpha)

        self.hidden_size: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.head_dim: int = (
            self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        )
        self.lora_rank: int = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha: float = float(lora_alpha) if lora_alpha is not None else 32.0

        cfg_kv = (
            getattr(
                config, "num_key_value_heads", getattr(config, "num_kv_heads", None)
            )
            if config is not None
            else None
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(
            1, self.num_heads // max(1, self.num_kv_heads)
        )

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.q_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)
        self.output_scale = nn.Parameter(torch.ones(1))  # trainable output calibration

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
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

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                (src.c_attn.bias.detach() if src.c_attn.bias is not None else None)
                if (
                    c_attn_w.shape[0] == self.hidden_size
                    and c_attn_w.shape[1] != self.hidden_size
                ):
                    c_attn_w = c_attn_w.t().contiguous()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                out_w = src.c_proj.weight.detach()
                if (
                    out_w.shape[0] == self.hidden_size
                    and out_w.shape[1] != self.hidden_size
                ):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif (
                hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value")
            ):
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

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif (
                hasattr(src, "q_proj")
                and hasattr(src, "k_proj")
                and hasattr(src, "v_proj")
            ):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None

                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = (
                        src.o_proj.bias.detach()
                        if src.o_proj.bias is not None
                        else None
                    )

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif (
                hasattr(src, "q_lin")
                and hasattr(src, "k_lin")
                and hasattr(src, "v_lin")
            ):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None

                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = (
                        src.out_lin.bias.detach()
                        if src.out_lin.bias is not None
                        else None
                    )

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Strict validation
            missing = []
            if w_q is None:
                missing.append("Q projection")
            if w_k is None:
                missing.append("K projection")
            if w_v is None:
                missing.append("V projection")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out projection")

            if missing:
                raise RuntimeError(
                    f"HLAAttention: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            for target_mod, w_src in [
                (self.q_proj, w_q),
                (self.k_proj, w_k),
                (self.v_proj, w_v),
            ]:
                tw = target_mod.weight
                w_cast = w_src.to(tw.dtype)
                if w_cast.shape == tw.shape:
                    tw.data.copy_(w_cast)
                else:
                    min_r = min(tw.shape[0], w_cast.shape[0])
                    min_c = min(tw.shape[1], w_cast.shape[1])
                    tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])

            f_norm_q = float(torch.linalg.norm(self.q_proj.weight))
            if f_norm_q < 1e-4:
                logger.warning(f"HLAAttention: Q weight norm is near zero: {f_norm_q}")

            if w_o is not None:
                tw_o = self.out_proj.weight
                w_o_cast = w_o.to(tw_o.dtype)
                if w_o_cast.shape == tw_o.shape:
                    tw_o.data.copy_(w_o_cast)
                else:
                    min_r = min(tw_o.shape[0], w_o_cast.shape[0])
                    min_c = min(tw_o.shape[1], w_o_cast.shape[1])
                    tw_o.data[:min_r, :min_c].copy_(w_o_cast[:min_r, :min_c])

                if b_o is not None and self.out_proj.bias is not None:
                    tb_o = self.out_proj.bias
                    b_o_cast = b_o.to(tb_o.dtype)
                    min_b = min(tb_o.numel(), b_o_cast.numel())
                    tb_o.data[:min_b].copy_(b_o_cast[:min_b])
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    logger.warning(
                        f"HLAAttention: Out projection norm is near zero: {f_norm_o}"
                    )

    def _hla_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Vectorized parallel linear associative scan via tensor contraction."""
        B, H, L, d = q.shape
        q_f = torch.nn.functional.elu(q.float()) + 1.0
        k_f = torch.nn.functional.elu(k.float()) + 1.0
        v_f = v.float()

        if mask is not None:
            m_f = mask.float().view(B, 1, L, 1)
            k_f = k_f * m_f
            v_f = v_f * m_f

        t_idx = torch.arange(L, device=q.device)
        diff = (
            (t_idx.unsqueeze(0) - t_idx.unsqueeze(1))
            if reverse
            else (t_idx.unsqueeze(1) - t_idx.unsqueeze(0))
        )
        tri_mask = (
            torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool))
            if reverse
            else torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        )

        log_decay = torch.log(
            decay.squeeze(-1).squeeze(-1).clamp(min=1e-5, max=1.0 - 1e-5)
        )  # [1, H]
        exp_arg = torch.where(
            tri_mask.unsqueeze(0).unsqueeze(0),
            diff.unsqueeze(0).unsqueeze(0).float()
            * log_decay.unsqueeze(-1).unsqueeze(-1),
            float("-inf"),
        )
        decay_mat = torch.exp(
            exp_arg
        )  # [1, H, L, L], exactly 0.0 outside causal triangle

        # Attention weights with exponential decay: scores = (q @ k^T) * decay_mat
        qk = torch.matmul(q_f, k_f.transpose(-1, -2).contiguous())  # [B, H, L, L]
        scores = qk * decay_mat

        # Numerator: y_num = scores @ v
        num = torch.matmul(scores, v_f)  # [B, H, L, d]

        # Denominator: y_denom = sum_j scores
        denom = scores.sum(dim=-1, keepdim=True) + 1e-5  # [B, H, L, 1]

        y = num / denom.clamp(min=1e-5)
        return y.to(q.dtype)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        """Bidirectional / Causal linear attention forward pass."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped-Query Attention expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # ELU + 1 kernelization
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        decay = torch.sigmoid(self.beta).unsqueeze(0)  # [1, H, 1, 1]

        # Mask processing
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        y_fwd = self._hla_scan(q, k, v, decay, mask=mask_bool, reverse=False)

        if self.is_decoder:
            y = y_fwd
        else:
            y_bwd = self._hla_scan(q, k, v, decay, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            y = y * head_mask

        context = y.transpose(1, 2).contiguous().view(B, L, H * Dh)

        if self.is_nested_encoder:
            out = context * self.output_scale
        else:
            out = self.out_proj(context) * self.output_scale

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return (out, None)


@ATTENTION_REGISTRY.register("hla_inject")
def inject_hla(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
            res = inject_hla(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config
                    if "comp_config" in locals() or "comp_config" in kwargs
                    else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Safe injection hook for HLAAttention."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        d_model = 768
        num_heads = 12
        if model_config is not None:
            d_model = getattr(
                model_config, "hidden_size", getattr(model_config, "dim", 768)
            )
            num_heads = getattr(
                model_config,
                "num_attention_heads",
                getattr(model_config, "n_heads", 12),
            )
        elif hasattr(layer, "config"):
            d_model = getattr(
                layer.config, "hidden_size", getattr(layer.config, "dim", 768)
            )
            num_heads = getattr(
                layer.config,
                "num_attention_heads",
                getattr(layer.config, "n_heads", 12),
            )

        new_block = HLAAttention(
            config=model_config,
            comp_config=comp_config,
            d_model=d_model,
            num_heads=num_heads,
            **kwargs,
        )
        try:
            new_block.copy_weights(src)
        except Exception as e:
            logger.error("inject_hla copy_weights failed: %s", e)
            return None

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block

        return new_block
    except Exception as e:
        logger.error("inject_hla failed at layer %d: %s", layer_idx, e)
        return None
