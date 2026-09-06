"""Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer.

Implements bidirectional/causal element-wise linear recurrence with dynamic input-dependent
log-space forget gating, input modulation, LoRA adapters, and multi-architecture pretrained weight transfer.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _unpack_weight_bias(module: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        cls_name = getattr(module, "__class__", type(module)).__name__
        base_cls_name = getattr(base, "__class__", type(base)).__name__
        if "Conv1D" in cls_name or "Conv1D" in base_cls_name:
            if w.ndim == 2 and w.shape[0] < w.shape[1]:
                w = w.t()
    if b is not None:
        b = b.data.clone()
    return w, b


@ATTENTION_REGISTRY.register("hgrn2_layer")
@ATTENTION_REGISTRY.register("hgrn2")
class HGRN2Layer(BaseAttention):
    r"""Hierarchically Gated Recurrent Network v2 (HGRN-2) Layer.
    Supports causal mode for decoders, bidirectional mode for encoders, LoRA adapters,
    and pretrained weight copying from DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        res_clip: float = 5.0,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        freeze_base: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # Resolve d_model
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size")
            if d_model is None:
                if isinstance(config, int):
                    d_model = config
                elif config is not None:
                    d_model = _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", _cfg_get(config, "n_embd", _cfg_get(config, "dim", None))))
            if d_model is None:
                d_model = kwargs.get("dim", 128)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model: int = int(d_model)
        self.dim: int = self.d_model
        self.hidden_size: int = self.d_model
        self.res_clip: float = res_clip
        self.freeze_base: bool = freeze_base

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
                or kwargs.get("is_decoder", False)
            )

        self.is_nested_encoder: bool = False
        self.lora_rank = lora_rank if lora_rank is not None else 16
        self.lora_alpha = lora_alpha
        self.lora_scaling = (self.lora_alpha / max(1, self.lora_rank)) if self.lora_rank > 0 else 1.0

        # Base linear projections
        self.forget_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.input_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # Initialize forget gate bias to -3.0 so softplus(-3) ~ 0.048 => f_t ~ exp(-0.048) ~ 0.95
        nn.init.constant_(self.forget_gate.bias, -3.0)
        nn.init.constant_(self.input_gate.bias, 0.0)

        # LoRA Adapters
        if self.lora_rank > 0:
            self.v_lora_A = nn.Parameter(torch.randn(self.lora_rank, self.d_model) * 0.01)
            self.v_lora_B = nn.Parameter(torch.zeros(self.d_model, self.lora_rank))
            self.out_lora_A = nn.Parameter(torch.randn(self.lora_rank, self.d_model) * 0.01)
            self.out_lora_B = nn.Parameter(torch.zeros(self.d_model, self.lora_rank))
        else:
            self.register_parameter("v_lora_A", None)
            self.register_parameter("v_lora_B", None)
            self.register_parameter("out_lora_A", None)
            self.register_parameter("out_lora_B", None)

        if self.freeze_base:
            for mod in (self.forget_gate, self.input_gate, self.v_proj, self.g_proj, self.out_proj):
                for p in mod.parameters():
                    p.requires_grad = False

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn", None))
        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights across all 4 architectures."""
        w_q, b_q = None, None
        w_k, b_k = None, None
        w_v, b_v = None, None
        w_o, b_o = None, None

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
                w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
        else:
            src = old_attn

        # Pattern 0: PyTorch MultiheadAttention
        if hasattr(src, "in_proj_weight") and src.in_proj_weight is not None:
            in_w, in_b = _unpack_weight_bias(src)
            if in_w is None:
                in_w = getattr(src, "in_proj_weight", None)
                in_b = getattr(src, "in_proj_bias", None)
            if in_w is not None:
                chunks = in_w.chunk(3, dim=0)
                w_q, w_k, w_v = chunks[0], chunks[1], chunks[2]
            if in_b is not None:
                b_chunks = in_b.chunk(3, dim=0)
                b_q, b_k, b_v = b_chunks[0], b_chunks[1], b_chunks[2]
            if hasattr(src, "out_proj"):
                w_o, b_o = _unpack_weight_bias(src.out_proj)

        # Pattern 1: GPT-2 Fused c_attn
        elif hasattr(src, "c_attn") and hasattr(src, "c_proj"):
            c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
            if c_attn_w is not None:
                split_size = c_attn_w.shape[0] // 3
                w_q = c_attn_w[:split_size, :]
                w_k = c_attn_w[split_size : 2 * split_size, :]
                w_v = c_attn_w[2 * split_size :, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:split_size]
                    b_k = c_attn_b[split_size : 2 * split_size]
                    b_v = c_attn_b[2 * split_size :]
            if hasattr(src, "c_proj"):
                w_o, b_o = _unpack_weight_bias(src.c_proj)

        # Pattern 2: RoBERTa / BERT separate query, key, value
        elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
            w_q, b_q = _unpack_weight_bias(src.query)
            w_k, b_k = _unpack_weight_bias(src.key)
            w_v, b_v = _unpack_weight_bias(src.value)
            if w_o is None:
                for attr in ("dense", "out_proj", "o_proj"):
                    if hasattr(src, attr):
                        w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                        break

        # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
        elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
            w_q, b_q = _unpack_weight_bias(src.q_proj)
            w_k, b_k = _unpack_weight_bias(src.k_proj)
            w_v, b_v = _unpack_weight_bias(src.v_proj)
            if hasattr(src, "o_proj"):
                w_o, b_o = _unpack_weight_bias(src.o_proj)

        # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
        elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
            w_q, b_q = _unpack_weight_bias(src.q_lin)
            w_k, b_k = _unpack_weight_bias(src.k_lin)
            w_v, b_v = _unpack_weight_bias(src.v_lin)
            if hasattr(src, "out_lin"):
                w_o, b_o = _unpack_weight_bias(src.out_lin)

        # Fallback: UniversalAdapter
        if w_q is None or w_k is None or w_v is None:
            try:
                adapter = UniversalAdapter(old_attn)
                extracted = adapter.extract_qkv_out()
                if extracted:
                    w_q = extracted.get("q_weight")
                    b_q = extracted.get("q_bias")
                    w_k = extracted.get("k_weight")
                    b_k = extracted.get("k_bias")
                    w_v = extracted.get("v_weight")
                    b_v = extracted.get("v_bias")
                    if w_o is None:
                        w_o = extracted.get("out_weight")
                        b_o = extracted.get("out_bias")
            except Exception as ex:
                logger.debug("UniversalAdapter fallback failed: %s", ex)

        def _copy_param(target_linear: nn.Linear, w: Optional[torch.Tensor], b: Optional[torch.Tensor]):
            if w is None:
                return
            with torch.no_grad():
                tw = target_linear.weight
                if w.shape == tw.shape:
                    tw.copy_(w)
                elif w.shape[0] < tw.shape[0]:
                    tw[:w.shape[0], :w.shape[1]].copy_(w)
                elif w.shape[0] > tw.shape[0]:
                    tw.copy_(w[:tw.shape[0], :tw.shape[1]])
                if b is not None and target_linear.bias is not None:
                    tb = target_linear.bias
                    if b.shape == tb.shape:
                        tb.copy_(b)
                    elif b.shape[0] < tb.shape[0]:
                        tb[:b.shape[0]].copy_(b)
                    elif b.shape[0] > tb.shape[0]:
                        tb.copy_(b[:tb.shape[0]])

        # Map Q -> g_proj, K -> input_gate, V -> v_proj, O -> out_proj
        _copy_param(self.g_proj, w_q, b_q)
        _copy_param(self.input_gate, w_k, b_k)
        _copy_param(self.v_proj, w_v, b_v)
        if not self.is_nested_encoder:
            _copy_param(self.out_proj, w_o, b_o)

        if self.freeze_base:
            for mod in (self.forget_gate, self.input_gate, self.v_proj, self.g_proj, self.out_proj):
                for p in mod.parameters():
                    p.requires_grad = False

    def _recurrent_scan_parallel(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parallel element-wise associative scan in FP32."""
        B, L, D = iv.shape
        f_fp32 = f.float()
        iv_fp32 = iv.float()

        log_f = torch.log(f_fp32.clamp(min=1e-4, max=1.0 - 1e-4))  # [B, L, D]
        C = torch.cumsum(log_f, dim=1).permute(0, 2, 1)            # [B, D, L]
        C_diff = C.unsqueeze(-1) - C.unsqueeze(-2)                 # [B, D, L, L]

        # 1. Forward causal decay matrix
        tril_mask = torch.tril(torch.ones(L, L, device=iv.device, dtype=torch.bool))
        D_causal = torch.exp(C_diff.clamp(min=-40.0, max=0.0)) * tril_mask.float()  # [B, D, L, L]

        u = iv_fp32.permute(0, 2, 1).unsqueeze(-1)  # [B, D, L, 1]

        if mask is not None:
            mask_float = mask.float() if mask.dim() == 2 else mask[:, 0, 0, :].float()
            mask_2d = mask_float.unsqueeze(1).unsqueeze(-1) * mask_float.unsqueeze(1).unsqueeze(-2)
            D_causal = D_causal * mask_2d

        h_fwd = torch.matmul(D_causal, u).squeeze(-1).permute(0, 2, 1).to(iv.dtype)  # [B, L, D]
        h_fwd = torch.clamp(h_fwd, -self.res_clip, self.res_clip)

        if self.is_decoder:
            return h_fwd, None

        # 2. Backward anticausal decay matrix for bidirectional mode
        triu_mask = torch.triu(torch.ones(L, L, device=iv.device, dtype=torch.bool))
        D_anti = torch.exp((-C_diff).clamp(min=-40.0, max=0.0)) * triu_mask.float()  # [B, D, L, L]
        if mask is not None:
            D_anti = D_anti * mask_2d

        h_bwd = torch.matmul(D_anti, u).squeeze(-1).permute(0, 2, 1).to(iv.dtype)    # [B, L, D]
        h_bwd = torch.clamp(h_bwd, -self.res_clip, self.res_clip)

        return h_fwd, h_bwd

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        # 1. Compute gates and values
        f = torch.exp(-F.softplus(self.forget_gate(hidden_states)))
        i = torch.sigmoid(self.input_gate(hidden_states))
        v = self.v_proj(hidden_states)
        if self.lora_rank > 0 and self.v_lora_A is not None and self.v_lora_B is not None:
            v = v + torch.matmul(torch.matmul(hidden_states, self.v_lora_A.t()), self.v_lora_B.t()) * self.lora_scaling

        z = self.g_proj(hidden_states)
        iv = i * v

        # 2. Mask processing
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        # 3. Recurrent Scan
        h_fwd, h_bwd = self._recurrent_scan_parallel(f, iv, mask_bool)

        if self.is_decoder or h_bwd is None:
            h = h_fwd
        else:
            h = 0.5 * (h_fwd + h_bwd)

        # 4. Gating and output projection
        gated = h * F.silu(z)
        if self.is_nested_encoder:
            out = gated
        else:
            out = self.out_proj(gated)
            if self.lora_rank > 0 and self.out_lora_A is not None and self.out_lora_B is not None:
                out = out + torch.matmul(torch.matmul(gated, self.out_lora_A.t()), self.out_lora_B.t()) * self.lora_scaling

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("hgrn2_layer_inject")
@ATTENTION_REGISTRY.register("hgrn2_inject")
def inject_hgrn2_layer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    **kwargs: Any,
) -> Optional[Any]:
    """Safely constructs and injects HGRN2Layer into a model or transformer layer."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config", None)

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
                res = inject_hgrn2_layer(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # 2. Single layer injection
    curr_attn = old_attn
    if curr_attn is None and target is not None:
        if hasattr(target, "attention"):
            curr_attn = target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
        elif hasattr(target, "attn"):
            curr_attn = target.attn

    if curr_attn is None:
        return None

    try:
        new_layer = HGRN2Layer(
            config=cfg,
            old_attn=curr_attn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **kwargs,
        )
    except Exception as ex:
        logger.warning("Failed to instantiate HGRN2Layer: %s", ex)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            target.attention = new_layer
        elif hasattr(target, "self"):
            target.self = new_layer
        elif hasattr(target, "attn"):
            target.attn = new_layer

    return new_layer
