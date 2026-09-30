"""Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer.

Implements bidirectional/causal element-wise linear recurrence with dynamic input-dependent
log-space forget gating, input modulation, LoRA adapters, and multi-architecture pretrained weight transfer.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
try:
    from framework.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
    HAS_TRITON_HGRN2 = True
except Exception as _e:
    logging.getLogger(__name__).warning("Triton hgrn2_scan unavailable: %s", _e)
    hgrn2_scan_triton = None
    HAS_TRITON_HGRN2 = False
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
        if "Conv1D" in cls_name or "Conv1D" in base_cls_name or hasattr(base, "nf"):
            if w.ndim == 2:
                w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


@ATTENTION_REGISTRY.register("hgrn2_layer")
@ATTENTION_REGISTRY.register("hgrn2")
class HGRN2Layer(BaseAttention):
    r"""Hierarchically Gated Recurrent Network v2 (HGRN-2) Layer.
    Supports causal mode for decoders, bidirectional mode for encoders, LoRA adapters,
    and pretrained weight copying from DistilBERT, RoBERTa, GPT-2, and Qwen3.5.

    NOTE: This is NOT a compression module. It replaces attention with an RNN.
    Parameter count is the same. Only use for speed improvements on
    long sequences (when parallel scan is implemented).
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        res_clip: float = 30.0,
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

        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
                or kwargs.get("is_decoder", False)
                or kwargs.get("is_causal", False)
            )

        self.is_nested_encoder: bool = False
        self.lora_rank = lora_rank if lora_rank is not None else 16
        self.lora_alpha = lora_alpha
        self.lora_scaling = (self.lora_alpha / max(1, self.lora_rank)) if self.lora_rank > 0 else 1.0

        self.forget_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.input_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.state_norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # Initialize gates with sensible defaults for long-term retention and unity gain
        nn.init.normal_(self.forget_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.forget_gate.bias, -2.5)
        nn.init.normal_(self.input_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.input_gate.bias, 0.0)
        nn.init.normal_(self.g_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.g_proj.bias, 1.28)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)

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
            for mod in (self.v_proj, self.out_proj):
                for p in mod.parameters():
                    p.requires_grad = False
            for mod in (self.forget_gate, self.input_gate, self.g_proj):
                for p in mod.parameters():
                    p.requires_grad = True

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn", None))
        if old_attn is not None:
            self.copy_weights(old_attn)
            target_dtype = kwargs.get("dtype", None)
            if target_dtype is None:
                p = next(old_attn.parameters(), None)
                if p is not None and isinstance(p.dtype, torch.dtype):
                    target_dtype = p.dtype
            if target_dtype is not None:
                self.to(target_dtype)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights across all 4 architectures."""
        w_q, b_q = None, None
        w_k, b_k = None, None
        w_v, b_v = None, None
        w_o, b_o = None, None

        src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
        is_nested = (
            src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            )
            or (hasattr(old_attn, "self") and hasattr(old_attn, "output"))
        )
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
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
                    tw.copy_(w.to(tw.dtype))
                else:
                    min_r = min(tw.shape[0], w.shape[0])
                    min_c = min(tw.shape[1], w.shape[1])
                    tw.data.zero_()
                    tw.data[:min_r, :min_c].copy_(w[:min_r, :min_c].to(tw.dtype))
                    logger.warning(
                        "Partial weight copy: source %s → target %s",
                        tuple(w.shape), tuple(tw.shape)
                    )
                if b is not None and target_linear.bias is not None:
                    tb = target_linear.bias
                    if b.shape == tb.shape:
                        tb.copy_(b.to(tb.dtype))
                    else:
                        min_b = min(tb.numel(), b.numel())
                        tb.data.zero_()
                        tb.data[:min_b].copy_(b[:min_b].to(tb.dtype))

        # Copy V -> v_proj and O -> out_proj across all architectures
        _copy_param(self.v_proj, w_v, b_v)
        _copy_param(self.out_proj, w_o, b_o)

        fnorm = float(torch.linalg.norm(self.v_proj.weight.detach().float()))
        if fnorm < 1e-4:
            logger.warning("HGRN2Layer v_proj weight norm is near zero: %g", fnorm)

        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj):
                for p in mod.parameters():
                    p.requires_grad = False
            for mod in (self.forget_gate, self.input_gate, self.g_proj):
                for p in mod.parameters():
                    p.requires_grad = True

    @property
    def query(self) -> nn.Module:
        return self.g_proj

    @property
    def key(self) -> nn.Module:
        return self.input_gate

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    def _scan_direction(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:

        if HAS_TRITON_HGRN2 and f.is_cuda and not torch.is_grad_enabled() is False:
        # Triton path: faster, supports fp16/bf16/fp32 uniformly
            return hgrn2_scan_triton(f.contiguous(), iv.contiguous(), res_clip=self.res_clip)

        """Parallel chunked log-space associative scan."""
        B, L, D = iv.shape
        f_fp32 = f.float()
        iv_fp32 = iv.float()
        log_f = torch.log(f_fp32.clamp(min=1e-6))

        h_chunks = []
        h_prev = torch.zeros(B, D, device=iv.device, dtype=torch.float32)

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            C = end - start
            f_c = log_f[:, start:end]
            iv_c = iv_fp32[:, start:end]
            A = torch.cumsum(f_c, dim=1)
            A_diff = A.unsqueeze(2) - A.unsqueeze(1)
            mask = torch.tril(torch.ones(C, C, device=iv.device, dtype=torch.bool))
            A_diff = torch.where(mask.unsqueeze(0).unsqueeze(-1), A_diff, torch.full_like(A_diff, -1e4))
            decay_weights = torch.exp(A_diff) * mask.unsqueeze(0).unsqueeze(-1)
            intra = torch.sum(decay_weights * iv_c.unsqueeze(1), dim=2)
            inter = h_prev.unsqueeze(1) * torch.exp(A)
            h_c = inter + intra
            if self.res_clip is not None and self.res_clip > 0:
                h_c = torch.clamp(h_c, -self.res_clip, self.res_clip)
            h_chunks.append(h_c)
            h_prev = h_c[:, -1]

        return torch.cat(h_chunks, dim=1).to(iv.dtype)

    def _recurrent_scan_parallel(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Memory-efficient parallel element-wise recurrence scan in FP32."""
        if mask is not None:
            if mask.dim() == 2:
                mask_float = mask.float()
            elif mask.dim() == 4:
                v = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                v = v.any(dim=1)
                if v.dim() == 3:
                    mask_float = v.any(dim=1).float()
                else:
                    mask_float = v.float()
            else:
                mask_float = mask.float()
            iv = iv * mask_float.unsqueeze(-1)

        if self.training:
            import torch.utils.checkpoint as cp
            h_fwd = cp.checkpoint(self._scan_direction, f, iv, use_reentrant=False)
        else:
            h_fwd = self._scan_direction(f, iv)

        if self.is_decoder:
            return h_fwd, None

        if HAS_TRITON_HGRN2 and f.is_cuda:
            h_fwd = self._scan_direction(f, iv)
        else:
            h_fwd = self._scan_direction(f, iv)  # fallback still uses checkpoint inside
        if self.is_decoder:
            return h_fwd, None
        # backward direction
        f_rev = torch.flip(f, dims=[1])
        iv_rev = torch.flip(iv, dims=[1])
        h_rev = self._scan_direction(f_rev, iv_rev)
        h_bwd = torch.flip(h_rev, dims=[1])
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

        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported in HGRN2Layer")

        B, L, D = hidden_states.shape

        target_dtype = hidden_states.dtype

        # 1. Compute gates and values with bounded log decay to prevent gate saturation
        raw_log_decay = -F.softplus(self.forget_gate(hidden_states))
        bounded_log_decay = raw_log_decay.clamp(min=-3.0, max=-1e-4)
        f = torch.exp(bounded_log_decay).to(target_dtype)

        # Input modulation: tie input scale with (1 - f) to prevent recurrence state sign-flipping
        i = torch.sigmoid(self.input_gate(hidden_states)).to(target_dtype)
        input_scale = ((1.0 - f) * i).to(target_dtype)
        v = self.v_proj(hidden_states).to(target_dtype)
        if self.lora_rank > 0 and self.v_lora_A is not None and self.v_lora_B is not None:
            v_lora = torch.matmul(
                torch.matmul(hidden_states, self.v_lora_A.t().contiguous().to(target_dtype)),
                self.v_lora_B.t().contiguous().to(target_dtype),
            ) * self.lora_scaling
            v = v + v_lora

        z = self.g_proj(hidden_states).to(target_dtype)
        iv = (input_scale * v).to(target_dtype)

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
            h_ctx = h_fwd.to(target_dtype)
        else:
            h_ctx = (h_fwd + h_bwd).to(target_dtype)

        # Context stabilization with token self-value preservation
        sn_dtype = self.state_norm.weight.dtype if hasattr(self.state_norm, 'weight') and self.state_norm.weight is not None else target_dtype
        h_norm = self.state_norm(h_ctx.to(sn_dtype)).to(target_dtype)
        gated = ((0.5 * v + 0.5 * h_norm) * F.silu(z)).to(target_dtype)
        out = self.out_proj(gated).to(target_dtype)
        if self.lora_rank > 0 and self.out_lora_A is not None and self.out_lora_B is not None:
            out_lora = torch.matmul(
                torch.matmul(gated, self.out_lora_A.t().contiguous().to(target_dtype)),
                self.out_lora_B.t().contiguous().to(target_dtype),
            ) * self.lora_scaling
            out = out + out_lora

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out.to(target_dtype), None


@ATTENTION_REGISTRY.register("hgrn2_layer_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_layer")
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
    is_nested_target = False
    if curr_attn is None and target is not None:
        if hasattr(target, "attention") and hasattr(target.attention, "self"):
            curr_attn = target.attention
            is_nested_target = True
        elif hasattr(target, "attention"):
            curr_attn = target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
            is_nested_target = True
        elif hasattr(target, "attn"):
            curr_attn = target.attn
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn

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
        if curr_attn is not None:
            p = next(curr_attn.parameters(), None)
            if p is not None:
                new_layer.to(device=p.device, dtype=p.dtype)
    except Exception as ex:
        logger.error("Failed to instantiate HGRN2Layer: %s", ex)
        return None

    if target is not None:
        if is_nested_target and hasattr(target, "attention") and hasattr(target.attention, "self"):
            target.attention.self = new_layer
            if hasattr(target.attention, "output") and hasattr(target.attention.output, "dense"):
                target.attention.output.dense = nn.Identity()
        elif hasattr(target, "attention"):
            target.attention = new_layer
        elif hasattr(target, "self"):
            target.self = new_layer
            if hasattr(target, "output") and hasattr(target.output, "dense"):
                target.output.dense = nn.Identity()
        elif hasattr(target, "attn"):
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer

    return new_layer
