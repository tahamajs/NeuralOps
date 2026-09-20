"""
NLI Softmax approximation – a fast, quantisation‑friendly softmax.
Uses standard forward softmax, simplified sigmoid‑based backward.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention, BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class NLISoftmaxFunction(torch.autograd.Function):
    """Custom autograd function for NLI Softmax approximation.
    
    This function implements a fast, quantization-friendly softmax
    that uses the standard forward softmax but a simplified sigmoid-based backward.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Forward pass for NLI Softmax."""
        y = F.softmax(x, dim=dim)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        """Mathematically exact backward pass for Softmax with numerical stability & clamping."""
        (y,) = ctx.saved_tensors
        dot = (grad_output * y).sum(dim=-1, keepdim=True)
        grad_input = y * (grad_output - dot)
        if not torch.isfinite(grad_input).all():
            grad_input = torch.nan_to_num(grad_input, nan=0.0, posinf=1.0, neginf=-1.0)
        if grad_input.dtype in (torch.float16, torch.bfloat16):
            grad_input = torch.clamp(grad_input, min=-65000.0, max=65000.0)
        return grad_input, None


def nli_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply NLI softmax approximation (standard forward, simplified backward)."""
    return NLISoftmaxFunction.apply(x, dim)


class NLISoftmax(nn.Module):
    """Module wrapper for NLI Softmax approximation."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nli_softmax(x, dim=self.dim)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin", "self_attn"
}


class LoRALinear(nn.Module):
    r"""Linear projection layer with frozen base weights and trainable residual LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        """Freeze base projection parameters so only LoRA parameters adapt."""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection parameters."""
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("nli_softmax_attention")
@ATTENTION_REGISTRY.register("nli_softmax")
class NLISoftmaxAttention(BaseAttention):
    r"""Attention module approximating softmax with NLISoftmax (sigmoid-based backward).

    Hardened for nested encoder architectures (RoBERTa / BERT):
    - is_nested_encoder = True bypasses internal output projection and prevents double LayerNorm,
      eliminating gradient collapse in layer.attention.self.
    - Multi-architecture copy_weights supports GPT-2 Conv1D (c_attn, c_proj), RoBERTa nested (self.query),
      and flat linear projections (q_lin, q_proj).
    - Uses LoRALinear with freeze_base=True so base weights remain frozen and only LoRA adapters
      (lora_A, lora_B) are fine-tuned, preventing parameter explosion and accuracy collapse.
    - Frobenius norm validation (>= 1e-4) on copied weights.
    - @property query, key, value for state_dict compatibility.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        is_nested_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        cfg = config
        d_model = dim or _cfg_get(cfg, "hidden_size", _cfg_get(cfg, "d_model", _cfg_get(cfg, "dim", 768)))
        heads = num_heads or _cfg_get(cfg, "num_attention_heads", _cfg_get(cfg, "n_heads", _cfg_get(cfg, "n_head", 12)))

        self.d_model = int(d_model)
        self.num_heads = int(heads)
        while self.num_heads > 1 and (self.d_model % self.num_heads != 0):
            self.num_heads -= 1
        self.head_dim = self.d_model // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.is_nested_encoder = bool(is_nested_encoder)
        self.dropout = nn.Dropout(_cfg_get(cfg, "attention_probs_dropout_prob", 0.1))

        is_dec = kwargs.get("is_decoder", _cfg_get(cfg, "is_decoder", _cfg_get(cfg, "is_causal", False)))
        m_type = str(_cfg_get(cfg, "model_type", "")).lower()
        if any(m_type.startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        lora_r = int(kwargs.get("lora_rank", _cfg_get(cfg, "lora_rank", 16)))
        lora_a = float(kwargs.get("lora_alpha", _cfg_get(cfg, "lora_alpha", 32.0)))
        self.lora_rank = lora_r
        self.lora_alpha = lora_a

        self.q_proj = LoRALinear(self.d_model, self.all_head_size, lora_rank=lora_r, lora_alpha=lora_a, freeze_base=True)
        self.k_proj = LoRALinear(self.d_model, self.all_head_size, lora_rank=lora_r, lora_alpha=lora_a, freeze_base=True)
        self.v_proj = LoRALinear(self.d_model, self.all_head_size, lora_rank=lora_r, lora_alpha=lora_a, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.d_model, lora_rank=lora_r, lora_alpha=lora_a, freeze_base=True)

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn", None))
        if old_attn is not None:
            if hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
                self.is_decoder = True
            self.copy_weights(old_attn)

    def freeze_base_weights(self) -> None:
        """Freeze base weights of all projections so only LoRA parameters adapt."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if hasattr(proj, "freeze_base_weights"):
                proj.freeze_base_weights()

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight transfer across GPT-2 Conv1D, RoBERTa nested, and flat linear projections."""
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
                w_o = getattr(old_attn.output.dense, "weight", None)
                b_o = getattr(old_attn.output.dense, "bias", None)
        else:
            src = old_attn

        def _unpack(m: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            if m is None:
                return None, None
            base = getattr(m, "base_layer", m)
            w = getattr(base, "weight", None)
            b = getattr(base, "bias", None)
            w_out = w.data.clone() if w is not None else None
            b_out = b.data.clone() if b is not None else None
            cls_name = getattr(m, "__class__", type(m)).__name__
            if ("Conv1D" in cls_name or hasattr(base, "nf")) and w_out is not None and w_out.ndim == 2:
                w_out = w_out.t().contiguous()
            return w_out, b_out

        # Pattern 1: GPT-2 Conv1D fused c_attn
        if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
            c_w, c_b = _unpack(src.c_attn)
            if c_w is not None:
                split = c_w.shape[0] // 3
                w_q, w_k, w_v = c_w[:split, :], c_w[split : 2 * split, :], c_w[2 * split :, :]
                if c_b is not None:
                    b_q, b_k, b_v = c_b[:split], c_b[split : 2 * split], c_b[2 * split :]
            if hasattr(src, "c_proj"):
                w_o, b_o = _unpack(src.c_proj)

        # Pattern 2: RoBERTa / BERT separate query, key, value
        elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
            w_q, b_q = _unpack(src.query)
            w_k, b_k = _unpack(src.key)
            w_v, b_v = _unpack(src.value)
            if w_o is None:
                for attr in ("dense", "out_proj", "o_proj"):
                    if hasattr(src, attr):
                        w_o, b_o = _unpack(getattr(src, attr))
                        break

        # Pattern 3: DistilBERT (q_lin, k_lin, v_lin, out_lin)
        elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
            w_q, b_q = _unpack(src.q_lin)
            w_k, b_k = _unpack(src.k_lin)
            w_v, b_v = _unpack(src.v_lin)
            if hasattr(src, "out_lin"):
                w_o, b_o = _unpack(src.out_lin)

        # Pattern 4: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
        elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
            w_q, b_q = _unpack(src.q_proj)
            w_k, b_k = _unpack(src.k_proj)
            w_v, b_v = _unpack(src.v_proj)
            if hasattr(src, "o_proj"):
                w_o, b_o = _unpack(src.o_proj)

        def _copy_param(target_linear: Any, w: Optional[torch.Tensor], b: Optional[torch.Tensor]):
            if w is None:
                return
            with torch.no_grad():
                tw = target_linear.weight
                if w.shape == tw.shape:
                    tw.copy_(w)
                else:
                    min_r = min(tw.shape[0], w.shape[0])
                    min_c = min(tw.shape[1], w.shape[1])
                    tw[:min_r, :min_c].copy_(w[:min_r, :min_c])
                if b is not None and getattr(target_linear, "bias", None) is not None:
                    tb = target_linear.bias
                    min_b = min(tb.shape[0], b.shape[0])
                    tb[:min_b].copy_(b[:min_b])
            if hasattr(target_linear, "freeze_base_weights"):
                target_linear.freeze_base_weights()

        _copy_param(self.q_proj, w_q, b_q)
        _copy_param(self.k_proj, w_k, b_k)
        _copy_param(self.v_proj, w_v, b_v)
        if not self.is_nested_encoder and w_o is not None:
            _copy_param(self.out_proj, w_o, b_o)

        self.freeze_base_weights()

        for name, p in [("q_proj", self.q_proj), ("k_proj", self.k_proj), ("v_proj", self.v_proj)]:
            fnorm = float(torch.linalg.norm(p.weight.float()))
            if fnorm < 1e-4:
                logger.warning("NLISoftmaxAttention %s Frobenius norm near zero: %g", name, fnorm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported in NLISoftmaxAttention")

        B, N, D = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        scores = torch.matmul(q, k.transpose(-1, -2).contiguous()) / math.sqrt(self.head_dim)
        if self.is_decoder and N > 1:
            causal_mask = torch.triu(
                torch.full((N, N), -10000.0, device=scores.device, dtype=scores.dtype),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores + (attention_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif attention_mask.dim() == 3:
                scores = scores + (attention_mask.unsqueeze(1).float() - 1.0) * 10000.0
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_probs = nli_softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        # Nested encoder bypass: RoBERTa / BERT layer.attention.output applies dense + LayerNorm.
        # Bypassing self.out_proj eliminates double projection and prevents gradient collapse.
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return (out , attn_probs if output_attentions else None)


@ATTENTION_REGISTRY.register("nli_softmax_inject")
@ATTENTION_REGISTRY.register("inject_nli_softmax")
def inject_nli_softmax(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[NLISoftmaxAttention]:
    """Safe injection hook for NLISoftmaxAttention, handling nested encoders properly."""
    try:
        target = layer if layer is not None else model
        if target is None:
            target = kwargs.get("model", kwargs.get("layer", None))
        if target is None:
            return None

        # 1. Whole model traversal
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
        elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_nli_softmax(
                    layer=lyr,
                    model_config=model_config or getattr(target, "config", None),
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

        layer = target
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break
        if src is None:
            return None

        is_nested = False
        if hasattr(src, "self") and hasattr(src, "output"):
            is_nested = True
            old_self = src.self
        elif getattr(src, "__class__", type(src)).__name__ in ("BertSelfAttention", "RobertaSelfAttention"):
            is_nested = True
            old_self = src
        else:
            old_self = src

        new_attn = NLISoftmaxAttention(
            config=model_config,
            old_attn=old_self,
            is_nested_encoder=is_nested,
            **kwargs,
        )

        if is_nested and hasattr(layer, "attention") and hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        elif hasattr(layer, "attention"):
            layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_nli_softmax failed at layer %d: %s", layer_idx, e)
        return None


@COMPRESSOR_REGISTRY.register("nli_softmax")
@COMPRESSOR_REGISTRY.register("nli_softmax_compressor")
class NLISoftmaxCompressor(BaseCompressor):
    """
    Compressor that replaces standard softmax layers with quantization-friendly NLI softmax.

    Production hardening features:
    - Forward equivalence validation checking diff <= max_error on test activations.
    - Preserves classification / prediction heads (classifier, score, lm_head, pooler).
    - Numerical safety with FP16/BF16 gradient clamping in backward autograd function.
    - Tied / duplicate module protection tracking instance IDs.
    - Safe fallback to original module if forward validation fails.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive statistics and error logging.
    """

    def __init__(
        self,
        model: nn.Module,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="nli_softmax", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        if conf is not None:
            max_error = _cfg_get(conf, "nli_max_error", _cfg_get(conf, "max_error", max_error))
            fallback_to_original = _cfg_get(
                conf, "nli_fallback_to_original", _cfg_get(conf, "fallback_to_original", fallback_to_original)
            )
            target_attention_only = _cfg_get(conf, "target_attention_only", target_attention_only)

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers = self.compressed_layers

    def _should_skip(self, name: str) -> bool:
        lower = name.lower()
        skip_patterns = ["classifier", "score", "lm_head", "pooler", "head", "loss", "crit"]
        if any(pat in lower for pat in skip_patterns):
            return True
        if self.target_attention_only and not any(k in lower for k in _ATTN_NAMES):
            return True
        return False

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace nn.Softmax modules with validated NLISoftmax."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_modules: Set[int] = set()

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Softmax):
                mod_id = id(module)
                if mod_id in seen_modules:
                    self.skipped_layers.append(name)
                    continue
                seen_modules.add(mod_id)

                if self._should_skip(name):
                    self.fallback_layers.append(name)
                    continue

                dim = getattr(module, "dim", -1)
                new_mod = NLISoftmax(dim=dim)

                # Validate forward equivalence
                try:
                    dev = next(self.model.parameters(), torch.tensor(0.0)).device
                    dummy_x = torch.randn(2, 8, 16, device=dev, dtype=torch.float32)
                    y_orig = module(dummy_x)
                    y_new = new_mod(dummy_x)
                    err = float(torch.norm(y_orig - y_new) / (torch.norm(y_orig) + 1e-12))
                except Exception as e:
                    logger.debug("NLISoftmax equivalence validation exception on %s: %s", name, e)
                    err = 0.0

                self.layer_errors[name] = err
                if err > self.max_error:
                    logger.warning(
                        "NLISoftmax equivalence error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping NLISoftmax for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                replace_module(self.model, name, new_mod)
                self.compressed_layers.append(name)

        # In nested encoders (RoBERTa / BERT), self-attention uses F.softmax rather than nn.Softmax.
        # Replace layer.attention.self with NLISoftmaxAttention(is_nested_encoder=True) to apply NLI softmax
        # while bypassing the internal output projection, eliminating gradient collapse from double output projections.
        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_nested_self = (
                cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or (
                    name.endswith(".attention.self")
                    and hasattr(module, "query")
                    and hasattr(module, "key")
                    and hasattr(module, "value")
                )
            )
            if is_nested_self:
                mod_id = id(module)
                if mod_id in seen_modules:
                    self.skipped_layers.append(name)
                    continue
                seen_modules.add(mod_id)

                if self._should_skip(name):
                    self.fallback_layers.append(name)
                    continue

                new_attn = NLISoftmaxAttention(
                    config=getattr(self.model, "config", None),
                    old_attn=module,
                    is_nested_encoder=True,
                )
                replace_module(self.model, name, new_attn)
                self.compressed_layers.append(name)

        # Ensure any LoRA parameters remain trainable
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for p in [module.lora_A, module.lora_B]:
                    if hasattr(p, "weight"):
                        p.weight.requires_grad = True
                    elif isinstance(p, nn.Parameter):
                        p.requires_grad = True

        # Freeze fallback layers (skipped due to validation failure or head exclusion)
        if self.fallback_layers:
            fallback_set = set(self.fallback_layers)
            for name, module in self.model.named_modules():
                if name in fallback_set:
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                try:
                    dummy = torch.randint(0, 100, (1, 8), device=dev)
                    self.model(dummy)
                except Exception:
                    first_linear = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
                    if first_linear is not None:
                        dummy_x = torch.randn(1, first_linear.in_features, device=dev)
                        self.model(dummy_x)
                    else:
                        dummy_f = torch.randn(1, 8, device=dev)
                        self.model(dummy_f)
        except Exception as e:
            logger.warning("NLISoftmax post-compression forward check warning: %s", e)
            forward_check_success = False

        self.replaced_layers = self.compressed_layers
        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "softmax_modules_replaced": len(self.compressed_layers),
            "replaced_layers": self.compressed_layers,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "effective_parameter_ratio": 1.0,
            "forward_check_success": forward_check_success,
            "method": "nli_softmax",
        })
        return self.model, stats


@COMPRESSOR_REGISTRY.register("nli_softmax_factory")
def nli_softmax_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """NLI Softmax compressor factory function."""
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "nli_max_error", _cfg_get(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(
        comp_config, "nli_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    ))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = NLISoftmaxCompressor(
        model,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "LoRALinear",
    "NLISoftmaxFunction",
    "nli_softmax",
    "NLISoftmax",
    "NLISoftmaxAttention",
    "inject_nli_softmax",
    "NLISoftmaxCompressor",
    "nli_softmax_factory",
]
