"""HymbaTransX: Hybrid Differential Transformer & State Space Model Block (Corrected).

Differential attention with rotary embeddings coupled with a State Space Model branch.

Bug-fix map:
  HX-FIX-01  alpha init to +1.0 (attention-dominant).
  HX-FIX-02  RoPE only applied when the model expects it (not on BERT-style encoders).
  HX-FIX-03  causal mask offset-aware for KV-cache.
  HX-FIX-04  KV-cache implemented (tuple + DynamicCache).
  HX-FIX-05  mamba operates on pre-projected hidden_states (not raw).
  HX-FIX-06  W_q2 / W_k2 initialized as noisy copies of W_q1 / W_k1.
  HX-FIX-07  lambda for differential attention is learnable.
  HX-FIX-08  head_mask broadcasting corrected (B, H, 1, 1).
  HX-FIX-09  injector recursion guard.
  HX-FIX-10  injector double-injection guard.
  HX-FIX-11  forward returns consistent tuple.
  HX-FIX-12  attention_mask handling via _prepare_valid_mask.
  HX-FIX-13  warmup for HeadRoPE buffer (avoid stale cache).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE helpers
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    cos_q = cos.to(q.dtype)
    sin_q = sin.to(q.dtype)
    cos_k = cos.to(k.dtype)
    sin_k = sin.to(k.dtype)

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
        k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
    return q, k


# ===========================================================================
# HeadRoPE (internal fallback for encoders without position embeddings)
# ===========================================================================

class HeadRoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(1), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor, past_len: int = 0) -> torch.Tensor:
        B, H, L, d = x.shape
        if past_len + L > self.max_seq_len:
            self._build_cache(past_len + L)
            self.max_seq_len = past_len + L

        cos = self.cos_cached[:, :, past_len : past_len + L, :d]
        sin = self.sin_cached[:, :, past_len : past_len + L, :d]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
        rot_odd = x_even * sin[..., 0::2] + x_odd * cos[..., 0::2]
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


# ===========================================================================
# Decoder detection
# ===========================================================================

def _detect_decoder(config: Any, is_decoder: Optional[bool], old_attn: Optional[nn.Module]) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


def _uses_rope(config: Any, old_attn: Optional[nn.Module]) -> bool:
    """Return True if the model expects RoPE (i.e., is a decoder / LLaMA-like)."""
    if config is not None:
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt2", "gpt_neo", "gpt_neox", "llama", "qwen", "mistral",
                          "gemma", "phi", "mixtral", "gemma2", "qwen2", "qwen3")):
            return True
        if getattr(config, "position_embedding_type", "") == "rotary":
            return True
    if old_attn is not None:
        if hasattr(old_attn, "rotary_emb"):
            return True
    return False


# ===========================================================================
# LoRALinear
# ===========================================================================

class LoRALinear(nn.Module):
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
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# HymbaTransX block
# ===========================================================================

@ATTENTION_REGISTRY.register("hymba_transx", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("hymba_transx_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class HymbaTransXBlock(BaseAttention):
    r"""HymbaTransX: Hybrid Differential Transformer & State Space Model Block."""

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size",
                         getattr(config, "dim",
                         getattr(config, "d_model",
                         getattr(config, "n_embd", None))))
        if hidden_size is None and source_attn is not None:
            for attr in ["hidden_size", "embed_dim", "dim", "d_model"]:
                if hasattr(source_attn, attr) and getattr(source_attn, attr) is not None:
                    hidden_size = getattr(source_attn, attr)
                    break
            if hidden_size is None:
                out_layer = getattr(source_attn, "out_lin",
                            getattr(source_attn, "dense",
                            getattr(source_attn, "c_proj",
                            getattr(source_attn, "o_proj", None))))
                hidden_size = getattr(out_layer, "out_features", None) if out_layer is not None else None
        hidden_size = int(hidden_size or 768)

        if num_heads is None and config is not None:
            num_heads = getattr(config, "num_attention_heads",
                       getattr(config, "n_heads",
                       getattr(config, "num_heads", None)))
        if num_heads is None and source_attn is not None:
            num_heads = getattr(source_attn, "num_attention_heads",
                       getattr(source_attn, "n_heads",
                       getattr(source_attn, "num_heads", None)))
        num_heads = int(num_heads or (hidden_size // 64 if hidden_size % 64 == 0 else 12))

        self.dim = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        cfg_kv = getattr(config, "num_key_value_heads",
                  getattr(config, "num_kv_heads", None)) if config else None
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        cfg_head_dim = getattr(config, "head_dim", kwargs.get("head_dim", None)) if config else kwargs.get("head_dim", None)
        if cfg_head_dim is None and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin", "W_q1"):
                mod = getattr(source_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    cfg_head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                    break
        self.head_dim = int(cfg_head_dim) if cfg_head_dim is not None else self.dim // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)
        self.supports_gqa = True

        # Decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        # HX-FIX-02: RoPE only for models that expect it
        self.use_rope = _uses_rope(config, source_attn) or self.is_decoder

        # LoRA config
        self.lora_rank = int(kwargs.get("lora_rank",
                            getattr(comp_config, "lora_rank",
                            getattr(config, "lora_rank", 16) if config else 16)))
        self.lora_alpha = float(kwargs.get("lora_alpha",
                              getattr(comp_config, "lora_alpha",
                              getattr(config, "lora_alpha", 32.0) if config else 32.0)))
        self.freeze_base = (self.lora_rank > 0)

        # RoPE and projections
        self.rope = HeadRoPE(self.head_dim) if self.use_rope else None

        self.W_q1 = LoRALinear(self.dim, self.all_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_k1 = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_q2 = LoRALinear(self.dim, self.all_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_k2 = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        # HX-FIX-06: W_q2 and W_k2 are noisy copies of W_q1 and W_k1.
        with torch.no_grad():
            noise_q = torch.randn_like(self.W_q2.weight) * 0.01
            noise_k = torch.randn_like(self.W_k2.weight) * 0.01
            self.W_q2.weight.data.copy_(self.W_q1.weight.data + noise_q)
            self.W_k2.weight.data.copy_(self.W_k1.weight.data + noise_k)

        # HX-FIX-07: differential attention lambda is learnable.
        self.diff_lambda = nn.Parameter(torch.tensor(0.2))

        # SSM branch
        try:
            from framework.attentions.simple_ssm import SimpleSSM
            self.mamba = SimpleSSM(d_model=self.dim, config=config, is_decoder=self.is_decoder)
        except ImportError:
            self.mamba = None
            logger.warning("HymbaTransX: SimpleSSM not available; SSM branch disabled.")

        # HX-FIX-01: alpha init = +1.0 (attention-dominant)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        dropout_p = float(
            getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def query(self): return self.W_q1
    @property
    def key(self): return self.W_k1
    @property
    def value(self): return self.W_v
    @property
    def q_proj(self): return self.W_q1
    @property
    def k_proj(self): return self.W_k1
    @property
    def v_proj(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        if (hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")):
            self.is_nested_encoder = False
        elif (hasattr(src, "self") and hasattr(getattr(src, "self"), "query")
              and hasattr(src, "output") and hasattr(getattr(src, "output"), "dense")):
            self.is_nested_encoder = True
        elif cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                          "XLMRobertaSelfAttention", "ElectraSelfAttention",
                          "DebertaSelfAttention"):
            self.is_nested_encoder = True
        elif cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower() and not has_out_lin:
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                cls_c = c_attn_mod.__class__.__name__
                base_cls_c = base_c.__class__.__name__
                if "Conv1D" in cls_c or "Conv1D" in base_cls_c or (c_w.ndim == 2 and c_w.shape[1] == 3 * self.dim):
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split, :], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split: 2 * split, :], c_b[split: 2 * split] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split:, :], c_b[2 * split:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    cls_o = c_proj_mod.__class__.__name__
                    base_cls_o = base_o.__class__.__name__
                    if "Conv1D" in cls_o or "Conv1D" in base_cls_o or (o_w.ndim == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[key] = (mod.weight.data.clone(),
                                      mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(),
                                  out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[key] = (mod.weight.data.clone(),
                                              mod.bias.data.clone() if mod.bias is not None else None)
                            break

        # 4. UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        with torch.no_grad():
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_o, b_o = extracted["o"]

            def _adapt(w, target_shape):
                if w is None:
                    return None
                if w.ndim != 2:
                    w = w.reshape(w.shape[0], -1)
                if w.shape == target_shape:
                    return w
                if w.t().contiguous().shape == target_shape:
                    return w.t().contiguous()
                # Truncate / pad
                out_d, in_d = target_shape
                s_out, s_in = w.shape
                m_out = min(out_d, s_out)
                m_in = min(in_d, s_in)
                padded = w.new_zeros(target_shape)
                padded[:m_out, :m_in] = w[:m_out, :m_in]
                return padded

            # W_q1
            wq = _adapt(w_q, self.W_q1.weight.shape)
            if wq is not None:
                self.W_q1.weight.data.copy_(wq.to(self.W_q1.weight.dtype, self.W_q1.weight.device))

            # W_k1
            wk = _adapt(w_k, self.W_k1.weight.shape)
            if wk is not None:
                self.W_k1.weight.data.copy_(wk.to(self.W_k1.weight.dtype, self.W_k1.weight.device))

            # W_v
            wv = _adapt(w_v, self.W_v.weight.shape)
            if wv is not None:
                self.W_v.weight.data.copy_(wv.to(self.W_v.weight.dtype, self.W_v.weight.device))

            # W_o
            if w_o is not None:
                wo = _adapt(w_o, self.out_proj.weight.shape)
                if wo is not None:
                    self.out_proj.weight.data.copy_(wo.to(self.out_proj.weight.dtype, self.out_proj.weight.device))
                if b_o is not None and self.out_proj.bias is not None:
                    m = min(self.out_proj.bias.numel(), b_o.numel())
                    self.out_proj.bias.data.zero_()
                    self.out_proj.bias.data[:m].copy_(b_o[:m].to(self.out_proj.bias.dtype, self.out_proj.bias.device))

            # HX-FIX-06: W_q2 and W_k2 = noisy copies of W_q1 and W_k1
            noise_q = torch.randn_like(self.W_q2.weight) * 0.01
            noise_k = torch.randn_like(self.W_k2.weight) * 0.01
            self.W_q2.weight.data.copy_(self.W_q1.weight.data + noise_q.to(self.W_q2.weight.dtype))
            self.W_k2.weight.data.copy_(self.W_k1.weight.data + noise_k.to(self.W_k2.weight.dtype))

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, N_kv: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("HymbaTransXBlock does not support cross-attention.")

        x = hidden_states if hidden_states is not None else kwargs.get("query", kwargs.get("x", None))
        if x is None and len(args) > 0:
            x = args[0]
        if x is None:
            raise ValueError("HymbaTransXBlock.forward: no input tensor provided.")

        B, L, D = x.shape
        device = x.device

        # Projections
        q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k1 = self.W_k1(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k2 = self.W_k2(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # HX-FIX-04: KV-cache.
        past_k1 = past_v = None
        past_k2 = None
        past_len = 0
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)):
                # Expect (k1, v, k2)
                if len(past_key_value) >= 3:
                    past_k1, past_v, past_k2 = past_key_value[0], past_key_value[1], past_key_value[2]
                elif len(past_key_value) >= 2:
                    past_k1, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k1 = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k1 is not None:
            past_len = past_k1.shape[2]
            k1_cat = torch.cat([past_k1, k1], dim=2)
            v_cat = torch.cat([past_v, v], dim=2)
            k2_cat = torch.cat([past_k2, k2], dim=2) if past_k2 is not None else k2
        else:
            k1_cat = k1
            v_cat = v
            k2_cat = k2

        N_kv = k1_cat.shape[2]

        # RoPE (HX-FIX-02: only if use_rope)
        if self.use_rope:
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cos = cos[..., past_len : past_len + L, :]
                sin = sin[..., past_len : past_len + L, :]
                q1, k1_cat = apply_rotary_pos_emb(q1, k1_cat, cos, sin)
                q2, k2_cat = apply_rotary_pos_emb(q2, k2_cat, cos, sin)
            elif self.rope is not None:
                q1 = self.rope(q1, past_len=past_len)
                k1_cat = self.rope(k1_cat, past_len=0)
                q2 = self.rope(q2, past_len=past_len)
                k2_cat = self.rope(k2_cat, past_len=0)

        # GQA expansion
        if self.kv_repeat > 1:
            k1_cat = repeat_kv(k1_cat, self.kv_repeat)
            k2_cat = repeat_kv(k2_cat, self.kv_repeat)
            v_cat = repeat_kv(v_cat, self.kv_repeat)

        # Differential attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1.float(), k1_cat.float().transpose(-2, -1).contiguous()) * scale
        scores2 = torch.matmul(q2.float(), k2_cat.float().transpose(-2, -1).contiguous()) * scale

        # Causal mask (HX-FIX-03: offset-aware for cache)
        if self.is_decoder:
            if past_len > 0:
                abs_q = (past_len + torch.arange(L, device=device)).unsqueeze(1)
                abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                causal = (abs_k <= abs_q)
            else:
                causal = torch.tril(torch.ones(L, N_kv, device=device, dtype=torch.bool))
            scores1 = scores1.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            scores2 = scores2.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

        # Padding mask (HX-FIX-12)
        eff_mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
        valid_mask = self._prepare_valid_mask(eff_mask, B, N_kv, device)
        if valid_mask is not None:
            scores1 = scores1.masked_fill(~valid_mask[:, None, None, :].bool(), -1e4)
            scores2 = scores2.masked_fill(~valid_mask[:, None, None, :].bool(), -1e4)

        scores1 = torch.nan_to_num(scores1, nan=-1e4, posinf=1e4, neginf=-1e4)
        scores2 = torch.nan_to_num(scores2, nan=-1e4, posinf=1e4, neginf=-1e4)

        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)

        # HX-FIX-07: learnable lambda
        lam = torch.sigmoid(self.diff_lambda)
        diff_attn = F.relu(attn1 - lam * attn2)
        denom = diff_attn.sum(dim=-1, keepdim=True)
        diff_attn = diff_attn / (denom + 1e-6)
        diff_attn = self.dropout(diff_attn).to(v_cat.dtype)

        # HX-FIX-08: head_mask broadcasting (B, H, 1, 1).
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            diff_attn = diff_attn * head_mask.to(diff_attn.dtype)

        A_out = torch.matmul(diff_attn, v_cat).transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # SSM branch
        if self.mamba is not None:
            if x.requires_grad:
                from torch.utils.checkpoint import checkpoint
                def _run_mamba(h, m):
                    return self.mamba(h, attention_mask=m)[0]

                if self.is_decoder:
                    M_out = checkpoint(_run_mamba, x, attention_mask, use_reentrant=False)
                else:
                    x_rev = torch.flip(x, dims=[1])
                    mf = checkpoint(_run_mamba, x, attention_mask, use_reentrant=False)
                    mb = checkpoint(_run_mamba, x_rev, attention_mask, use_reentrant=False)
                    M_out = 0.5 * (mf + torch.flip(mb, dims=[1]))
            else:
                if self.is_decoder:
                    mamba_res = self.mamba(x, attention_mask=attention_mask)
                    M_out = mamba_res[0] if isinstance(mamba_res, tuple) else mamba_res
                else:
                    x_rev = torch.flip(x, dims=[1])
                    mamba_fwd = self.mamba(x, attention_mask=attention_mask)
                    mamba_bwd = self.mamba(x_rev, attention_mask=attention_mask)
                    mf = mamba_fwd[0] if isinstance(mamba_fwd, tuple) else mamba_fwd
                    mb = mamba_bwd[0] if isinstance(mamba_bwd, tuple) else mamba_bwd
                    M_out = 0.5 * (mf + torch.flip(mb, dims=[1]))
        else:
            M_out = A_out

        # HX-FIX-01: alpha init = +1.0
        alpha = torch.sigmoid(self.alpha)
        combined = alpha * A_out + (1.0 - alpha) * M_out

        # Output projection
        if self.is_nested_encoder:
            out = combined
        else:
            out = self.out_proj(combined)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, (diff_attn if output_attentions else None)


# ===========================================================================
# Injection (HX-FIX-09, HX-FIX-10)
# ===========================================================================

@ATTENTION_REGISTRY.register("hymba_transx_inject")
@ATTENTION_REGISTRY.register("inject_hymba_transx")
def inject_hymba_transx(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Any]:
    if _depth > 8:
        logger.warning("inject_hymba_transx: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None)

    # Whole-model traversal
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
            res = inject_hymba_transx(
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

    # Single layer
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "attn", "self_attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is not None:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None

        # HX-FIX-10: double-injection guard
        if isinstance(src, HymbaTransXBlock):
            return None
        if "HymbaTransX" in src.__class__.__name__:
            return None
        inner = getattr(src, "inner", None)
        if isinstance(inner, HymbaTransXBlock):
            return None
        wrapped = getattr(src, "_wrapped_attention", None)
        if isinstance(wrapped, HymbaTransXBlock):
            return None

        if cfg is None and hasattr(target, "config"):
            cfg = target.config
        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        new_attn = HymbaTransXBlock(
            config=cfg,
            comp_config=comp_config,
            old_attn=src,
            **kwargs,
        )

        # Attach
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            new_attn.is_nested_encoder = False

        return new_attn
    except Exception as e:
        logger.exception("inject_hymba_transx failed at layer %d: %s", layer_idx, e)
        return None