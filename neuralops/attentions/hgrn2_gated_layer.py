"""
HGRN-2 Layer v5 — Production-Hardened with proper causal safety.

FIXES in v5:
  - FIX 15: Removed duplicate decoder detection blocks
  - FIX 16: output_scale init = 0.1 (was 0.01 — too small)
  - FIX 17: inject_hgrn2_layer passes is_decoder explicitly
  - FIX 18: Default fusion_gate = -4.0 (98% local, safer)
  - FIX 19: Warning log when is_decoder=False

All v4 fixes preserved:
  - FIX 11: copy_weights maps w_q → g_proj and w_k → input_gate
  - FIX 12: 4D causal mask handling corrected
  - FIX 13: forget gate f is NOT decayed on padding tokens
  - FIX 14: res_clip default = None

All v3 fixes preserved:
  - FIX 1: forget_gate.bias = -4.0
  - FIX 2: state_norm init = identity
  - FIX 4: Remove 0.5*v in output
  - FIX 5: output_scale added
  - FIX 6: env var toggles
  - FIX 7: g_proj.bias = 0.0
  - FIX 8: ADDED local sliding-window attention (Q·K)
  - FIX 10: fusion gate between HGRN and local attention
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

try:
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
    HAS_TRITON_HGRN2 = True
    logger.info("HGRN2: Triton scan available")
except Exception as _e:
    logger.warning("HGRN2: Triton scan unavailable: %s", _e)
    hgrn2_scan_triton = None
    HAS_TRITON_HGRN2 = False


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any):
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
@ATTENTION_REGISTRY.register("hgrn2_gated_layer")
@ATTENTION_REGISTRY.register("hgrn2")
class HGRN2Layer(BaseAttention):
    """HGRN-2 hybrid with local sliding-window attention."""

    is_custom_attention: bool = True

    def __init__(self, config=None, rank=None, d_model=None, res_clip=None,
                 lora_rank=16, lora_alpha=32.0, is_decoder=None,
                 freeze_base=True, window_size=64, **kwargs):
        super().__init__(config, rank)

        # ---- d_model resolution ----
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size")
            if d_model is None:
                if isinstance(config, int):
                    d_model = config
                elif config is not None:
                    d_model = _cfg_get(config, "hidden_size",
                                       _cfg_get(config, "d_model",
                                                _cfg_get(config, "n_embd",
                                                         _cfg_get(config, "dim", None))))
            if d_model is None:
                d_model = kwargs.get("dim", 128)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model

        # ---- FIX 14: res_clip default = None ----
        _res_clip_default = _env_float("HGRN2_RES_CLIP", 0.0)
        self.res_clip = res_clip if res_clip is not None else (
            _res_clip_default if _res_clip_default > 0 else None
        )

        self.freeze_base = freeze_base

        # ---- FIX 15: single decoder detection block ----
        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            cfg_is_dec = (
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
                or kwargs.get("is_decoder", False)
                or kwargs.get("is_causal", False)
            )
            old_attn_kw = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
            if old_attn_kw is not None:
                if (hasattr(old_attn_kw, "c_attn")
                        or "gpt" in old_attn_kw.__class__.__name__.lower()):
                    cfg_is_dec = True
            if config is None and old_attn_kw is None:
                logger.warning("HGRN2: defaulting to is_decoder=True for causal safety")
                cfg_is_dec = True
            self.is_decoder = bool(cfg_is_dec)

        # ---- FIX 19: warning for bidirectional on decoder-like ----
        if not self.is_decoder:
            logger.warning(
                "HGRN2: is_decoder=False — bidirectional mode. "
                "This will leak future tokens for GPT-family models."
            )

        self.is_nested_encoder = False
        self.lora_rank = lora_rank if lora_rank is not None else 16
        self.lora_alpha = lora_alpha
        self.lora_scaling = (self.lora_alpha / max(1, self.lora_rank)) if self.lora_rank > 0 else 1.0

        # ---- num_heads ----
        self.num_heads = int(_cfg_get(config, "num_attention_heads",
                                       _cfg_get(config, "num_heads", 12)))
        while self.num_heads > 1 and (self.d_model % self.num_heads != 0):
            self.num_heads -= 1
        self.head_dim = self.d_model // max(1, self.num_heads)

        # ---- Projections ----
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj_attn = nn.Linear(self.d_model, self.d_model, bias=True)

        self.forget_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.input_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.g_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---- FIX 2: state_norm init = identity ----
        self.state_norm = nn.LayerNorm(self.d_model)
        with torch.no_grad():
            self.state_norm.weight.fill_(1.0)
            self.state_norm.bias.zero_()

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---- FIX 1: forget_gate.bias = -4.0 ----
        _forget_bias = _env_float("HGRN2_FORGET_BIAS", -4.0)
        nn.init.normal_(self.forget_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.forget_gate.bias, _forget_bias)

        nn.init.normal_(self.input_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.input_gate.bias, 0.0)

        _g_bias = _env_float("HGRN2_G_BIAS", 0.0)
        nn.init.normal_(self.g_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.g_proj.bias, _g_bias)

        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.v_proj_attn.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.v_proj_attn.bias)

        # ---- FIX 16: output_scale init = 1.0 (was 0.1/0.01) ----
        _out_scale = _env_float("HGRN2_OUTPUT_SCALE", 1.0)
        self.output_scale = nn.Parameter(torch.full((1,), _out_scale))

        # ---- FIX 18: fusion_gate default = -4.0 (98% local) ----
        _fusion_init = _env_float("HGRN2_FUSION_GATE", -4.0)
        self.fusion_gate = nn.Linear(self.d_model, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.fusion_gate.weight)
            self.fusion_gate.bias.fill_(_fusion_init)

        self.window_size = int(_cfg_get(config, "window_size", window_size))

        # ---- LoRA ----
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

        # ---- Freeze base ----
        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj, self.q_proj, self.k_proj, self.v_proj_attn):
                for p in mod.parameters():
                    p.requires_grad = False
            for mod in (self.forget_gate, self.input_gate, self.g_proj, self.fusion_gate):
                for p in mod.parameters():
                    p.requires_grad = True

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)
            target_dtype = kwargs.get("dtype")
            if target_dtype is None:
                p = next(old_attn.parameters(), None)
                if p is not None and isinstance(p.dtype, torch.dtype):
                    target_dtype = p.dtype
            if target_dtype is not None:
                self.to(target_dtype)

    def copy_weights(self, old_attn):
        w_q = w_k = w_v = w_o = None
        b_q = b_k = b_v = b_o = None

        src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
        is_nested = (
            src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (hasattr(old_attn, "query") and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj"))
            or (hasattr(old_attn, "self") and hasattr(old_attn, "output"))
        )
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
            src = old_attn.self
            if hasattr(old_attn.output, "dense"):
                w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
        else:
            src = old_attn

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
            c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
            if c_attn_w is not None:
                split_size = c_attn_w.shape[0] // 3
                w_q = c_attn_w[:split_size, :]
                w_k = c_attn_w[split_size:2*split_size, :]
                w_v = c_attn_w[2*split_size:, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:split_size]
                    b_k = c_attn_b[split_size:2*split_size]
                    b_v = c_attn_b[2*split_size:]
            if hasattr(src, "c_proj"):
                w_o, b_o = _unpack_weight_bias(src.c_proj)

        elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
            w_q, b_q = _unpack_weight_bias(src.query)
            w_k, b_k = _unpack_weight_bias(src.key)
            w_v, b_v = _unpack_weight_bias(src.value)
            if w_o is None:
                for attr in ("dense", "out_proj", "o_proj"):
                    if hasattr(src, attr):
                        w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                        break

        elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
            w_q, b_q = _unpack_weight_bias(src.q_proj)
            w_k, b_k = _unpack_weight_bias(src.k_proj)
            w_v, b_v = _unpack_weight_bias(src.v_proj)
            if hasattr(src, "o_proj"):
                w_o, b_o = _unpack_weight_bias(src.o_proj)

        elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
            w_q, b_q = _unpack_weight_bias(src.q_lin)
            w_k, b_k = _unpack_weight_bias(src.k_lin)
            w_v, b_v = _unpack_weight_bias(src.v_lin)
            if hasattr(src, "out_lin"):
                w_o, b_o = _unpack_weight_bias(src.out_lin)

        if w_q is None or w_k is None or w_v is None:
            try:
                adapter = UniversalAdapter(old_attn)
                extracted = adapter.extract_qkv_out()
                if extracted:
                    w_q = extracted.get("q_weight") or w_q
                    w_k = extracted.get("k_weight") or w_k
                    w_v = extracted.get("v_weight") or w_v
                    b_v = extracted.get("v_bias") or b_v
                    if w_o is None:
                        w_o = extracted.get("out_weight")
                        b_o = extracted.get("out_bias")
            except Exception as ex:
                logger.debug("UniversalAdapter fallback failed: %s", ex)

        def _copy_param(target_linear, w, b):
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
                if b is not None and target_linear.bias is not None:
                    tb = target_linear.bias
                    if b.shape == tb.shape:
                        tb.copy_(b.to(tb.dtype))
                    else:
                        min_b = min(tb.numel(), b.numel())
                        tb.data.zero_()
                        tb.data[:min_b].copy_(b[:min_b].to(tb.dtype))

        # Copy Q, K, V, O
        _copy_param(self.q_proj, w_q, b_q)
        _copy_param(self.k_proj, w_k, b_k)
        _copy_param(self.v_proj_attn, w_v, b_v)
        _copy_param(self.v_proj, w_v, b_v)
        _copy_param(self.out_proj, w_o, b_o)

        # FIX 11: map W_q → g_proj and W_k → input_gate
        _copy_param(self.g_proj, w_q, b_q)
        _copy_param(self.input_gate, w_k, b_k)

        fnorm = float(torch.linalg.norm(self.v_proj.weight.detach().float()))
        if fnorm < 1e-4:
            logger.warning("HGRN2Layer v_proj weight norm near zero: %g", fnorm)

        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj, self.q_proj, self.k_proj, self.v_proj_attn):
                for p in mod.parameters():
                    p.requires_grad = False

    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj_attn

    def _scan_direction(self, f, iv, chunk_size=32):
        if HAS_TRITON_HGRN2 and hgrn2_scan_triton is not None and f.is_cuda:
            try:
                return hgrn2_scan_triton(f.contiguous(), iv.contiguous(), res_clip=self.res_clip)
            except Exception as e:
                logger.debug("HGRN2 Triton scan failed, falling back: %s", e)

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

    def _recurrent_scan_parallel(self, f, iv, mask=None):
        if mask is not None:
            if mask.dim() == 2:
                mask_float = mask.float()
            elif mask.dim() == 4:
                v = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                v = v.any(dim=1)
                mask_float = v.any(dim=1).float() if v.dim() == 3 else v.float()
            else:
                mask_float = mask.float()

            # FIX 13: iv masked to 0 on padding, but f set to 1.0
            iv = iv * mask_float.unsqueeze(-1)
            f = torch.where(
                mask_float.unsqueeze(-1).bool(),
                f,
                torch.ones_like(f),
            )

        h_fwd = self._scan_direction(f, iv)
        if self.is_decoder:
            return h_fwd, None
        f_rev = torch.flip(f, dims=[1])
        iv_rev = torch.flip(iv, dims=[1])
        h_rev = self._scan_direction(f_rev, iv_rev)
        h_bwd = torch.flip(h_rev, dims=[1])
        return h_fwd, h_bwd

    def _local_attention(self, hidden_states):
        """FIX 8: local sliding-window attention."""
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj_attn(hidden_states).view(B, L, H, d).transpose(1, 2)

        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        idx = torch.arange(L, device=hidden_states.device)
        window_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        if self.is_decoder:
            causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
            window_mask = window_mask & causal
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), -1e4)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return out

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False,
                output_hidden_states=False, return_dict=True, *args, **kwargs):
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention not supported in HGRN2Layer")

        B, L, D = hidden_states.shape
        target_dtype = hidden_states.dtype

        # ---- HGRN-2 branch ----
        raw_log_decay = -F.softplus(self.forget_gate(hidden_states))
        bounded_log_decay = raw_log_decay.clamp(min=-3.0, max=-1e-4)
        f = torch.exp(bounded_log_decay).to(target_dtype)

        i = torch.sigmoid(self.input_gate(hidden_states)).to(target_dtype)
        input_scale = ((1.0 - f) * i).to(target_dtype)
        v = self.v_proj(hidden_states).to(target_dtype)
        if self.lora_rank > 0 and self.v_lora_A is not None and self.v_lora_B is not None:
            v_lora = (torch.matmul(
                torch.matmul(hidden_states, self.v_lora_A.t().contiguous().to(target_dtype)),
                self.v_lora_B.t().contiguous().to(target_dtype)) * self.lora_scaling)
            v = v + v_lora

        z = self.g_proj(hidden_states).to(target_dtype)
        iv = (input_scale * v).to(target_dtype)

        # ---- FIX 12: 4D mask handling ----
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 3:
                m = (attention_mask > -1e4) if attention_mask.dtype.is_floating_point else attention_mask.bool()
                if m.dim() == 3:
                    mask_bool = m.any(dim=1) if m.shape[1] > 1 else m[:, 0, :]
            elif attention_mask.dim() == 4:
                m = (attention_mask > -1e4) if attention_mask.dtype.is_floating_point else attention_mask.bool()
                m = m.any(dim=-2)
                if m.dim() == 3:
                    mask_bool = m.any(dim=1)
                else:
                    mask_bool = m

        # ---- Scan ----
        h_fwd, h_bwd = self._recurrent_scan_parallel(f, iv, mask_bool)
        if self.is_decoder or h_bwd is None:
            h_ctx = h_fwd.to(target_dtype)
        else:
            h_ctx = (h_fwd + h_bwd).to(target_dtype)

        sn_dtype = (self.state_norm.weight.dtype
                    if hasattr(self.state_norm, "weight") and self.state_norm.weight is not None
                    else target_dtype)
        h_norm = self.state_norm(h_ctx.to(sn_dtype)).to(target_dtype)
        hgrn_out = (h_norm * F.silu(z)).to(target_dtype)

        # ---- Local attention branch ----
        local_out = self._local_attention(hidden_states)

        # ---- Fusion ----
        fusion_logits = self.fusion_gate(hidden_states).to(target_dtype)
        gate = torch.sigmoid(fusion_logits)
        fused = gate * hgrn_out + (1 - gate) * local_out

        # ---- Output ----
        out = self.out_proj(fused).to(target_dtype)
        if self.lora_rank > 0 and self.out_lora_A is not None and self.out_lora_B is not None:
            out_lora = (torch.matmul(
                torch.matmul(fused, self.out_lora_A.t().contiguous().to(target_dtype)),
                self.out_lora_B.t().contiguous().to(target_dtype)) * self.lora_scaling)
            out = out + out_lora

        out = out * self.output_scale

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out.to(target_dtype), None


@ATTENTION_REGISTRY.register("hgrn2_layer_inject")
@ATTENTION_REGISTRY.register("hgrn2_gated_layer_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_layer")
@ATTENTION_REGISTRY.register("hgrn2_inject")
def inject_hgrn2_layer(layer=None, model=None, model_config=None, comp_config=None,
                        layer_idx=0, total_layers=12, old_attn=None,
                        lora_rank=16, lora_alpha=32.0, **kwargs):
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # ---- Whole-model traversal ----
    if target is not None and (hasattr(target, "transformer") or hasattr(target, "distilbert")
                               or hasattr(target, "roberta") or hasattr(target, "bert")
                               or (hasattr(target, "model") and not hasattr(target, "forward"))):
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
            if kwargs.get("freeze_base", True):
                for p in target.parameters():
                    p.requires_grad = False
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_hgrn2_layer(layer=lyr, model_config=cfg, comp_config=comp_config,
                                          layer_idx=idx, total_layers=tot,
                                          lora_rank=lora_rank, lora_alpha=lora_alpha, **kwargs)
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # ---- Single layer ----
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

    if isinstance(curr_attn, HGRN2Layer):
        return None
    if "HGRN2" in curr_attn.__class__.__name__:
        return None

    # ---- FIX 17: explicit is_decoder detection and pass ----
    _mt = str(getattr(cfg, "model_type", "") or "").lower()
    _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
    is_dec = (
        bool(getattr(cfg, "is_decoder", False) if cfg else False)
        or bool(getattr(cfg, "causal", False) if cfg else False)
        or any(_mt.startswith(p) for p in _dec_pfx)
        or hasattr(curr_attn, "c_attn")
        or "gpt" in curr_attn.__class__.__name__.lower()
    )

    try:
        new_layer = HGRN2Layer(
            config=cfg,
            old_attn=curr_attn,
            is_decoder=is_dec,  # ← FIX 17: explicit
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

    # ---- Splice ----
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


__all__ = ["HGRN2Layer", "inject_hgrn2_layer"]