"""
Cosformer — Production-Hardened (Hybrid Local Attention + Fixed Math).
"""

from __future__ import annotations
from neuralops.core.utils.config import _cfg_get
from neuralops.core.utils.lora import LoRALinear
from neuralops.core.utils.rope import apply_rotary_pos_emb, rotate_half, _rotate_half
from neuralops.core.utils.gqa import repeat_kv

import logging
import math
import os
import threading

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

try:
    from neuralops.core.triton_kernels import triton_dispatch
except Exception:
    triton_dispatch = None




_inject_state = threading.local()
_MAX_DEPTH = 4

def _get_inject_depth() -> int:
    return getattr(_inject_state, "depth", 0)

def _set_inject_depth(depth: int) -> None:
    _inject_state.depth = depth
def _neg_inf(dtype: torch.dtype) -> float:
    """Return the smallest finite value representable in `dtype`."""
    return torch.finfo(dtype).min








@ATTENTION_REGISTRY.register("cosformer", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("cosformer_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class CosformerAttention(BaseAttention):
    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    triton_kernel = "linear_attn"

    def __init__(self, config=None, rank=None, max_seq_len=512, act_fun="relu",
                 eps=1e-6, pretrained_attn=None, old_attn=None, comp_config=None,
                 layer_id=None, layer_idx=None, window_size=256, **kwargs):
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn


        hidden_size = (
            kwargs.get("hidden_size")
            or kwargs.get("embed_dim")
            or kwargs.get("d_model")
            or kwargs.get("dim")
        )
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")

        if hidden_size is None:
            hidden_size = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "n_embd", _cfg_get(config, "d_model", None))))
            if hidden_size is None and source_attn is not None:
                out_layer = getattr(source_attn, "out_lin",
                                    getattr(source_attn, "dense",
                                            getattr(source_attn, "c_proj",
                                                    getattr(source_attn, "o_proj", None))))
                hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            elif hidden_size is None:
                hidden_size = 768

        if num_heads is None:
            num_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", _cfg_get(config, "n_head", None)))
            if num_heads is None and source_attn is not None:
                num_heads = getattr(source_attn, "n_heads",
                                    getattr(source_attn, "num_attention_heads",
                                            getattr(source_attn, "num_heads", 12)))
            elif num_heads is None:
                num_heads = 12

        super().__init__(config, rank)
        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)

        resolved_head_dim = _cfg_get(config, "head_dim", None)
        if not resolved_head_dim and source_attn is not None:
            resolved_head_dim = getattr(source_attn, "head_dim", None)
        if not resolved_head_dim:
            resolved_head_dim = max(1, self.dim // max(1, self.num_heads))
        self.head_dim = int(resolved_head_dim)

        kv_heads = _cfg_get(config, "num_key_value_heads", None)
        if kv_heads is None and source_attn is not None:
            kv_heads = getattr(source_attn, "num_key_value_heads",
                               getattr(source_attn, "num_kv_heads",
                                       getattr(source_attn, "n_kv_heads", None)))
            if kv_heads is None and hasattr(source_attn, "k_proj") \
                    and hasattr(source_attn.k_proj, "out_features") and self.head_dim > 0:
                kv_heads = source_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads = int(kv_heads if kv_heads is not None else self.num_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.max_seq_len = int(_cfg_get(config, "max_position_embeddings", max_seq_len))
        self.act_fun = act_fun
        self.eps = float(eps)
        
        # Local attention parameters
        self.window_size = int(_cfg_get(config, "window_size", window_size))

        self.layer_idx = int(kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        self.lora_rank = int(_cfg_get(config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha = float(_cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base = self.lora_rank > 0

        model_type = getattr(config, "model_type", "").lower() if config is not None else ""
        is_dec = _cfg_get(config, "is_decoder", _cfg_get(config, "is_causal", kwargs.get("is_causal", False)))
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "phi", "gemma", "opt", "bloom", "falcon")
        
        if (any(model_type.startswith(p) for p in _decoder_prefixes)
            or (hasattr(config, "architectures") and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])))):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or "gpt" in source_attn.__class__.__name__.lower():
                is_dec = True
        if config is None and source_attn is None:
            is_dec = True
            
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = bool(kwargs.get("is_nested_encoder", False))

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        
        self.v_proj_attn = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                       lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # Additive residual scale for global cosformer branch.
        # Zero-init to match GPT-2 exactly at step 0, avoiding KD shock.
        # As LoRA trains, fusion_scale learns when/how much global context to blend.
        self.fusion_scale = nn.Parameter(torch.zeros(1))

        dropout_p = getattr(config, "attention_probs_dropout_prob", 0.1) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config = comp_config
        self.triton_kwargs = {"phi": "elu+1", "eps": self.eps, "cos_weight": None, "sin_weight": None}

        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj

    def copy_weights(self, src):
        if src is None: return

        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb

        extracted = {"q": (None, None), "k": (None, None),
                     "v": (None, None), "o": (None, None)}

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out_lin = any(hasattr(src, a) for a in ("out_lin", "c_proj", "o_proj", "out_proj", "dense"))
        is_nested = False
        if ((hasattr(src, "self") and hasattr(src.self, "query") and hasattr(src, "output") and hasattr(src.output, "dense")) or
            (cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention") or
             (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) and not has_out_lin):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_w = src.c_attn.weight.data.clone()
            if hasattr(src.c_attn, "nf") or c_w.shape[0] < c_w.shape[1]: c_w = c_w.t().contiguous()
            sz = c_w.size(0) // 3
            c_b = src.c_attn.bias.data.clone() if src.c_attn.bias is not None else None
            extracted["q"] = (c_w[:sz, :], c_b[:sz] if c_b is not None else None)
            extracted["k"] = (c_w[sz:2*sz, :], c_b[sz:2*sz] if c_b is not None else None)
            extracted["v"] = (c_w[2*sz:, :], c_b[2*sz:] if c_b is not None else None)
            if hasattr(src, "c_proj"):
                o_w = src.c_proj.weight.data.clone()
                if hasattr(src.c_proj, "nf") or o_w.shape[0] < o_w.shape[1]: o_w = o_w.t().contiguous()
                o_b = src.c_proj.bias.data.clone() if src.c_proj.bias is not None else None
                extracted["o"] = (o_w, o_b)
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(), b.data.clone() if b is not None else None)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(), b.data.clone() if b is not None else None)
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj", "c_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(), b.data.clone() if b is not None else None)
                            break
        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            raise RuntimeError(f"CosformerAttention.copy_weights: missing {missing} from {src.__class__.__name__}")

        target_map = {"q": self.q_proj, "k": self.k_proj, 
                      "v": [self.v_proj, self.v_proj_attn], "o": self.out_proj}

        with torch.no_grad():
            for proj_key, targets in target_map.items():
                w, b = extracted[proj_key]
                if w is None: continue
                if w.ndim != 2: w = w.reshape(w.size(0), -1)
                
                targets_list = targets if isinstance(targets, list) else [targets]
                for target_mod in targets_list:
                    w_to_use = w
                    tw = target_mod.weight
                    if w_to_use.shape != tw.shape:
                        if tw.shape == w_to_use.t().contiguous().shape:
                            w_to_use = w_to_use.t().contiguous()
                        else:
                            can_slice = (
                                w_to_use.size(0) >= tw.size(0)
                                or w_to_use.size(1) >= tw.size(1)
                            )
                            if not can_slice:
                                raise ValueError(
                                    f"[CosformerAttention.copy_weights] Cannot reconcile shapes: "
                                    f"source {tuple(w_to_use.shape)} vs target {tuple(tw.shape)}. "
                                    f"Check the source architecture."
                                )
                            logger.warning(
                                "copy_weights: shape mismatch %s vs %s — slicing.",
                                tuple(w_to_use.shape), tuple(tw.shape),
                            )
                    out_d = min(tw.size(0), w_to_use.size(0))
                    in_d = min(tw.size(1), w_to_use.size(1))
                    if tw.shape != (w_to_use.size(0), w_to_use.size(1)) and min(w_to_use.shape) == min(tw.shape):
                        target_mod.weight = nn.Parameter(w_to_use.to(tw.dtype))
                        tw = target_mod.weight
                    else:
                        tw.data[:out_d, :in_d].copy_(w_to_use[:out_d, :in_d].to(tw.dtype))

                    if b is not None and getattr(target_mod, "bias", None) is not None:
                        tb = target_mod.bias
                        b_to_use = b.data.clone() if hasattr(b, "data") else b.clone()
                        if b_to_use.shape != tb.shape:
                            b_out_d = min(tb.size(0), b_to_use.size(0))
                            tb.data[:b_out_d].copy_(b_to_use[:b_out_d].to(tb.dtype))
                        else:
                            tb.data.copy_(b_to_use.to(tb.dtype))

    def _get_cos_sin_weights(self, seq_len, device, dtype, pos_ids=None):
        if pos_ids is not None:
            pos = pos_ids.float().to(device=device)
            m = max(16384, self.max_seq_len)
            angles = (math.pi / 2.0 / m) * pos
            cos_weight = torch.cos(angles).unsqueeze(1).unsqueeze(-1).to(dtype=dtype)  # [B, 1, seq_len, 1]
            sin_weight = torch.sin(angles).unsqueeze(1).unsqueeze(-1).to(dtype=dtype)
            return cos_weight, sin_weight

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        m = max(16384, self.max_seq_len)
        angles = (math.pi / 2.0 / m) * pos
        cos_weight = torch.cos(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        sin_weight = torch.sin(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        return cos_weight, sin_weight

    
    def _build_padding_mask(self, attention_mask, L, device, dtype):
        """Convert (B, L) padding mask to additive (B, 1, 1, L) mask.
        Returns None if attention_mask is None.
        """
        if attention_mask is None:
            return None
        am = attention_mask
        if am.dim() == 4:
            return am.to(device=device, dtype=dtype)
        if am.dim() == 3:
            am = am.unsqueeze(2)
        elif am.dim() == 2:
            am = am[:, None, None, :]
        else:
            raise ValueError(f"Unexpected attention_mask shape: {tuple(am.shape)}")
        return (1.0 - am.to(dtype=dtype)) * _neg_inf(dtype)

    def _local_attention(self, Q, K, V_local, padding_mask, seq_len, device, dtype):
        B, H, L, d = Q.shape
        neg_inf = _neg_inf(dtype)

        idx = torch.arange(L, device=device)
        window_bool = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window_size
        if self.is_decoder:
            causal_bool = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
            window_bool = window_bool & causal_bool

        window_additive = torch.where(
            window_bool,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), neg_inf, device=device, dtype=dtype),
        ).unsqueeze(0).unsqueeze(0)

        additive_mask = window_additive
        if padding_mask is not None:
            additive_mask = additive_mask + padding_mask

        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                Q, K, V_local, attn_mask=additive_mask, dropout_p=0.0, is_causal=False,
            )

        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale + additive_mask
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V_local)


    def _forward_causal(self, Q_cos, Q_sin, K_cos, K_sin, V, padding_mask, B, H, d, N, device, dtype):
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            Q_cos = Q_cos.float()
            Q_sin = Q_sin.float()
            K_cos = K_cos.float()
            K_sin = K_sin.float()
            V = V.float()

            if N <= 2048:
                score_c = torch.matmul(Q_cos, K_cos.transpose(-1, -2))
                score_s = torch.matmul(Q_sin, K_sin.transpose(-1, -2))
                scores = score_c + score_s

                if self.is_decoder:
                    causal_bool = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
                    scores = scores.masked_fill(~causal_bool, 0.0)

                if padding_mask is not None:
                    # Zero contribution from padded keys.
                    scores = scores * (padding_mask == 0).float()

                scores = torch.clamp(scores, min=0.0, max=1e4)
                num = torch.matmul(scores, V)
                from neuralops.utils.amp_safety import safe_linear_attn_denominator
                denom = safe_linear_attn_denominator(scores.sum(dim=-1, keepdim=True), eps=1e-5, floor_val=1e-4)
                res = num / denom
                return torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4).to(dtype)

        logger.warning(
            "CosformerAttention._forward_causal: N=%d > 2048, using chunked accumulation.",
            N,
        )
        from neuralops.utils.amp_safety import safe_linear_attn_denominator
        chunk = 1024
        out = torch.zeros(B, H, N, d, device=device, dtype=dtype)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Qc = Q_cos[:, :, start:end]
            Qs = Q_sin[:, :, start:end]
            sc = torch.matmul(Qc, K_cos.transpose(-1, -2))
            ss = torch.matmul(Qs, K_sin.transpose(-1, -2))
            s = sc + ss
            if self.is_decoder:
                rows = torch.arange(start, end, device=device)
                cols = torch.arange(N, device=device)
                allowed = rows.unsqueeze(1) >= cols.unsqueeze(0)
                s = s.masked_fill(~allowed, 0.0)
            scores = torch.clamp(s, min=0.0, max=1e4)
            num = torch.matmul(scores, V)
            denom = safe_linear_attn_denominator(scores.sum(dim=-1, keepdim=True), eps=1e-5, floor_val=1e-4)
            res = num / denom
            out[:, :, start:end] = torch.nan_to_num(res, nan=0.0, posinf=1e4, neginf=-1e4)
        return out
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                position_ids=None, position_embeddings=None, *args, **kwargs):
        
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        input_dtype = hidden_states.dtype

        # Cast projections to float32 immediately so backward gradients stay in fp32 range.
        # Without this, Q_raw/K_raw are fp16, and the grad from _local_attention (fp32)
        # overflows fp16 on the way back, producing NaN that propagates to all trainable params.
        Q_raw = self.q_proj(hidden_states).float().view(B, N, H, d).transpose(1, 2).contiguous()
        K_raw = self.k_proj(hidden_states).float().view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_global = self.v_proj(hidden_states).float().view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_local = self.v_proj_attn(hidden_states).float().view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        if self.num_kv_heads != self.num_heads:
            K_raw = repeat_kv(K_raw, self.kv_repeat)
            V_global = repeat_kv(V_global, self.kv_repeat)
            V_local = repeat_kv(V_local, self.kv_repeat)

        scale = 1.0 / math.sqrt(d)
        Q_scaled = torch.clamp(Q_raw * scale, min=-50.0, max=50.0)
        K_clamped = torch.clamp(K_raw, min=-50.0, max=50.0)
        
        if self.act_fun == "softplus":
            Q_k = F.softplus(Q_scaled.float())
            K_k = F.softplus(K_clamped.float())
        else:
            Q_k = F.elu(Q_scaled.float()) + 1.0
            K_k = F.elu(K_clamped.float()) + 1.0

        cos_w, sin_w = self._get_cos_sin_weights(N, device, dtype=torch.float32)
        Q_cos = Q_k * cos_w
        Q_sin = Q_k * sin_w
        K_cos = K_k * cos_w
        K_sin = K_k * sin_w

        padding_mask = self._build_padding_mask(
            attention_mask, N, device, dtype=torch.float32,
        )

        global_out = self._forward_causal(
            Q_cos, Q_sin, K_cos, K_sin, V_global.float(),
            padding_mask, B, H, d, N, device, torch.float32,
        )

        local_out = self._local_attention(
            torch.clamp(Q_raw.float(), min=-1e4, max=1e4),
            torch.clamp(K_raw.float(), min=-1e4, max=1e4),
            torch.clamp(V_local.float(), min=-1e4, max=1e4),
            padding_mask, N, device, torch.float32,
        )
        local_out = torch.nan_to_num(local_out, nan=0.0, posinf=1e4, neginf=-1e4)

        # Zero-init to match GPT-2 exactly at step 0, avoiding KD shock.
        context = local_out + self.fusion_scale.to(torch.float32) * global_out
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        context = torch.clamp(context, min=-65500.0, max=65500.0).to(input_dtype)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = self.dropout(context)
        final_out = self.out_proj(context)
        final_out = torch.nan_to_num(final_out, nan=0.0, posinf=65500.0, neginf=-65500.0)

        return final_out.to(input_dtype), None


def inject_cosformer(layer=None, model=None, model_config=None, comp_config=None,
                     layer_idx=0, total_layers=12, old_attn=None, **kwargs):
    depth = _get_inject_depth()
    if depth >= _MAX_DEPTH: return None
    _set_inject_depth(depth + 1)

    try:
        target = layer if layer is not None else model
        if target is None: return None

        _BLOCK_PATHS = (
            ("transformer", "h"),
            ("encoder", "layer"),
            ("model", "layers"),
            ("model", "decoder", "layers"),
            ("decoder", "layers"),
        )
        _ATTN_ATTRS = ("attn", "attention", "self_attn")
        _INNER_ATTRS = ("self", "attention")

        layer_list = None
        for path in _BLOCK_PATHS:
            obj = target
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok and isinstance(obj, (list, nn.ModuleList)):
                layer_list = obj
                break

        if layer_list is not None:
            num_injected = 0
            for idx, lyr in enumerate(layer_list):
                res = inject_cosformer(
                    layer=lyr, model_config=model_config,
                    comp_config=comp_config, layer_idx=idx, **kwargs,
                )
                if res is not None:
                    num_injected += 1
            if num_injected == 0:
                logger.warning(
                    "inject_cosformer: found block list but injected 0 layers."
                )
                return None
            return target

        container, inner, src_attn = None, None, old_attn
        for attr in _ATTN_ATTRS:
            sub = getattr(target, attr, None)
            if sub is None:
                continue
            for inner_attr in _INNER_ATTRS:
                inner_sub = getattr(sub, inner_attr, None)
                if inner_sub is not None and not isinstance(inner_sub, nn.ModuleList):
                    container, inner = sub, inner_attr
                    if src_attn is None:
                        src_attn = inner_sub
                    break
            if container is None:
                container, inner = target, attr
                if src_attn is None:
                    src_attn = sub
            break

        if container is None:
            logger.warning(
                "inject_cosformer: no attention attribute on %s",
                type(target).__name__,
            )
            return None

        new_attn = CosformerAttention(
            config=model_config, comp_config=comp_config,
            layer_idx=layer_idx, old_attn=src_attn, **kwargs,
        )

        if inner:
            setattr(container, inner, new_attn)
        else:
            setattr(target, container, new_attn)
        return new_attn
        
    finally:
        _set_inject_depth(depth)

__all__ = ["CosformerAttention", "inject_cosformer", "apply_rotary_pos_emb", "LoRALinear"]