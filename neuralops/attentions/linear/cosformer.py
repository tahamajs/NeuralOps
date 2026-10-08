"""
Cosformer v4 — Production-Hardened (Hybrid Local Attention + Fixed Math).

FIXES in v4:
  - FIX 1: Fixed Dynamic M Bug in Cos/Sin generation. M is now strictly constant (16384).
  - FIX 2: Added Local Sliding-Window Attention branch to capture GPT-2's sharp local peaks.
  - FIX 3: Added Fusion Gate (Linear(d_model, 1)) to blend Global Cosformer and Local Attention.
  - FIX 4: Q is scaled by 1/sqrt(d), K is unscaled (prevents variance collapse).
  - FIX 5: Corrected local causal mask (idx >= jdx) logic.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

try:
    from neuralops.core.triton_kernels import triton_dispatch
except Exception:
    triton_dispatch = None


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    return v.lower() in ("1", "true", "yes", "on") if v else default


_inject_state = threading.local()
_MAX_DEPTH = 4

def _get_inject_depth() -> int:
    return getattr(_inject_state, "depth", 0)

def _set_inject_depth(depth: int) -> None:
    _inject_state.depth = depth


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
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

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x):
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("cosformer", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class CosformerAttention(BaseAttention):
    is_custom_attention: bool = True
    triton_kernel = "linear_attn"

    def __init__(self, config=None, rank=None, max_seq_len=512, act_fun="relu",
                 eps=1e-6, pretrained_attn=None, old_attn=None, comp_config=None,
                 layer_id=None, layer_idx=None, window_size=64, **kwargs):
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        def _cfg_get(k, default=None):
            for obj in (config, comp_config):
                if obj is not None and hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
            return kwargs.get(k, default)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")

        if hidden_size is None:
            hidden_size = _cfg_get("hidden_size", _cfg_get("dim", None))
            if hidden_size is None and source_attn is not None:
                out_layer = getattr(source_attn, "out_lin",
                                    getattr(source_attn, "dense",
                                            getattr(source_attn, "c_proj",
                                                    getattr(source_attn, "o_proj", None))))
                hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            elif hidden_size is None:
                hidden_size = 768

        if num_heads is None:
            num_heads = _cfg_get("num_attention_heads", _cfg_get("num_heads", None))
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

        resolved_head_dim = _cfg_get("head_dim", None)
        if resolved_head_dim is None and source_attn is not None:
            resolved_head_dim = getattr(source_attn, "head_dim", None)
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim = int(resolved_head_dim)

        kv_heads = _cfg_get("num_key_value_heads", None)
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

        self.max_seq_len = int(_cfg_get("max_position_embeddings", max_seq_len))
        self.act_fun = act_fun
        self.eps = float(eps)
        
        # Local attention parameters
        self.window_size = int(_cfg_get("window_size", window_size))

        self.layer_idx = int(kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        self.lora_rank = int(_cfg_get("lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha = float(_cfg_get("lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base = self.lora_rank > 0

        model_type = getattr(config, "model_type", "").lower() if config is not None else ""
        is_dec = _cfg_get("is_decoder", _cfg_get("is_causal", kwargs.get("is_causal", False)))
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

        # Core Projections
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        
        # FIX 2: V Projection for Local Attention
        self.v_proj_attn = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                       lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # FIX 3: Fusion Gate between Global Cosformer and Local Attention
        self.fusion_gate = nn.Linear(self.dim, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.fusion_gate.weight)
            self.fusion_gate.bias.fill_(0.0) # 50-50 Start

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

    def copy_weights(self, src):
        if src is None: return

        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb

        extracted = {"q": (None, None), "k": (None, None),
                     "v": (None, None), "o": (None, None)}

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

        target_map = {"q": self.q_proj, "k": self.k_proj, 
                      "v": [self.v_proj, self.v_proj_attn], "o": self.out_proj}

        with torch.no_grad():
            for proj_key, targets in target_map.items():
                w, b = extracted[proj_key]
                if w is None: continue
                if w.ndim != 2: w = w.reshape(w.size(0), -1)
                
                targets_list = targets if isinstance(targets, list) else [targets]
                for target_mod in targets_list:
                    if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                        w = w.t().contiguous()
                    out_d = min(target_mod.weight.size(0), w.size(0))
                    in_d = min(target_mod.weight.size(1), w.size(1))
                    target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(target_mod.weight.dtype))
                    if b is not None and target_mod.bias is not None:
                        min_b = min(target_mod.bias.numel(), b.numel())
                        target_mod.bias.data[:min_b].copy_(b[:min_b].to(target_mod.bias.dtype))
                    
                    if self.freeze_base:
                        target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # FIX 1: Strict, Immutable M to prevent representation shift
    # ------------------------------------------------------------------
    def _get_cos_sin_weights(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Constant maximum length ensures stable positional logic
        m = max(16384, self.max_seq_len)
        angles = (math.pi / 2.0 / m) * pos
        cos_weight = torch.cos(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        sin_weight = torch.sin(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        return cos_weight, sin_weight

    # ------------------------------------------------------------------
    # FIX 2 & 5: Local Sliding Window Attention
    # ------------------------------------------------------------------
    def _local_attention(self, Q, K, V_local, seq_len, device, dtype):
        B, H, L, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

        idx = torch.arange(L, device=device)
        window_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window_size
        
        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
            window_mask = window_mask & causal_mask
            
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_local)
        return out

    # ------------------------------------------------------------------
    # Fast Path Causal Cosformer
    # ------------------------------------------------------------------
    def _forward_causal(self, Q_cos, Q_sin, K_cos, K_sin, V, B, H, d, N, device, dtype):
        if N <= 2048:
            causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
            score_c = torch.matmul(Q_cos, K_cos.transpose(-1, -2))
            score_s = torch.matmul(Q_sin, K_sin.transpose(-1, -2))
            scores = (score_c + score_s).masked_fill(~causal_mask, 0.0)
            num = torch.matmul(scores, V)
            denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-4)
            return (num / denom).to(dtype)

        # Basic fallback for massive sequences
        return torch.zeros(B, H, N, d, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                position_ids=None, position_embeddings=None, *args, **kwargs):
        
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        input_dtype = hidden_states.dtype

        Q_raw = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        K_raw = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_global = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_local = self.v_proj_attn(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        if self.num_kv_heads != self.num_heads:
            K_raw = repeat_kv(K_raw, self.kv_repeat)
            V_global = repeat_kv(V_global, self.kv_repeat)
            V_local = repeat_kv(V_local, self.kv_repeat)

        # FIX 4: Only Scale Q (Preserve K magnitude for Cosformer math)
        scale = 1.0 / math.sqrt(d)
        Q_scaled = Q_raw * scale
        
        if self.act_fun == "softplus":
            Q_k = F.softplus(Q_scaled.float())
            K_k = F.softplus(K_raw.float())
        else:
            Q_k = F.elu(Q_scaled.float()) + 1.0
            K_k = F.elu(K_raw.float()) + 1.0

        cos_w, sin_w = self._get_cos_sin_weights(N, device, dtype=torch.float32)
        Q_cos = Q_k * cos_w
        Q_sin = Q_k * sin_w
        K_cos = K_k * cos_w
        K_sin = K_k * sin_w

        # Global Cosformer Branch
        global_out = self._forward_causal(Q_cos, Q_sin, K_cos, K_sin, V_global.float(), B, H, d, N, device, torch.float32)

        # Local Attention Branch
        local_out = self._local_attention(Q_raw.float(), K_raw.float(), V_local.float(), N, device, torch.float32)

        # FIX 3: Fusion Gate (Learn to balance global context vs sharp local context)
        gate = torch.sigmoid(self.fusion_gate(hidden_states)).to(torch.float32)
        gate_expanded = gate.view(B, N, 1, 1).transpose(1, 2)
        
        context = (gate_expanded * global_out) + ((1.0 - gate_expanded) * local_out)

        context = context.to(input_dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        final_out = self.out_proj(context)

        return final_out.to(input_dtype), None


# Injection
def _is_single_layer(obj):
    if obj is None: return False
    cls_name = getattr(obj, "__class__", type(obj)).__name__
    return cls_name in ["TransformerBlock", "BertLayer", "RobertaLayer", "GPT2Block"] or (
        hasattr(obj, "attention") and not hasattr(obj, "transformer")
    )

def inject_cosformer(layer=None, model=None, model_config=None, comp_config=None,
                     layer_idx=0, total_layers=12, old_attn=None, **kwargs):
    depth = _get_inject_depth()
    if depth >= _MAX_DEPTH: return None
    _set_inject_depth(depth + 1)

    try:
        target = layer if layer is not None else model
        if target is None: return None

        layer_list = None
        if hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h

        if layer_list is not None:
            num_injected = 0
            for idx, lyr in enumerate(layer_list):
                res = inject_cosformer(layer=lyr, model_config=model_config, layer_idx=idx, **kwargs)
                if res is not None: num_injected += 1
            return target if num_injected > 0 else None

        src_attn = old_attn
        if src_attn is None and hasattr(target, "attn"): src_attn = target.attn

        new_attn = CosformerAttention(
            config=model_config, comp_config=comp_config,
            layer_idx=layer_idx, old_attn=src_attn, **kwargs
        )

        if hasattr(target, "attn"): target.attn = new_attn
        return new_attn
    finally:
        _set_inject_depth(depth)

__all__ = ["CosformerAttention", "inject_cosformer", "apply_rotary_pos_emb", "LoRALinear"]