"""
Griffin SSD: Hybrid Local Sliding-Window Attention and Global State Space Duality.
Production-Hardened v3 (SIMPLIFIED):
  - Only Python loop for SSD (no chunked scan, no shape bugs)
  - decay_param init = 2.0 (was 0 → 0.5)
  - ssd_proj = 0.5 * W_q (was = W_q)
  - Learnable fusion gate (was direct addition)
  - Symmetric q/k normalization
  - Env var toggles for ablation
"""

from __future__ import annotations
from neuralops.core.common import _env_flag, _env_float, _unpack_weight_bias

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
)

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.gated_delta_net import LoRALinear
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)










@ATTENTION_REGISTRY.register("griffin_ssd")
class GriffinSSDBlock(BaseAttention):
    """Griffin Hybrid Block: local window + SSD linear attention."""

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        window_size: int = 64,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        def _get(k: str, default: Any = None) -> Any:
            for obj in (comp_config, config, getattr(config, "compression_config", None)):
                if obj is not None:
                    if isinstance(obj, dict) and k in obj and obj[k] is not None:
                        return obj[k]
                    elif hasattr(obj, k):
                        val = getattr(obj, k)
                        if val is not None:
                            return val
            return kwargs.get(k, default)

        if d_model is None:
            d_model = _get("hidden_size", _get("d_model", _get("n_embd", _get("dim", None))))
        if d_model is None and old_attn is not None:
            d_model = (
                getattr(old_attn, "hidden_size", None)
                or getattr(old_attn, "embed_dim", None)
                or getattr(old_attn, "all_head_size", None)
            )
            if d_model is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(old_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        d_model = m.in_features
                        break
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        if num_heads is None:
            num_heads = (
                _get("num_attention_heads", None) or _get("num_heads", None)
                or _get("n_heads", None) or _get("n_head", None)
            )
        if num_heads is None and old_attn is not None:
            num_heads = getattr(old_attn, "num_heads", getattr(old_attn, "num_attention_heads", None))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim = int(d_model)
        self.num_heads = int(num_heads)

        resolved_head_dim = _get("head_dim", None)
        if resolved_head_dim is None and old_attn is not None:
            resolved_head_dim = getattr(old_attn, "head_dim", None)
            if resolved_head_dim is None:
                for q_cand in ("q_proj", "q_lin", "query", "W_q"):
                    m = getattr(old_attn, q_cand, None)
                    if m is not None and hasattr(m, "out_features"):
                        resolved_head_dim = m.out_features // max(1, self.num_heads)
                        break
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim = int(resolved_head_dim)

        kv_heads = _get("num_key_value_heads", None)
        if kv_heads is None and old_attn is not None:
            kv_heads = getattr(
                old_attn, "num_key_value_heads",
                getattr(old_attn, "num_kv_heads", getattr(old_attn, "n_kv_heads", None)),
            )
            if (kv_heads is None and hasattr(old_attn, "k_proj")
                    and hasattr(old_attn.k_proj, "out_features") and self.head_dim > 0):
                kv_heads = old_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads = int(kv_heads if kv_heads is not None else self.num_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.window_size = int(_get("window_size", kwargs.get("window_size", window_size)))

        model_type = str(_get("model_type", "")).lower()
        is_dec = _get("is_decoder", _get("is_causal", kwargs.get("is_causal", kwargs.get("is_decoder", False))))
        if (any(model_type.startswith(pfx) for pfx in (
                "gpt", "qwen", "llama", "mistral", "bloom", "opt", "falcon", "codegen"
            ))
            or (hasattr(config, "architectures") and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])
            ))):
            is_dec = True
        if old_attn is not None and hasattr(old_attn, "c_attn"):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder

        is_nested = False
        if old_attn is not None:
            src_cls = old_attn.__class__.__name__
            if (src_cls in (
                    "BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention",
                    "ElectraSelfAttention", "DebertaSelfAttention",
                )
                or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())
                or (hasattr(old_attn, "self") and hasattr(old_attn, "output"))):
                is_nested = True
        self.is_nested_encoder = bool(is_nested)

        self.lora_rank = int(_get("lora_rank", kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha = float(_get("lora_alpha", kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base = self.lora_rank > 0

        self.W_q = LoRALinear(self.dim, self.all_head_size, bias=False,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.W_k = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.ssd_proj = LoRALinear(self.dim, self.all_head_size, bias=False,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        _decay_init = _env_float("GRIFFIN_DECAY_INIT", 2.0)
        self.decay_param = nn.Parameter(torch.full((self.num_heads, 1, 1), _decay_init))

        _gate_init = _env_float("GRIFFIN_GATE_INIT", 0.0)
        self.fusion_gate = nn.Parameter(torch.full((self.num_heads,), _gate_init))

        self.rotary_emb = getattr(old_attn, "rotary_emb", None) if old_attn is not None else None
        self.comp_config = comp_config

        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}

        try:
            adapter = UniversalAdapter()
            for proj in ["q", "k", "v", "o"]:
                w, b = adapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
        except Exception:
            pass

        if hasattr(old_attn, "c_attn") and extracted["q"][0] is None:
            c_w, c_b = _unpack_weight_bias(old_attn.c_attn)
            if c_w is not None:
                if c_w.dim() == 2 and c_w.shape[1] == 3 * self.dim:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2*split_size, :], c_b[split_size:2*split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2*split_size:, :], c_b[2*split_size:] if c_b is not None else None)
            if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
                o_w, o_b = _unpack_weight_bias(old_attn.c_proj)
                if o_w is not None:
                    if o_w.dim() == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim:
                        o_w = o_w.t().contiguous()
                    extracted["o"] = (o_w, o_b)
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            self.is_nested_encoder = True
            s = old_attn.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                extracted[proj_key] = _unpack_weight_bias(getattr(s, sub_name, None))
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                extracted["o"] = _unpack_weight_bias(old_attn.output.dense)
        else:
            candidates = {
                "q": ["q_proj", "q_lin", "query"],
                "k": ["k_proj", "k_lin", "key"],
                "v": ["v_proj", "v_lin", "value"],
                "o": ["out_proj", "out_lin", "o_proj", "dense"],
            }
            for proj_key, names in candidates.items():
                if extracted[proj_key][0] is None:
                    for name in names:
                        if hasattr(old_attn, name):
                            cw, cb = _unpack_weight_bias(getattr(old_attn, name))
                            if cw is not None:
                                extracted[proj_key] = (cw, cb)
                                break

        target_map = {"q": self.W_q, "k": self.W_k, "v": self.W_v, "o": self.out_proj}
        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    continue
                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (w.shape[0] < target_mod.weight.shape[0]
                      and target_mod.weight.shape[0] % w.shape[0] == 0):
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and getattr(target_mod, "bias", None) is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif (b.shape[0] < target_mod.bias.shape[0]
                          and target_mod.bias.shape[0] % b.shape[0] == 0):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))

            if extracted["q"][0] is not None:
                w_q = extracted["q"][0]
                if w_q.shape == self.ssd_proj.weight.shape:
                    self.ssd_proj.weight.copy_(w_q * 0.5)
                elif w_q.t().contiguous().shape == self.ssd_proj.weight.shape:
                    self.ssd_proj.weight.copy_(w_q.t().contiguous() * 0.5)
                else:
                    target_w = self.ssd_proj.weight
                    min_r = min(target_w.shape[0], w_q.shape[0])
                    min_c = min(target_w.shape[1], w_q.shape[1])
                    target_w.data.zero_()
                    target_w.data[:min_r, :min_c].copy_(w_q[:min_r, :min_c] * 0.5)

        if self.freeze_base:
            for mod in [self.W_q, self.W_k, self.W_v, self.ssd_proj, self.out_proj]:
                mod.freeze_base_weights()

    def _ssd_scan_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """
        Chunked parallel scan for causal SSD.
        
        Mathematical equivalence:
            y_t = q_t @ state_t
            state_t = decay * state_{t-1} + k_t^T @ v_t
        
        Chunked decomposition:
            y_t = q_t @ state_{chunk_start} * decay^(t - chunk_start) 
                + sum_{j=chunk_start}^{t} decay^(t-j) * (q_t @ k_j^T) @ v_j
        """
        B, H, L, d = q.shape
        target_dtype = q.dtype

        # Symmetric normalization (FIX 7)
        q_n = F.normalize(q.float(), p=2, dim=-1)               # (B, H, L, d)
        k_n = F.normalize(k.float(), p=2, dim=-1) / math.sqrt(d) # (B, H, L, d)
        v_f = v.float()                                          # (B, H, L, d)

        # decay: (1, H, 1, 1)
        decay_f = decay.to(torch.float32).view(1, H, 1, 1)
        # log decay for stable exponentiation
        log_decay = torch.log(decay_f.clamp(min=1e-6, max=1.0 - 1e-6))  # (1, H, 1, 1)

        # ---- Pad to multiple of chunk_size ----
        chunk_size = min(chunk_size, L)
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        Lp = L + pad_len
        if pad_len > 0:
            q_n = F.pad(q_n, (0, 0, 0, pad_len))
            k_n = F.pad(k_n, (0, 0, 0, pad_len))
            v_f = F.pad(v_f, (0, 0, 0, pad_len))

        n_chunks = Lp // chunk_size

        # ---- Reshape to (B, H, n_chunks, chunk_size, d) ----
        q_c = q_n.view(B, H, n_chunks, chunk_size, d)
        k_c = k_n.view(B, H, n_chunks, chunk_size, d)
        v_c = v_f.view(B, H, n_chunks, chunk_size, d)

        # ---- Intra-chunk: parallel attention within each chunk with decay ----
        # Positions within chunk
        pos = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
        # diff[i,j] = i - j (causal: i >= j)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (C, C) where diff[i, j] = i - j
        causal = diff >= 0  # (C, C) bool

        # D_intra[i,j] = decay^(i-j) if i>=j else 0
        log_decay_5d = log_decay.view(1, H, 1, 1, 1)
        diff_5d = diff.view(1, 1, 1, chunk_size, chunk_size)  # (1,1,1,C,C)
        exp_arg = (diff_5d * log_decay_5d).clamp(min=-40.0, max=0.0)  # (1,H,1,C,C)
        D_intra = torch.exp(exp_arg) * causal.view(1, 1, 1, chunk_size, chunk_size).float()

        # scores_intra[b,h,c,i,j] = q_c[b,h,c,i] · k_c[b,h,c,j]
        # q_c: (B,H,nc,C,d), k_c^T: (B,H,nc,d,C) → scores: (B,H,nc,C,C)
        scores_intra = torch.matmul(q_c, k_c.transpose(-1, -2))  # (B,H,nc,C,C)
        scores_intra = scores_intra * D_intra  # broadcast (1,H,1,C,C) → (B,H,nc,C,C) ✓

        # y_intra[b,h,c] = scores_intra[b,h,c] @ v_c[b,h,c]
        y_intra = torch.matmul(scores_intra, v_c)  # (B,H,nc,C,d)

        # ---- Inter-chunk: parallel scan over n_chunks ----
        # Each chunk's final state: state_after_chunk[c] = decay^C * state_before + sum of kv in chunk
        # Contribution of chunk c to next chunks: k_v_chunk[c] = sum_{i=0}^{C-1} decay^(C-1-i) * k_c[i]^T @ v_c[i]
        # Actually simpler: state contribution = k_c^T @ v_c (matrix) with proper decay weights
        
        # For chunk c, kv_contrib[c] = sum_{i=0}^{C-1} decay^(C-1-i) * (k_c[i]^T @ v_c[i])
        # decay^j where j = C-1-i
        pos_in_chunk = torch.arange(chunk_size, device=q.device, dtype=torch.float32)
        # decay_weights[i] = decay^(C-1-i)
        decay_in_chunk = torch.exp(
            ((chunk_size - 1 - pos_in_chunk).view(1, 1, 1, chunk_size, 1) * log_decay_5d).clamp(min=-40.0, max=0.0)
        )  # (1,H,1,C,1)

        # k_c: (B,H,nc,C,d), v_c: (B,H,nc,C,d)
        # weighted k: (B,H,nc,C,d) * (1,H,1,C,1) → (B,H,nc,C,d)
        k_weighted = k_c * decay_in_chunk  # (B,H,nc,C,d)
        
        # kv_contrib[c] = k_weighted[c]^T @ v_c[c]
        # (B,H,nc,d,C) @ (B,H,nc,C,d) → (B,H,nc,d,d)
        kv_contrib = torch.matmul(k_weighted.transpose(-1, -2), v_c)  # (B,H,nc,d,d)

        # Now parallel scan over chunks with chunk_decay = decay^chunk_size
        chunk_decay = torch.exp((log_decay.view(1, H, 1, 1) * chunk_size).clamp(min=-40.0, max=0.0))

        # states_before[c] = state before processing chunk c
        # states_before[0] = 0
        # states_before[c] = chunk_decay * states_before[c-1] + kv_contrib[c-1]
        states_before = torch.zeros(B, H, n_chunks, d, d, device=q.device, dtype=torch.float32)
        running = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
        for c in range(n_chunks):
            states_before[:, :, c] = running
            running = running * chunk_decay + kv_contrib[:, :, c]

        # ---- Inter-chunk output ----
        # y_inter[b,h,c,i] = q_c[b,h,c,i] @ states_before[b,h,c] * decay^(i+1)
        # Actually: q at position i in chunk c sees state_before[c] with decay^(i+1)?
        # Let's be precise: 
        # state_before[c] = sum_{c'<c} decay^(chunk_size * (c-1-c')) * kv_contrib[c']
        # For query at position i in chunk c, the contribution from past chunks is:
        #   q_i @ state_before[c] * decay^(i) — the (i) accounts for positions within current chunk
        # Wait, more carefully:
        # state after processing chunk c-1 is states_before[c]
        # At position i in chunk c, we've had i decay steps since start of chunk
        # state_at_position_i = decay^i * state_before[c] + sum_{j=0}^{i-1} decay^(i-1-j) * k[c,j]^T v[c,j]
        # The first term: query sees decay^i * state_before[c]
        # The second term is captured by y_intra
        
        # query positions: pos (C,)
        # decay^i for i in [0, C-1]
        q_decay = torch.exp((pos.view(1, 1, 1, chunk_size, 1) * log_decay_5d).clamp(min=-40.0, max=0.0))  # (1,H,1,C,1)

        # q_c: (B,H,nc,C,d), states_before: (B,H,nc,d,d)
        # q_c @ states_before: (B,H,nc,C,d)
        q_state = torch.matmul(q_c, states_before)  # (B,H,nc,C,d)
        y_inter = q_state * q_decay  # (1,H,1,C,1) broadcast → (B,H,nc,C,d) ✓

        # ---- Combine ----
        y = y_intra + y_inter  # (B,H,nc,C,d)
        y = y.view(B, H, Lp, d)[:, :, :L, :]

        # Apply mask to query output (FIX 6)
        if mask is not None:
            m = mask.view(B, 1, L, 1).float()
            y = y * m

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).to(target_dtype)
    def _ssd_symmetric(self, q, k, v, decay, mask=None):
        """Bidirectional symmetric SSD for encoders."""
        B, H, L, d = q.shape
        target_dtype = q.dtype

        q_n = F.normalize(q.float(), p=2, dim=-1)
        k_n = F.normalize(k.float(), p=2, dim=-1) / math.sqrt(d)

        decay_f = decay.to(torch.float32).view(1, H, 1, 1)
        log_decay = torch.log(decay_f.clamp(min=1e-6, max=1.0 - 1e-6))

        pos = torch.arange(L, device=q.device, dtype=torch.float32)
        dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))  # (L, L)
        D_sym = torch.exp((dist.unsqueeze(0).unsqueeze(0) * log_decay).clamp(min=-40.0, max=0.0))

        if mask is not None:
            m = mask.view(B, 1, L, 1).to(D_sym.dtype)
            D_sym = D_sym * m.transpose(-1, -2) * m

        scores = torch.matmul(q_n, k_n.transpose(-1, -2)) * D_sym
        y = torch.matmul(scores, v.float())

        if mask is not None:
            m = mask.view(B, 1, L, 1).to(y.dtype)
            y = y * m

        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).to(target_dtype)

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
    ):
        if encoder_hidden_states is not None:
            return hidden_states, None

        mask = kwargs.get("mask")
        if attention_mask is None and mask is not None:
            attention_mask = mask

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        target_dtype = hidden_states.dtype
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.W_q(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous().to(target_dtype)
        k = (self.W_k(hidden_states).view(B, L, self.num_kv_heads, d)
             .transpose(1, 2).contiguous().to(target_dtype))
        v = (self.W_v(hidden_states).view(B, L, self.num_kv_heads, d)
             .transpose(1, 2).contiguous().to(target_dtype))
        q_ssd = (self.ssd_proj(hidden_states).view(B, L, H, d)
                 .transpose(1, 2).contiguous().to(target_dtype))

        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        pos_ids = kwargs.get("position_ids")
        if pos_emb is not None:
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
                cos, sin = pos_emb[0], pos_emb[1]
            else:
                cos, sin = pos_emb, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)
        elif getattr(self, "rotary_emb", None) is not None:
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            cos, sin = None, None
            try:
                cos, sin = self.rotary_emb(q, seq_len=L)
            except Exception:
                try:
                    cos, sin = self.rotary_emb(q, pos_ids)
                except Exception:
                    try:
                        cos, sin = self.rotary_emb(q)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # Autoregressive KV Cache for decoders
        if self.is_decoder and past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                if past_k is not None and past_v is not None:
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if self.is_decoder else None

        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_exp = repeat_kv(k, repeats)
            v_exp = repeat_kv(v, repeats)
        else:
            k_exp = k
            v_exp = v

        # Local sliding window
        scores = torch.matmul(q, k_exp.transpose(-1, -2).contiguous()) / math.sqrt(d)
        L_kv = k_exp.shape[-2]
        idx_q = torch.arange(L, device=hidden_states.device).unsqueeze(1)
        idx_k = torch.arange(L_kv, device=hidden_states.device).unsqueeze(0)
        dist = torch.abs(idx_q - idx_k)
        window_mask = dist > self.window_size
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        if self.is_decoder:
            past_len = L_kv - L
            causal = torch.tril(torch.ones(L, L_kv, device=hidden_states.device, dtype=torch.bool), diagonal=past_len)
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -10000.0)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype.is_floating_point:
                    mask_bool = attention_mask > -1e4
                else:
                    mask_bool = attention_mask.bool()
                pad_mask = (~mask_bool).view(B, 1, 1, L)
                scores = scores.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        scores = torch.nan_to_num(scores, nan=-10000.0, posinf=10000.0, neginf=-10000.0)
        local_attn = F.softmax(scores, dim=-1).to(target_dtype)
        local_attn = torch.nan_to_num(local_attn, nan=0.0)
        local_out = torch.matmul(local_attn, v_exp).to(target_dtype)

        # Global SSD
        if _env_flag("GRIFFIN_DISABLE_SSD", False):
            global_out = torch.zeros_like(local_out)
        else:
            decay = torch.sigmoid(self.decay_param.to(target_dtype)).view(1, H, 1, 1)
            k_ssd = k[:, :, -L:, :] if k.shape[-2] > L else k
            v_ssd = v[:, :, -L:, :] if v.shape[-2] > L else v
            if self.num_heads != self.num_kv_heads:
                k_ssd = repeat_kv(k_ssd, repeats)
                v_ssd = repeat_kv(v_ssd, repeats)
            if self.is_decoder:
                global_out = self._ssd_scan_chunked(q_ssd, k_ssd, v_ssd, decay, mask_bool, chunk_size=32)
            else:
                global_out = self._ssd_symmetric(q_ssd, k_ssd, v_ssd, decay, mask_bool)

        # Fusion gate
        gate = torch.sigmoid(self.fusion_gate).view(1, H, 1, 1).to(target_dtype)
        fused = (gate * local_out + (1.0 - gate) * global_out).to(target_dtype)

        fused = fused.transpose(1, 2).contiguous().view(B, L, self.all_head_size).to(target_dtype)
        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused).to(target_dtype)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None and mask_bool.shape[-1] == L:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        if self.is_decoder:
            return (out.to(target_dtype), present_key_value)
        else:
            return (out.to(target_dtype), None)


def inject_griffin_ssd(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Any | None:
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    if target is not None and (
        hasattr(target, "transformer") or hasattr(target, "distilbert")
        or hasattr(target, "roberta") or hasattr(target, "bert")
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
                res = inject_griffin_ssd(
                    layer=lyr, model_config=cfg, comp_config=comp_config,
                    layer_idx=idx, total_layers=tot, lora_rank=lora_rank, **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    curr_attn = old_attn
    if curr_attn is None and target is not None:
        if hasattr(target, "attention"):
            curr_attn = target.attention.self if hasattr(target.attention, "self") else target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
        elif hasattr(target, "attn"):
            curr_attn = target.attn
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn

    if curr_attn is None and target is not None and hasattr(target, "forward"):
        curr_attn = target

    if curr_attn is None and cfg is None:
        return None

    if isinstance(curr_attn, GriffinSSDBlock) or "GriffinSSD" in curr_attn.__class__.__name__:
        return None

    try:
        new_layer = GriffinSSDBlock(config=cfg, old_attn=curr_attn, lora_rank=lora_rank, **kwargs)
    except Exception as ex:
        logger.warning("Failed to instantiate GriffinSSDBlock: %s", ex)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_layer.is_nested_encoder = True
                target.attention.self = new_layer
            else:
                new_layer.is_nested_encoder = False
                target.attention = new_layer
        elif hasattr(target, "self"):
            new_layer.is_nested_encoder = True
            target.self = new_layer
        elif hasattr(target, "attn"):
            new_layer.is_nested_encoder = False
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            new_layer.is_nested_encoder = False
            target.self_attn = new_layer

    return new_layer


__all__ = [
    "GriffinSSDBlock",
    "inject_griffin_ssd",
    "apply_rotary_pos_emb",
]
