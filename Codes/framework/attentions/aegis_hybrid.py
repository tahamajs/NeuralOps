"""
Aegis-Hybrid Attention: Zero-Shot Optimized Production Version.
Stabilized for Inference/Evaluation by bypassing untrained global linear paths
and utilizing clean, chunked sliding-window attention.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    old_attn: Optional[nn.Module],
) -> bool:
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


@dataclass
class AegisHybridConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: Optional[int] = None
    local_window: int = 512  # FIX: Increased window for better zero-shot context
    rank: int = 128
    lora_rank: int = 32
    dropout: float = 0.1
    use_adaptive_pruning: bool = False  # FIX: Disabled for zero-shot stability
    gate_init_bias: float = 1.8
    is_decoder: bool = False
    use_global: bool = False   # FIX: Disabled untrained global path for zero-shot eval
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class FactorizedLinear(nn.Module):
    is_custom_attention = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = min(int(lora_rank), min(self.in_features, self.out_features)) if lora_rank > 0 else 0

        factory_kwargs = {"device": device, "dtype": dtype}
        self.lin1 = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.lin2 = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)

        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lin1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lin2.weight, a=math.sqrt(5))
            if self.lin2.bias is not None:
                nn.init.zeros_(self.lin2.bias)
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if self.lora_A is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.lin2.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin2(self.lin1(x))
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out


def _local_window_attention_chunked(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: Optional[torch.Tensor],
    is_decoder: bool,
    past_len: int,
    dropout_p: float,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = Q.shape
    N_kv = K.shape[2]
    radius = window // 2
    scale = D ** -0.5
    device = Q.device

    abs_q = past_len + torch.arange(N, device=device)
    abs_k = torch.arange(N_kv, device=device)
    chunk = max(64, min(radius * 2, 256))

    outs = []
    attn_blocks = []

    for q_start in range(0, N, chunk):
        q_end = min(q_start + chunk, N)
        Qc = Q[:, :, q_start:q_end]
        q_abs = abs_q[q_start:q_end]

        q_min = int(q_abs.min().item())
        q_max = int(q_abs.max().item())
        k_lo = max(0, q_min - radius)
        k_hi = min(N_kv, q_max + radius + 1)

        if k_hi <= k_lo:
            outs.append(torch.zeros(B, H, q_end - q_start, D, device=device, dtype=Q.dtype))
            continue

        Kc = K[:, :, k_lo:k_hi]
        Vc = V[:, :, k_lo:k_hi]
        k_abs = abs_k[k_lo:k_hi]

        scores = torch.matmul(Qc.float(), Kc.float().transpose(-2, -1)) * scale

        band = (q_abs.unsqueeze(1) - k_abs.unsqueeze(0)).abs() <= radius
        if is_decoder:
            band = band & (k_abs.unsqueeze(0) <= q_abs.unsqueeze(1))
        band = band.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(~band, -1e4)

        if token_valid is not None:
            kv_valid = token_valid[:, k_lo:k_hi][:, None, None, :].bool()
            scores = scores.masked_fill(~kv_valid, -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        any_valid = band.any(dim=-1, keepdim=True).to(scores.dtype)
        attn = F.softmax(scores, dim=-1).to(Q.dtype) * any_valid
        attn = F.dropout(attn, p=dropout_p, training=training)

        outs.append(torch.matmul(attn, Vc))
        if not training:
            attn_blocks.append((q_start, q_end, k_lo, k_hi, attn))

    output = torch.cat(outs, dim=2)
    
    attn_full = None
    if len(attn_blocks) > 0 and N * N_kv <= 4_000_000:
        attn_full = torch.zeros(B, H, N, N_kv, device=device, dtype=Q.dtype)
        for q_s, q_e, k_l, k_h, blk in attn_blocks:
            attn_full[:, :, q_s:q_e, k_l:k_h] = blk

    return output, attn_full


@ATTENTION_REGISTRY.register("aegis_hybrid", supports_causal=True, supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_hybrid_attn", supports_causal=True, supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_hybrid_attention", supports_causal=True, supports_gqa=True, supports_rope=True)
class AegisHybridAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: Optional[int] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        if not isinstance(config, AegisHybridConfig):
            cfg_src = config if config is not None else kwargs
            h_size = hidden_size or getattr(cfg_src, "hidden_size", 768)
            n_heads = num_heads or getattr(cfg_src, "num_attention_heads", 12)
            n_kv = getattr(cfg_src, "num_key_value_heads", n_heads)
            h_dim = getattr(cfg_src, "head_dim", h_size // max(1, n_heads))
            r = rank or getattr(cfg_src, "rank", min(128, h_size))
            l_r = lora_rank or getattr(cfg_src, "lora_rank", 32)
            
            dec = _detect_decoder(config, is_decoder, pretrained_attn)
            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                num_key_value_heads=n_kv,
                local_window=getattr(cfg_src, "local_window", 256),
                rank=r,
                lora_rank=l_r,
                is_decoder=dec,
                use_global=False,  # FIX: Forced to False for Zero-Shot stability
            )

        self.config = config
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.num_kv_heads: int = config.num_key_value_heads
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.local_window: int = config.local_window
        self.rank: int = max(1, min(config.rank, self.hidden_size))
        self.lora_rank: int = max(1, min(config.lora_rank, self.hidden_size))
        self.dropout = nn.Dropout(config.dropout)

        self.is_decoder: bool = _detect_decoder(config, is_decoder, pretrained_attn)
        self.use_global: bool = False  # FIX: Disabled global path
        self.is_nested_encoder: bool = False
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.q_proj = FactorizedLinear(self.hidden_size, self.all_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.k_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.v_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.out_proj = FactorizedLinear(self.all_head_size, self.hidden_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

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

    @staticmethod
    def _svd_init(w: torch.Tensor, b: Optional[torch.Tensor], proj: FactorizedLinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.lin1.weight.data.zero_()
            proj.lin2.weight.data.zero_()
            in_copy = min(W.shape[1], proj.lin1.weight.size(1))
            out_copy = min(W.shape[0], proj.lin2.weight.size(0))
            proj.lin1.weight.data[:r, :in_copy].copy_(Vh[:r, :in_copy].to(proj.lin1.weight.dtype))
            US = U[:out_copy, :r] @ torch.diag(S[:r])
            proj.lin2.weight.data[:out_copy, :r].copy_(US.to(proj.lin2.weight.dtype))
            if b is not None and proj.lin2.bias is not None:
                n = min(b.numel(), proj.lin2.bias.numel())
                proj.lin2.bias.data[:n].copy_(b.detach()[:n].to(proj.lin2.bias.dtype))
            elif proj.lin2.bias is not None:
                proj.lin2.bias.data.zero_()

    def copy_weights(self, old_attn: nn.Module) -> None:
        from framework.core.adapters import UniversalAdapter
        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}

        for p in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, p)
                if w is not None:
                    extracted[p] = (w, b)
            except Exception:
                pass

        if hasattr(old_attn, "c_attn") and extracted["q"][0] is None:
            c_w = old_attn.c_attn.weight.data.t().contiguous()
            split = c_w.size(0) // 3
            c_b = getattr(old_attn.c_attn, "bias", None)
            c_b = c_b.data if c_b is not None else None
            extracted["q"] = (c_w[:split, :], c_b[:split] if c_b is not None else None)
            extracted["k"] = (c_w[split:2*split, :], c_b[split:2*split] if c_b is not None else None)
            extracted["v"] = (c_w[2*split:, :], c_b[2*split:] if c_b is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            w_o = old_attn.c_proj.weight.data.t().contiguous()
            b_o = getattr(old_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        for proj, target in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)):
            w, b = extracted[proj]
            if w is not None:
                self._svd_init(w, b, target)

    @staticmethod
    def _prepare_valid_mask(attention_mask, B, N_kv, device):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask.bool() if attention_mask.dtype == torch.bool else attention_mask > -1e4
        return (attention_mask > -1e4).squeeze(1) if attention_mask.dim() == 3 else attention_mask.any(dim=-2).squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        layer_idx: int = 0,
        **kwargs: Any,
    ):
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        Dh = self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, Dh).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        # KV cache handling
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        K_cat = torch.cat([past_k, K_new], dim=-2) if past_k is not None and not is_cross else K_new
        V_cat = torch.cat([past_v, V_new], dim=-2) if past_v is not None and not is_cross else V_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(K_new, V_new, layer_idx)

        N_kv = K_cat.shape[2]
        past_len = N_kv - N if (self.is_decoder and not is_cross) else 0

        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        if past_len > 0 and token_valid is not None:
            past_valid = torch.ones(B, past_len, device=device, dtype=torch.bool)
            token_valid = torch.cat([past_valid, token_valid], dim=1)

        # Pure Local Window Attention (Stable for Zero-Shot)
        local_out, attn_local = _local_window_attention_chunked(
            Q, K_full, V_full,
            window=self.local_window,
            token_valid=token_valid,
            is_decoder=self.is_decoder and not is_cross,
            past_len=past_len,
            dropout_p=self.config.dropout,
            training=self.training,
        )

        out = local_out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        final = self.out_proj(out)
        
        return final, (attn_local if output_attentions else None)


@ATTENTION_REGISTRY.register("aegis_hybrid_inject")
def inject_aegis_hybrid(layer=None, model=None, model_config=None, comp_config=None, layer_idx=0, old_attn=None, **kwargs):
    target = layer if layer is not None else model
    if target is None:
        return None
    cfg = model_config if model_config is not None else getattr(target, "config", None)
    
    # Simple single-layer injection wrapper
    src = old_attn
    if src is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if cand is not None:
                    src = getattr(cand, "self", cand)
                    break
    
    h_size = getattr(cfg, "hidden_size", 768)
    n_heads = getattr(cfg, "num_attention_heads", 12)
    
    config = AegisHybridConfig(hidden_size=h_size, num_attention_heads=n_heads)
    new_attn = AegisHybridAttention(config=config, pretrained_attn=src)
    
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            target.attention.self = new_attn
        else:
            target.attention = new_attn
    elif hasattr(target, "self_attn"):
        target.self_attn = new_attn
    elif hasattr(target, "attn"):
        target.attn = new_attn
        
    return new_attn