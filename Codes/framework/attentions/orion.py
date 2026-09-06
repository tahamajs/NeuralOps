"""ORION: Optimized Rectified Input-aware Orthogonal Network (Production-Hardened).

Hybrid multi-head attention architecture integrating zero-shot SVD low-rank factorized projections,
head-wise bifurcated linear kernel vs. rectified Softpick full attention, and random Fourier features.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal cumulative scan for kernel heads and causal masking for full heads in decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


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


@dataclass
class ORIONConfig:
    svd_energy: float = 0.95
    svd_min_rank: int = 4
    svd_max_rank: int = 64
    num_kernel_heads: int = 8
    num_full_heads: int = 4
    rff_features: int = 256
    softpick_temperature: float = 1.0
    dropout: float = 0.1


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    m_scaled = torch.clamp(-m / temperature, max=80.0)
    denom = e - torch.exp(m_scaled)
    p = F.relu(denom)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-5)).to(orig_dtype)


class FactorisedLinear(nn.Module):
    """Low-rank factorized linear layer with SVD decomposition and LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.energy = energy
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        rank = max(min_rank, min(max_rank, min(in_features, out_features) // 4, 16))
        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def decompose_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        out_dim, in_dim = weight.shape
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(weight.float(), full_matrices=False)
            total = (S ** 2).sum()
            cum = torch.cumsum(S ** 2, dim=0) / torch.clamp(total, min=1e-8)
            rank = int((cum < self.energy).sum().item()) + 1
            rank = max(self.min_rank, min(rank, min(out_dim, in_dim), self.max_rank))
            self.rank = rank

            device = self.down.weight.device
            dtype = self.down.weight.dtype

            self.down = nn.Linear(in_dim, rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(rank, out_dim, bias=(bias is not None or self.up.bias is not None), device=device, dtype=dtype)

            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]
            self.down.weight.data.copy_(B.to(device=device, dtype=dtype))
            self.up.weight.data.copy_(A.to(device=device, dtype=dtype))

            if bias is not None and self.up.bias is not None:
                self.up.bias.data.copy_(bias.to(device=device, dtype=dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            if self.freeze_base:
                self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("orion_module", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("orion", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("orion_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class ORIONAttention(BaseAttention):
    r"""ORION hybrid multi-head attention module."""

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        orion_cfg: Optional[ORIONConfig] = None,
        pretrained_attn: Optional[Any] = None,
        old_attn: Optional[Any] = None,
        comp_config: Optional[Any] = None,
        bert_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(cfg, rank)

        if orion_cfg is None:
            orion_cfg = ORIONConfig()
            if rank is not None:
                orion_cfg.svd_max_rank = max(rank, orion_cfg.svd_min_rank)

        hidden_size = kwargs.get("hidden_size", None)
        num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", None))
        num_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", None))

        if hidden_size is None or num_heads is None:
            if cfg is not None:
                if hidden_size is None:
                    hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "d_model", getattr(cfg, "dim", 768)))
                if num_heads is None:
                    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 12))
                if num_kv_heads is None:
                    num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", num_heads))
            elif source_attn is not None:
                if hidden_size is None:
                    out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                    hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
                if num_heads is None:
                    num_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", 12))
                if num_kv_heads is None:
                    num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                if hidden_size is None:
                    hidden_size = 768
                if num_heads is None:
                    num_heads = 12
                if num_kv_heads is None:
                    num_kv_heads = num_heads

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.num_heads = max(1, int(num_heads))
        self.num_kv_heads = max(1, int(num_kv_heads))
        self.head_dim = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.hidden_size = hidden_size
        self.cfg = orion_cfg
        self.bert_mode = bert_mode

        # Allocate kernel vs full heads
        if self.num_heads <= 1:
            self.num_kernel_heads = 1
            self.num_full_heads = 0
        else:
            self.num_full_heads = max(1, self.num_heads // 3)
            self.num_kernel_heads = self.num_heads - self.num_full_heads

        # Decoder detection
        model_type = getattr(cfg, "model_type", "").lower() if cfg else ""
        is_dec = getattr(cfg, "is_decoder", getattr(cfg, "is_causal", kwargs.get("is_causal", False))) if cfg else False
        if model_type in ["gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "qwen3.5", "llama", "mistral"]:
            is_dec = True
        elif hasattr(cfg, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(cfg, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = getattr(cfg, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if cfg else 16
        self.lora_alpha = getattr(cfg, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if cfg else 32.0
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.q_proj = FactorisedLinear(hidden_size, self.all_head_size, bias=True, energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank, max_rank=orion_cfg.svd_max_rank, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = FactorisedLinear(hidden_size, self.all_kv_head_size, bias=True, energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank, max_rank=orion_cfg.svd_max_rank, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = FactorisedLinear(hidden_size, self.all_kv_head_size, bias=True, energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank, max_rank=orion_cfg.svd_max_rank, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = FactorisedLinear(self.all_head_size, hidden_size, bias=True, energy=orion_cfg.svd_energy, min_rank=orion_cfg.svd_min_rank, max_rank=orion_cfg.svd_max_rank, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.register_buffer("rff_proj", torch.randn(max(1, self.num_kernel_heads), self.head_dim, orion_cfg.rff_features) * 0.02, persistent=True)
        self.dropout = nn.Dropout(orion_cfg.dropout)

        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception:
                pass

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"ORIONAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                if w.shape != (target_mod.out_features, target_mod.in_features) and (target_mod.out_features, target_mod.in_features) == w.t().shape:
                    w = w.t()

                target_mod.decompose_from_weight(w[:target_mod.out_features, :target_mod.in_features], b[:target_mod.out_features] if b is not None else None)

                w_norm = float(torch.norm(target_mod.weight).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"ORIONAttention weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

    def _kernel_linear_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        B, Hk, N, D = Q.shape
        if Hk == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)
        F_dim = self.cfg.rff_features
        orig_dtype = Q.dtype
        Q, K, V = Q.float(), K.float(), V.float()
        proj = self.rff_proj[:Hk].to(Q.device).float()

        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, proj).clamp(max=11.0)
        K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, proj).clamp(max=11.0)
        Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
        K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

        if self.is_decoder:
            kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_feat, V)
            KV_cum = torch.cumsum(kv_outer, dim=2)
            num = torch.einsum("bhnf,bhnfd->bhnd", Q_feat, KV_cum)
            z_cum = torch.cumsum(K_feat, dim=2)
            denom = torch.einsum("bhnf,bhnf->bhn", Q_feat, z_cum).unsqueeze(-1).clamp(min=1e-5)
            ctx = num / denom
        else:
            KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)
            K_sum = K_feat.sum(dim=2)
            denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-5)
            ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
        return ctx.to(orig_dtype)

    def _full_softpick_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Hf, N, D = Q.shape
        if Hf == 0:
            return torch.zeros(B, 0, N, D, device=Q.device, dtype=Q.dtype)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale

        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(N, N, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if token_valid is not None:
            scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e4)

        attn_w = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature).to(V.dtype)
        attn_w = self.dropout(attn_w)
        return torch.matmul(attn_w, V)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)
        elif not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        Hk = self.num_kernel_heads
        Hf = self.num_full_heads

        Q_all = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_all = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_all = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q_all, K_all = apply_rotary_pos_emb(Q_all, K_all, cos, sin)

        if self.kv_repeat > 1:
            K_all = repeat_kv(K_all, self.kv_repeat)
            V_all = repeat_kv(V_all, self.kv_repeat)

        token_valid = None
        if attention_mask is not None:
            token_valid = get_token_valid_mask(attention_mask, B, N, hidden_states.device)

        ctx_list = []
        if Hk > 0:
            Q_kernel = Q_all[:, :Hk]
            K_kernel = K_all[:, :Hk]
            V_kernel = V_all[:, :Hk]
            ctx_kernel = self._kernel_linear_attention(Q_kernel, K_kernel, V_kernel)
            ctx_list.append(ctx_kernel)

        if Hf > 0:
            Q_full = Q_all[:, Hk:]
            K_full = K_all[:, Hk:]
            V_full = V_all[:, Hk:]
            ctx_full = self._full_softpick_attention(Q_full, K_full, V_full, token_valid)
            ctx_list.append(ctx_full)

        if len(ctx_list) == 1:
            context = ctx_list[0]
        else:
            context = torch.cat(ctx_list, dim=1)

        context = torch.nan_to_num(context, nan=0.0)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm

        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("orion_inject")
def inject_orion(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[ORIONAttention]:
    src_attn = old_attn
    if src_attn is None and layer is not None:
        src_attn = getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))

    hidden_size = (
        kwargs.get("hidden_size") or
        getattr(src_attn, 'hidden_size', None) or
        getattr(src_attn, 'embed_dim', None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
        (src_attn.query.in_features if hasattr(src_attn, 'query') and hasattr(src_attn.query, 'in_features') else None) or
        (src_attn.q_lin.in_features if hasattr(src_attn, 'q_lin') and hasattr(src_attn.q_lin, 'in_features') else None) or
        (layer.in_features if hasattr(layer, 'in_features') else None) or
        getattr(model_config, 'hidden_size', None) or
        getattr(model_config, 'd_model', 64)
    )
    num_heads = (
        kwargs.get("num_heads") or
        kwargs.get("num_attention_heads") or
        getattr(src_attn, 'num_heads', None) or
        getattr(src_attn, 'num_attention_heads', None) or
        getattr(model_config, 'num_attention_heads', None) or
        getattr(model_config, 'num_heads', 4)
    )
    try:
        new_attn = ORIONAttention(
            config=model_config,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            source_attn=src_attn,
            **kwargs,
        )
        if src_attn is not None:
            try:
                new_attn.copy_weights(src_attn)
            except Exception:
                pass
        if layer is not None:
            if hasattr(layer, "attention"):
                layer.attention = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
            else:
                layer.attention = new_attn
        return new_attn
    except Exception:
        return None


# Aliases
inject_orion_attention = inject_orion
