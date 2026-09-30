"""
VSpec++ Attention: Variational Spectral Token-Adaptive Attention.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _masked_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes mean over valid non-padded tokens."""
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return x.mean(dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions: [-x2, x1].
    Robust to odd head_dim by rotating only the even slice 2 * (d // 2).
    """
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings safely handling even or odd dimensions."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.dtype)

    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed

    return q, k


class VariationalSpectralGate(nn.Module):
    r"""Input-dependent variational gate over truncated singular values with masked pooling."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, rank * 2)
        )
        # Initialize gate to neutral (near 1.0) so SVD representations are preserved at step 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:rank], 3.0)   # mu -> sigmoid(3.0) ≈ 0.95
        nn.init.constant_(self.net[-1].bias[rank:], -3.0)  # log_sigma -> low variance

    def forward(
        self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        net_dtype = next(self.net.parameters()).dtype
        pooled = _masked_mean(x, valid_mask).to(net_dtype)
        stats = self.net(pooled)
        mu, log_sigma = stats.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=2.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(log_sigma)
        else:
            z = mu
        g = torch.sigmoid(z).to(x.dtype)
        return g, mu, log_sigma


class HeadGate(nn.Module):
    r"""Learned per-example head importance scalars with masked pooling."""

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_heads)
        )
        # Initialize head gate to neutral 1.0 (bias=3.0 -> sigmoid(3.0) ≈ 0.95)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 3.0)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        net_dtype = next(self.net.parameters()).dtype
        pooled = _masked_mean(x, valid_mask).to(net_dtype)
        return torch.sigmoid(self.net(pooled)).to(x.dtype)


class TokenRouter(nn.Module):
    r"""Top-k token selector based on learned saliency scores with guaranteed [CLS] retention."""

    def __init__(self, hidden_dim: int, keep_ratio: float = 0.75, min_tokens: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        is_decoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        net_dtype = next(self.net.parameters()).dtype
        scores = torch.sigmoid(self.net(x.to(net_dtype)).squeeze(-1))  # (B, N)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)

        if is_decoder and N > 1:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask[0], -10000.0)

        # Guarantee preservation of first token (index 0)
        scores[:, 0] = 1e4

        keep = min(
            N,
            max(
                1,
                min(self.min_tokens, N)
                if N <= self.min_tokens
                else max(self.min_tokens, int(N * self.keep_ratio)),
            ),
        )
        idx = scores.topk(keep, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)

        routed = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return routed, idx


class DynamicLowRankProjector(nn.Module):
    is_custom_attention: bool = True
    r"""SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters."""

    def __init__(
        self,
        rank: int = 32,
        in_dim: int = 768,
        out_dim: int = 768,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.rank = rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(1, lora_rank)

        self.U = nn.Parameter(torch.randn(out_dim, rank, **factory_kwargs) * 0.02)
        self.S = nn.Parameter(torch.ones(rank, **factory_kwargs))
        self.V = nn.Parameter(torch.randn(in_dim, rank, **factory_kwargs) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_dim), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_dim, lora_rank), **factory_kwargs))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def set_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            eff_rank = min(self.rank, U.size(1), Vh.size(0))

            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()

            min_u = min(self.U.size(0), U.size(0))
            min_v = min(self.V.size(0), Vh.size(1))

            self.U.data[:min_u, :eff_rank].copy_(U[:min_u, :eff_rank])
            self.S.data[:eff_rank].copy_(S[:eff_rank])
            self.V.data[:min_v, :eff_rank].copy_(Vh[:eff_rank, :min_v].t().contiguous())

            if bias is not None and self.bias is not None:
                b_val = bias.data if isinstance(bias, nn.Parameter) else bias
                min_b = min(self.bias.size(0), b_val.size(0))
                self.bias.data[:min_b].copy_(b_val[:min_b])

            rec = (self.U[:, :eff_rank] * self.S[:eff_rank].unsqueeze(0)) @ self.V[:, :eff_rank].t().contiguous()
            f_norm = float(torch.linalg.norm(rec))
            if f_norm < 1e-4:
                logger.warning(f"DynamicLowRankProjector: reconstructed norm is near zero ({f_norm}), re-initializing randomly")
                nn.init.orthogonal_(self.U.data)
                nn.init.orthogonal_(self.V.data)
                self.S.data.fill_(1.0)

    def copy_lora_from(self, sub: Any) -> None:
        """Preserve source LoRA adapters."""
        if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
            src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
            src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
            if hasattr(src_lA, "data") and self.lora_A is not None:
                r_min = min(self.lora_A.shape[0], src_lA.shape[0])
                in_min = min(self.lora_A.shape[1], src_lA.shape[1])
                self.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                self.lora_A.requires_grad = True
            if hasattr(src_lB, "data") and self.lora_B is not None:
                out_min = min(self.lora_B.shape[0], src_lB.shape[0])
                r_min = min(self.lora_B.shape[1], src_lB.shape[1])
                self.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                self.lora_B.requires_grad = True
            if hasattr(sub, "lora_scaling"):
                self.scaling = sub.lora_scaling

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        v = self.V.to(x.dtype)
        s = self.S.to(x.dtype)
        u = self.U.to(x.dtype)
        g = gate.to(x.dtype)
        z = torch.matmul(x, v)
        z = z * s.unsqueeze(0).unsqueeze(0) * g.unsqueeze(1)
        out = torch.matmul(z, u.t().contiguous())
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            la = self.lora_A.to(x.dtype)
            lb = self.lora_B.to(x.dtype)
            lora_out = F.linear(F.linear(x, la), lb) * self.scaling
            out = out + lora_out

        return out


@ATTENTION_REGISTRY.register("vspec", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("vspec_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class VSpecAttention(BaseAttention):
    r"""VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module."""

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        cfg_head_dim = _cfg_get(config, "head_dim", kwargs.get("head_dim", None))
        if cfg_head_dim is None and src_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(src_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    num_h = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
                    cfg_head_dim = mod.weight.shape[0] // max(1, num_h)
                    break

        _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_dec = (
            getattr(config, "is_decoder", False)
            or getattr(config, "causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        ) if config else False
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True

        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.hidden_size: int = self.adapter.hidden_size
            self.head_dim: int = int(cfg_head_dim) if cfg_head_dim is not None else self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = is_dec
        else:
            self.adapter = None
            self.hidden_size = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = int(cfg_head_dim) if cfg_head_dim is not None else (self.hidden_size // max(1, self.num_heads))
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = is_dec

        # Explicit overrides
        if "hidden_size" in kwargs:
            self.hidden_size = kwargs["hidden_size"]
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
        if "head_dim" in kwargs:
            self.head_dim = kwargs["head_dim"]
        elif cfg_head_dim is not None:
            self.head_dim = int(cfg_head_dim)
        else:
            self.head_dim = self.hidden_size // max(1, self.num_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        self.is_causal = self.is_decoder

        self.is_nested_encoder: bool = False
        self.rank: int = rank
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and src_attn is not None:
            target_dtype = next(src_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        self.Pq = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.hidden_size, out_dim=self.all_head_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.Pk = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.hidden_size, out_dim=self.kv_head_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.Pv = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.hidden_size, out_dim=self.kv_head_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.Po = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.all_head_size, out_dim=self.hidden_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )

        self.sg_q = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_k = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_v = VariationalSpectralGate(self.hidden_size, self.rank)

        self.head_gate = HeadGate(self.hidden_size, self.num_heads)
        self.router = TokenRouter(self.hidden_size, keep_ratio=0.75)

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        if src_attn is not None:
            self.copy_weights(src_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    @property
    def query(self): return self.Pq
    @property
    def key(self): return self.Pk
    @property
    def value(self): return self.Pv
    @property
    def q_proj(self): return self.Pq
    @property
    def k_proj(self): return self.Pk
    @property
    def v_proj(self): return self.Pv
    @property
    def out_proj(self): return self.Po
    @property
    def o_proj(self): return self.Po
    @property
    def dense(self): return self.Po
    @property
    def c_proj(self): return self.Po

    def copy_weights(self, src: Any) -> None:
        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            c_mod = src.c_attn
            base_c = getattr(c_mod, "base_layer", c_mod)
            weight = getattr(base_c, "weight", getattr(c_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                is_conv1d = (
                    "Conv1D" in c_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                )
                if is_conv1d:
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                q_w = weight[:split_size, :]
                k_w = weight[split_size : 2 * split_size, :]
                v_w = weight[2 * split_size :, :]
                q_b = bias_data[:split_size] if bias_data is not None else None
                k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
                v_b = bias_data[2 * split_size :] if bias_data is not None else None
                extracted["q"] = (q_w, q_b)
                extracted["k"] = (k_w, k_b)
                extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            p_mod = src.c_proj
            base_o = getattr(p_mod, "base_layer", p_mod)
            w_o = getattr(base_o, "weight", getattr(p_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    "Conv1D" in p_mod.__class__.__name__
                    or "Conv1D" in base_o.__class__.__name__
                    or hasattr(p_mod, "nf")
                    or hasattr(base_o, "nf")
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                )
                if is_conv1d:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(p_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        base = getattr(layer, "base_layer", layer)
                        if hasattr(base, "weight"):
                            extracted[proj] = (base.weight.data, getattr(base, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = hasattr(src, "dense") or hasattr(src, "out_proj") or hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (hasattr(src, "query") and not has_out)
        ):
            is_nested = True
        if hasattr(src, "self") and hasattr(src, "output"):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("VSpecAttention missing required weights for projections: %s; keeping defaults", missing)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy into projectors
        if extracted["q"][0] is not None:
            self.Pq.set_weight(_orient(extracted["q"][0], self.hidden_size), extracted["q"][1])
        if extracted["k"][0] is not None:
            self.Pk.set_weight(_orient(extracted["k"][0], self.hidden_size), extracted["k"][1])
        if extracted["v"][0] is not None:
            self.Pv.set_weight(_orient(extracted["v"][0], self.hidden_size), extracted["v"][1])

        if extracted["o"][0] is not None and not self.is_nested_encoder:
            self.Po.set_weight(_orient(extracted["o"][0], self.all_head_size), extracted["o"][1])
        elif self.is_nested_encoder:
            eye_w = torch.eye(self.all_head_size, self.all_head_size)
            self.Po.set_weight(eye_w, None)

        # Preserve source LoRA adapters if present
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.Pq), ("k", self.Pk), ("v", self.Pv), ("o", self.Po)]:
            for name in source_submodules[proj_key]:
                if hasattr(src, name):
                    sub = getattr(src, name)
                    target.copy_lora_from(sub)
                    break

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        if encoder_hidden_states is not None or kwargs.get("encoder_hidden_states", None) is not None:
            raise NotImplementedError("Cross-attention is not implemented for VSpecAttention")

        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Mask handling
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        valid = get_token_valid_mask(mask, B, N, device) if mask is not None else None

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "causal", False)
            or any(
                str(getattr(self.config, "model_type", "") or "").lower().startswith(p)
                for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
            )
        )

        # Token routing: Skip for nested encoders (RoBERTa/BERT) to preserve context
        if self.is_nested_encoder:
            routed = hidden_states
            idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            k_len = N
            valid_routed = valid
        else:
            routed, idx = self.router(hidden_states, valid, is_decoder=is_decoder)
            k_len = routed.shape[1]
            valid_routed = torch.gather(valid, 1, idx) if valid is not None else None

        # Variational spectral gates using masked mean over routed tokens
        gq, _, _ = self.sg_q(routed, valid_routed)
        gk, _, _ = self.sg_k(routed, valid_routed)
        gv, _, _ = self.sg_v(routed, valid_routed)

        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)

        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        Q = Q.view(B, k_len, H, Dh).transpose(1, 2).contiguous()
        K = K.view(B, k_len, H_kv, Dh).transpose(1, 2).contiguous()
        V = V.view(B, k_len, H_kv, Dh).transpose(1, 2).contiguous()

        # Rotary position embeddings
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            cos_b = cos.expand(B, -1, -1) if cos.size(0) == 1 else cos
            sin_b = sin.expand(B, -1, -1) if sin.size(0) == 1 else sin
            cos_routed = torch.gather(cos_b, 1, idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
            sin_routed = torch.gather(sin_b, 1, idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
            Q, K = apply_rotary_pos_emb(Q, K, cos_routed, sin_routed)
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                Q, K = maybe_apply_rotary(Q, K, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
            except Exception as e:
                logger.warning(f"maybe_apply_rotary failed: {e}")

        # GQA expansion
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        hg = self.head_gate(routed, valid_routed).unsqueeze(-1).unsqueeze(-1)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2).contiguous()) * scale * hg.float()

        # Check causal decoder
        if is_decoder and k_len > 1:
            causal_mask = torch.triu(
                torch.full((k_len, k_len), float("-inf"), device=device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply padding mask to attention scores over routed tokens
        if valid_routed is not None:
            pad_mask = (~valid_routed).unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, k_len, k_len)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0).to(dtype)
        attn_weights = self.dropout(attn_weights)

        # Head mask broadcasting
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                if head_mask.shape[-1] == head_mask.shape[-2] and head_mask.shape[-1] != attn_weights.shape[-1]:
                    head_mask = head_mask.mean(dim=-1, keepdim=True)
            attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, k_len, self.all_head_size)

        if self.is_nested_encoder:
            return (context, (attn_weights if output_attentions else None))

        out = self.Po(context, gq)

        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1])) if out.shape[-1] < D else out[..., :D]

        # Scatter back to full sequence length preserving residual:
        # Non-routed tokens have zero attention delta, so outer block residual keeps them unchanged.
        if k_len == N:
            output = out
        else:
            zeros = torch.zeros(B, N, D, device=device, dtype=out.dtype)
            output = zeros.scatter(1, idx.unsqueeze(-1).expand(-1, -1, D), out)

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("vspec_inject")
@ATTENTION_REGISTRY.register("vspec_attention_inject")
@ATTENTION_REGISTRY.register("inject_vspec")
@ATTENTION_REGISTRY.register("inject_vspec_attn")
def inject_vspec_attn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[VSpecAttention, nn.Module]]:
    """Safe injection hook for VSpecAttention across diverse transformer architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_vspec_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    try:
        src = old_attn
        if src is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "self"):
                        src = cand.self
                    else:
                        src = cand
                    break

        if src is None and target is not None and hasattr(target, "forward"):
            src = target

        if src is None:
            return None

        # Double-injection guard
        if src is not None:
            if isinstance(src, VSpecAttention):
                return None
            if "VSpec" in src.__class__.__name__:
                return None
            if hasattr(src, "inner") and isinstance(getattr(src, "inner"), VSpecAttention):
                return None
            if hasattr(src, "_wrapped_attention") and isinstance(getattr(src, "_wrapped_attention"), VSpecAttention):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, VSpecAttention):
                    return None
                if "VSpec" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), VSpecAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), VSpecAttention):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src is not None:
            c_name = src.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        hidden_size = (
            _cfg_get(cfg, "hidden_size")
            or _cfg_get(cfg, "d_model")
            or _cfg_get(cfg, "dim")
            or getattr(src, "hidden_size", None)
            or getattr(src, "embed_dim", None)
            or (src.q_proj.in_features if hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features") else None)
            or (src.q_lin.in_features if hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features") else None)
            or (src.query.in_features if hasattr(src, "query") and hasattr(src.query, "in_features") else None)
            or 768
        )
        num_heads = (
            _cfg_get(cfg, "num_attention_heads")
            or _cfg_get(cfg, "num_heads")
            or _cfg_get(cfg, "n_heads")
            or getattr(src, "num_heads", None)
            or getattr(src, "num_attention_heads", None)
            or 12
        )

        rank = kwargs.pop("rank", _cfg_get(comp_config, "vspec_rank", _cfg_get(comp_config, "rank", 32)))
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

        new_block = VSpecAttention(
            config=cfg,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_block
                    new_block.is_nested_encoder = True
                else:
                    target.attention = new_block
                    new_block.is_nested_encoder = False
                return new_block
            for attr in ("self_attn", "attn"):
                if hasattr(target, attr):
                    setattr(target, attr, new_block)
                    new_block.is_nested_encoder = False
                    return new_block
            if hasattr(target, "self"):
                target.self = new_block
                new_block.is_nested_encoder = True
                return new_block
            target.attention = new_block
            new_block.is_nested_encoder = False
            return new_block
        return new_block
    except Exception as e:
        logger.error("inject_vspec_attn failed at layer %d: %s", layer_idx, e)
        return None
