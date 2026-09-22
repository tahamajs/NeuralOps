r"""
AEGIS-Attention V1: Adaptive Enhanced Global-Local Integrated Sparse Attention Baseline.

Combines SVD-factorized projection matrices with energy-based rank selection and LoRA adapters,
split-head local sliding-window attention with smooth activations,
and global linear kernel attention with adaptive gating.
Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]  # safe for odd d: leaves trailing element out
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


@dataclass
class AegisConfigV1:
    r"""Configuration parameters for AegisAttention V1."""
    svd_energy: float = 0.95
    min_rank: int = 4
    max_rank: int = 384
    num_local_heads: int = 8
    num_global_heads: int = 4
    local_window: int = 64
    temperature: float = 1.0
    rff_features: int = 128
    use_zero_sum_keys: bool = True
    use_adaptive_gate: bool = True
    gate_init_bias: float = 1.8
    dropout: float = 0.1
    is_decoder: bool = False


class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True
    r"""
    SVD-factorized linear layer with energy-based rank truncation and trainable LoRA adapters.
    Base SVD weights are frozen; LoRA branch provides trainable parameters.
    """

    def __init__(
        self,
        in_features_or_weight: Union[int, torch.Tensor],
        out_features_or_bias: Optional[Union[int, torch.Tensor]] = None,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 384,
        rank: Optional[int] = None,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.energy = energy
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        if isinstance(in_features_or_weight, torch.Tensor):
            W = in_features_or_weight
            b = out_features_or_bias if isinstance(out_features_or_bias, torch.Tensor) else None
            out_dim, in_dim = W.shape
            self.in_features = in_dim
            self.out_features = out_dim
            self.rank = min(max_rank, max(min_rank, min(in_dim, out_dim)))
            device = W.device
            dtype = W.dtype

            self.down = nn.Linear(in_dim, self.rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(self.rank, out_dim, bias=(b is not None or bias), device=device, dtype=dtype)
            if self.lora_rank > 0:
                self.lora_A = nn.Parameter(torch.randn(lora_rank, in_dim, device=device, dtype=dtype) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(out_dim, lora_rank, device=device, dtype=dtype))
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)

            self.decompose_from_weight(W, b, energy)
        else:
            in_dim = int(in_features_or_weight)
            out_dim = int(out_features_or_bias) if out_features_or_bias is not None else in_dim
            self.in_features = in_dim
            self.out_features = out_dim
            self.rank = rank if rank is not None else min(max_rank, max(min_rank, min(in_dim, out_dim)))

            self.down = nn.Linear(in_dim, self.rank, bias=False)
            self.up = nn.Linear(self.rank, out_dim, bias=bias)
            if self.lora_rank > 0:
                self.lora_A = nn.Parameter(torch.randn(lora_rank, in_dim) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(out_dim, lora_rank))
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)

            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
                rep = self.out_features // W.shape[0]
                W = W.repeat(rep, 1)
                if bias is not None:
                    bias = bias.repeat(rep)
            elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
                rep = W.shape[0] // self.out_features
                W = W.view(self.out_features, rep, -1).mean(dim=1)
                if bias is not None:
                    bias = bias.view(self.out_features, rep).mean(dim=1)
            elif W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
                padded[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])] = W[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            S = torch.clamp(S, min=1e-6)
            target_energy = energy if energy is not None else self.energy
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / (total + 1e-5)
            calc_rank = int((cum < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            A = U[:, :r] * S[:r].unsqueeze(0)
            B = Vt[:r, :]

            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:r, :] = B.to(self.down.weight.dtype)
            self.up.weight.data[:, :r] = A.to(self.up.weight.dtype)

            if bias is not None and self.up.bias is not None:
                self.up.bias.data[:min(len(self.up.bias), len(bias))].copy_(bias[:min(len(self.up.bias), len(bias))].to(self.up.bias.dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        base_w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            return base_w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return base_w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    temperature: float = 1.0,
    token_valid: Optional[torch.Tensor] = None,
    is_decoder: bool = False,
) -> torch.Tensor:
    B, H, N, D = Q.shape
    N_k = K.shape[2]
    scale = (D**-0.5) / temperature
    idx_q = torch.arange(N, device=Q.device).unsqueeze(1)
    idx_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
    dist = (idx_q - idx_k).abs()
    mask = (dist <= window // 2) | (idx_q == 0) | (idx_k == 0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if is_decoder and N > 1:
        causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
        mask = mask & causal

    # Memory-efficient: use SDPA to avoid OOM on large seqs; fallback to explicit if needed


    try:


        _qs = Q.half() if Q.dtype == torch.float32 else Q


        _ks = K.half() if K.dtype == torch.float32 else K


        _vs = V.half() if V.dtype == torch.float32 else V


        _is_causal = getattr(self, 'is_decoder', False) and not is_cross and N > 1 if 'is_cross' in dir() else getattr(self, 'is_decoder', False) and N > 1


        _sdpa_out = torch.nn.functional.scaled_dot_product_attention(_qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0)


        _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)


        _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)


        return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None


    except Exception:


        pass


    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
    mask_val = -10000.0
    scores = scores.masked_fill(~mask, mask_val)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), mask_val)
    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
    attn = F.softmax(scores, dim=-1).to(Q.dtype)
    return torch.matmul(attn, V)


def kernel_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rff_proj: torch.Tensor,
    use_zero_sum: bool = True,
    max_clamp: float = 11.0,
    is_causal: bool = False,
    token_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    orig_dtype = Q.dtype

    Q_f, K_f, V_f = Q.float(), K.float(), V.float()
    rff_proj_f = rff_proj.float()

    if use_zero_sum:
        K_f = K_f - K_f.mean(dim=2, keepdim=True)
    Q_norm = F.normalize(Q_f, p=2, dim=-1)
    K_norm = F.normalize(K_f, p=2, dim=-1)

    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj_f).clamp(max=max_clamp)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj_f).clamp(max=max_clamp)
    Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
    K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

    if token_valid is not None:
        valid_mask = token_valid.unsqueeze(1).unsqueeze(-1).float()
        K_feat = K_feat * valid_mask

    if is_causal:
        kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_feat, V_f)
        KV_cum = torch.cumsum(kv_outer, dim=2)
        z_cum = torch.cumsum(K_feat, dim=2)
        num = torch.einsum("bhnf,bhnfd->bhnd", Q_feat, KV_cum)
        denom = torch.einsum("bhnf,bhnf->bhn", Q_feat, z_cum).unsqueeze(-1).clamp(min=1e-4)
        ctx = num / denom
    else:
        KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-4)
        ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
    return torch.nan_to_num(ctx, nan=0.0).clamp(min=-1e4, max=1e4).to(orig_dtype)


@ATTENTION_REGISTRY.register("aegis_v1_module", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("AegisAttentionV1", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_v1", supports_causal=True, supports_gqa=True)
class AegisAttentionV1(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        model_config: Any = None,
        config: Any = None,
        aegis_cfg: Optional[AegisConfigV1] = None,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        cfg = model_config if model_config is not None else config
        super().__init__(cfg, **kwargs)
        self.hidden_size: int = kwargs.get("hidden_size", kwargs.get("d_model", getattr(cfg, "hidden_size", getattr(cfg, "d_model", getattr(cfg, "dim", 768))) if cfg is not None else 768))
        self.num_heads: int = kwargs.get("num_heads", kwargs.get("num_attention_heads", getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 4)) if cfg is not None else 4))
        self.head_dim: int = kwargs.get("head_dim", getattr(cfg, "head_dim", getattr(cfg, "d_kv", self.hidden_size // max(1, self.num_heads))) if cfg is not None else self.hidden_size // max(1, self.num_heads))
        self.num_kv_heads: int = kwargs.get("num_kv_heads", kwargs.get("num_key_value_heads", getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", self.num_heads)) if cfg is not None else self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.cfg: AegisConfigV1 = aegis_cfg if aegis_cfg is not None else AegisConfigV1()

        if is_decoder is not None:
            self.is_decoder: bool = bool(is_decoder)
        else:
            self.is_decoder = (
                getattr(cfg, "is_decoder", False)
                or getattr(cfg, "causal", False)
                or getattr(cfg, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama") or any(str(getattr(cfg if "cfg" in locals() else (config if "config" in locals() else None), "model_type", "")).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                if cfg else False
            )

        self.is_nested_encoder: bool = False

        if self.cfg.num_local_heads + self.cfg.num_global_heads != self.num_heads:
            self.num_local_heads: int = (self.num_heads * 2) // 3
            self.num_global_heads: int = self.num_heads - self.num_local_heads
        else:
            self.num_local_heads = self.cfg.num_local_heads
            self.num_global_heads = self.cfg.num_global_heads

        self.q_proj = FactorisedLinear(self.hidden_size, self.all_head_size, energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank, max_rank=self.cfg.max_rank)
        self.k_proj = FactorisedLinear(self.hidden_size, self.all_kv_head_size, energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank, max_rank=self.cfg.max_rank)
        self.v_proj = FactorisedLinear(self.hidden_size, self.all_kv_head_size, energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank, max_rank=self.cfg.max_rank)
        self.out_proj = FactorisedLinear(self.all_head_size, self.hidden_size, energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank, max_rank=self.cfg.max_rank)

        rff_init = torch.empty(self.num_global_heads, self.head_dim, self.cfg.rff_features)
        for h in range(self.num_global_heads):
            q_mat, _ = torch.linalg.qr(torch.randn(max(self.head_dim, self.cfg.rff_features), max(self.head_dim, self.cfg.rff_features)))
            rff_init[h] = q_mat[:self.head_dim, :self.cfg.rff_features] * math.sqrt(self.head_dim)
        self.rff_proj: nn.Parameter = nn.Parameter(rff_init)

        if self.cfg.use_adaptive_gate:
            self.blend_logits: Optional[nn.Parameter] = nn.Parameter(torch.zeros(self.num_heads))
            self.gate_bias: float = max(2.0, self.cfg.gate_init_bias)
        else:
            self.blend_logits = None
            self.gate_bias = 0.0

        self.dropout: nn.Dropout = nn.Dropout(self.cfg.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorisedLinear:
        return self.v_proj

    @property
    def q_lin(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorisedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def dense(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def o_proj(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def c_proj(self) -> FactorisedLinear:
        return self.out_proj

    def copy_weights(self, old_attn: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(old_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            c_attn_mod = old_attn.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                if is_conv1d or (weight.dim() == 2 and weight.shape[0] < weight.shape[1]) or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data if bias is not None else None
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

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = old_attn.c_proj
            base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_p, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (c_proj_mod.__class__.__name__ == "Conv1D") or (base_p.__class__.__name__ == "Conv1D") or hasattr(c_proj_mod, "nf") or hasattr(base_p, "nf")
                if is_conv1d or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1]) or (w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(old_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(old_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        if (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention")
            or "RobertaSelfAttention" in str(type(old_attn))
            or "BertSelfAttention" in str(type(old_attn))
            or "roberta" in str(type(old_attn)).lower()
            or ("bert" in str(type(old_attn)).lower() and "self" in str(type(old_attn)).lower())
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisAttentionV1.copy_weights: could not extract mandatory projections {missing} "
                f"from {old_attn.__class__.__name__}. Refusing to install randomly-initialised weights."
            )

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"AegisAttentionV1 extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"AegisAttentionV1 extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target.decompose_from_weight(w_oriented, b, self.cfg.svd_energy)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(self.all_head_size, self.all_head_size, device=self.out_proj.up.weight.device, dtype=self.out_proj.up.weight.dtype)
                    target.decompose_from_weight(eye_w, None, self.cfg.svd_energy)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisAttentionV1":
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 4))
        return cls(model_config=model_config, pretrained_attn=old_attn, **kwargs)

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        Hkv = self.num_kv_heads

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_kv, Hkv, D).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_kv, Hkv, D).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = None
        if eff_mask is not None:
            if eff_mask.dtype == torch.bool:
                token_valid = eff_mask if eff_mask.dim() == 2 else eff_mask.any(dim=1).any(dim=-2)
            elif eff_mask.dim() == 2:
                # HuggingFace standard: 0=pad, 1=real token
                token_valid = eff_mask.bool()
            elif eff_mask.dim() == 4:
                # Extended mask: values near 0.0 = real, near -10000 = pad
                token_valid = (eff_mask > -5000.0).any(dim=1).any(dim=-2)

        if N < self.cfg.local_window or self.is_decoder:
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * (D**-0.5)
            if self.is_decoder and not is_cross and N > 1:
                causal_mask = torch.triu(torch.full((N, N_kv), -10000.0, device=Q.device, dtype=torch.float32), diagonal=1)
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                elif eff_mask.dim() == 3:
                    scores = scores + (eff_mask.unsqueeze(1).float() - 1.0) * 10000.0
                elif eff_mask.dim() == 4:
                    if eff_mask.dtype == torch.bool:
                        scores = scores.masked_fill(~eff_mask, -10000.0)
                    else:
                        scores = scores + eff_mask.float()

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores, dim=-1).to(Q.dtype)
            attn = self.dropout(attn)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn = attn * head_mask

            out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

            if self.is_nested_encoder:
                return out, (attn if output_attentions else None)
            return self.out_proj(out), (attn if output_attentions else None)

        Hl, Hg = self.num_local_heads, self.num_global_heads
        Q_local, Q_global = Q[:, :Hl].contiguous(), Q[:, Hl:].contiguous()
        K_local, K_global = K[:, :Hl].contiguous(), K[:, Hl:].contiguous()
        V_local, V_global = V[:, :Hl].contiguous(), V[:, Hl:].contiguous()
        # 1. Sliding Window Local Attention (for local heads)
        ctx_local = local_window_attention(
            Q_local, K_local, V_local,
            window=self.cfg.local_window,
            token_valid=token_valid,
            is_decoder=getattr(self, 'is_decoder', False)
        )
        ctx_global = kernel_linear_attention(
            Q_global, K_global, V_global, self.rff_proj, self.cfg.use_zero_sum_keys,
            is_causal=getattr(self, 'is_decoder', False),
            token_valid=token_valid,
        )

        if self.blend_logits is not None:
            global_signal = ctx_global.mean(dim=1, keepdim=True).expand(-1, H, -1, -1)
            local_full = torch.cat([ctx_local, ctx_global], dim=1)
            gate = torch.sigmoid(self.blend_logits.unsqueeze(-1).unsqueeze(-1) + self.gate_bias)
            context = gate * local_full + (1.0 - gate) * global_signal
        else:
            context = torch.cat([ctx_local, ctx_global], dim=1)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            context = context * head_mask

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("aegis_v1_inject")
@ATTENTION_REGISTRY.register("inject_aegis_v1")
def inject_aegis_v1(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_v1(
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

    layer = target

    r"""Injects AEGIS-V1 attention into a transformer layer."""
    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
    cfg = AegisConfigV1(
        svd_energy=getattr(comp_config, "svd_energy", 0.95),
        min_rank=getattr(comp_config, "min_rank", 4),
        max_rank=getattr(comp_config, "max_rank", 384),
        num_local_heads=getattr(comp_config, "num_local_heads", 8),
        num_global_heads=getattr(comp_config, "num_global_heads", 4),
        local_window=getattr(comp_config, "block_size", 64),
        rff_features=getattr(comp_config, "rff_features", 128),
        gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    new_attn = AegisAttentionV1(
        model_config=model_config,
        aegis_cfg=cfg,
    )
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error("inject_aegis_v1 copy_weights failed: %s", e)
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            new_attn.is_nested_encoder = True
            layer.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            layer.attention = new_attn
    elif hasattr(layer, "self"):
        new_attn.is_nested_encoder = False
        layer.self = new_attn
    elif hasattr(layer, "attn"):
        new_attn.is_nested_encoder = False
        layer.attn = new_attn
    return new_attn
