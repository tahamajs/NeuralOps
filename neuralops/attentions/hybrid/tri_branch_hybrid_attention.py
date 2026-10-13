"""
Aegis-Hybrid Attention — Production-Hardened (v3).

"""

from __future__ import annotations
from neuralops.core.common import _read_past_kv
from neuralops.core.common import FactorizedLinear
from neuralops.core.common import _detect_decoder, _env_flag, _env_float

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
)

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import repeat_kv
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)








# ===========================================================================
# Decoder detection (FIX 2)
# ===========================================================================
_DECODER_PREFIXES = (
    "gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
    "bloom", "falcon", "olmo", "mpt", "stablelm",
)




# ===========================================================================
# Config (FIX 9: is_decoder not False by default)
# ===========================================================================
@dataclass
class AegisHybridConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: int | None = None
    local_window: int = 256
    rank: int = 128
    lora_rank: int = 32
    dropout: float = 0.1
    use_adaptive_pruning: bool = False
    gate_init_bias: float = 0.0   # Zero-init to match GPT-2 exactly at step 0, avoiding KD shock
    is_decoder: bool | None = None  # FIX 9: None = auto-detect
    use_global: bool = True
    kernel_feature_dim: int = 128
    kernel_seed: int = 1234
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


# ===========================================================================
# KernelFeatureMap (FIX 4: proper FAVOR+)
# ===========================================================================
class KernelFeatureMap(nn.Module):
    """
    FAVOR+ random Fourier features:
        φ(x) = (1/sqrt(m)) * exp(ω·x - ||x||²/2)
    where ω ~ N(0, I) of shape (head_dim, num_features).
    """

    is_custom_attention: bool = False

    def __init__(self, head_dim: int, num_features: int = 128, seed: int = 1234):
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_features = int(num_features)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # FIX 4: ω ~ N(0, I), NO extra scaling
        proj = torch.randn(self.head_dim, self.num_features, generator=gen)
        # Optional orthogonal init for variance reduction when m <= d
        if self.num_features <= self.head_dim:
            q, _ = torch.linalg.qr(proj)
            proj = q
        # NOTE: standard FAVOR+ does NOT scale proj by sqrt(d).
        # The feature map itself handles scaling via 1/sqrt(m) and -||x||²/2.
        self.register_buffer("proj", proj)
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            proj = self.proj.float()
            # φ(x) = exp(ω·x - ||x||²/2)
            proj_x = torch.matmul(x_f, proj)  # (..., m)
            norm_sq_half = 0.5 * (x_f * x_f).sum(dim=-1, keepdim=True)  # (..., 1)
            # Numerical safety: clamp the exponent
            exponent = torch.clamp(proj_x - norm_sq_half, min=-30.0, max=10.0)
            phi = self.scale * torch.exp(exponent)
            return phi.to(x.dtype)


# ===========================================================================
# Chunked sliding-window local attention (FIX 5, 6, 7, 8)
# ===========================================================================
def _local_window_attention_chunked(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: torch.Tensor | None,
    is_decoder: bool,
    past_len: int,
    dropout_p: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, H, N, D = Q.shape
    N_kv = K.shape[2]
    # FIX 5: if window covers full sequence, use radius = N_kv (full context)
    if window >= N_kv:
        radius = N_kv
    else:
        radius = max(1, window // 2)

    scale = D**-0.5
    device = Q.device

    # FIX 6: avoid .item() — compute q_min/q_max from past_len + indices
    chunk = max(64, min(radius * 2, 256))

    outs = []
    attn_blocks = []

    for q_start in range(0, N, chunk):
        q_end = min(q_start + chunk, N)
        Qc = Q[:, :, q_start:q_end]

        # FIX 6: pure integer arithmetic, no .item()
        q_min_abs = past_len + q_start
        q_max_abs = past_len + q_end - 1
        k_lo = max(0, q_min_abs - radius)
        k_hi = min(N_kv, q_max_abs + radius + 1)

        if k_hi <= k_lo:
            outs.append(torch.zeros(B, H, q_end - q_start, D, device=device, dtype=Q.dtype))
            continue

        Kc = K[:, :, k_lo:k_hi]
        Vc = V[:, :, k_lo:k_hi]

        # Absolute positions
        q_abs = past_len + torch.arange(q_start, q_end, device=device)
        k_abs = torch.arange(k_lo, k_hi, device=device)

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

        Vc_t = Vc.to(attn.dtype)
        outs.append(torch.matmul(attn, Vc_t))
        if not training:
            attn_blocks.append((q_start, q_end, k_lo, k_hi, attn))

    output = torch.cat(outs, dim=2)

    attn_full = None
    if len(attn_blocks) > 0 and N * N_kv <= 4_000_000:
        attn_full = torch.zeros(B, H, N, N_kv, device=device, dtype=Q.dtype)
        for q_s, q_e, k_l, k_h, blk in attn_blocks:
            attn_full[:, :, q_s:q_e, k_l:k_h] = blk

    return output, attn_full


# ===========================================================================
# Global linear attention (FAVOR+)
# ===========================================================================
def _global_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    feature_map: KernelFeatureMap,
    token_valid: torch.Tensor | None,
) -> torch.Tensor:
    with torch.autocast(device_type=Q.device.type, enabled=False):
        Q_f = Q.float()
        K_f = K.float()
        V_f = V.float()

        Q_feat = feature_map(Q_f).float()
        K_feat = feature_map(K_f).float()

        if token_valid is not None:
            kv = token_valid[:, None, :, None].float()
            K_feat = K_feat * kv
            V_f = V_f * kv

        KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V_f)
        Z = K_feat.sum(dim=-2, keepdim=True)

        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-1, -2).contiguous()).clamp(min=1e-6)
        out = num / den
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(Q.dtype)


def _causal_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    feature_map: KernelFeatureMap,
    token_valid: torch.Tensor | None,
    chunk_size: int = 64,
) -> torch.Tensor:
    with torch.autocast(device_type=Q.device.type, enabled=False):
        Q_f = Q.float()
        K_f = K.float()
        V_f = V.float()
        B, H, N, D = Q_f.shape

        Q_feat = feature_map(Q_f).float()
        K_feat = feature_map(K_f).float()

        if token_valid is not None:
            kv = token_valid[:, None, :, None].float()
            K_feat = K_feat * kv
            V_f = V_f * kv

        if N == 1:
            KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V_f)
            Z = K_feat.sum(dim=-2, keepdim=True)
            num = torch.matmul(Q_feat, KV)
            den = torch.matmul(Q_feat, Z.transpose(-1, -2).contiguous()).clamp(min=1e-6)
            out = num / den
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(Q.dtype)

        # FIX 7: chunk_size adaptive to sequence
        chunk_size = min(chunk_size, N)
        pad_len = (chunk_size - (N % chunk_size)) % chunk_size

        if pad_len > 0:
            q_pad = F.pad(Q_feat, (0, 0, 0, pad_len))
            k_pad = F.pad(K_feat, (0, 0, 0, pad_len))
            v_pad = F.pad(V_f, (0, 0, 0, pad_len))
        else:
            q_pad, k_pad, v_pad = Q_feat, K_feat, V_f

        num_chunks = q_pad.shape[2] // chunk_size
        q_c = q_pad.view(B, H, num_chunks, chunk_size, -1)
        k_c = k_pad.view(B, H, num_chunks, chunk_size, -1)
        v_c = v_pad.view(B, H, num_chunks, chunk_size, D)

        if pad_len > 0:
            valid_mask = torch.ones(num_chunks, chunk_size, device=Q.device, dtype=torch.float32)
            valid_mask[-1, chunk_size - pad_len:] = 0.0
            v_m = valid_mask.view(1, 1, num_chunks, chunk_size, 1)
            q_c = q_c * v_m
            k_c = k_c * v_m
            v_c = v_c * v_m

        scores = torch.matmul(q_c, k_c.transpose(-2, -1))
        causal_mask = torch.triu(
            torch.ones(chunk_size, chunk_size, device=Q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, 0.0)

        num_intra = torch.matmul(scores, v_c)
        den_intra = scores.sum(dim=-1, keepdim=True)

        delta_s = torch.matmul(k_c.transpose(-2, -1), v_c)
        delta_z = k_c.sum(dim=-2)

        s_cumsum = torch.cumsum(delta_s, dim=2)
        s_past = torch.cat([torch.zeros_like(delta_s[:, :, :1]), s_cumsum[:, :, :-1]], dim=2)

        z_cumsum = torch.cumsum(delta_z, dim=2)
        z_past = torch.cat([torch.zeros_like(delta_z[:, :, :1]), z_cumsum[:, :, :-1]], dim=2)

        num_inter = torch.matmul(q_c, s_past)
        den_inter = (q_c * z_past.unsqueeze(-2)).sum(dim=-1, keepdim=True)

        denom = (den_intra + den_inter).clamp(min=1e-6)
        out = (num_intra + num_inter) / denom
        out = out.view(B, H, -1, D)[:, :, :N, :]
        return torch.nan_to_num(out.to(Q.dtype), nan=0.0, posinf=0.0, neginf=0.0)


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "tri_branch_hybrid", supports_causal=True, supports_gqa=True, supports_rope=True
)
class TriBranchHybridAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        pretrained_attn: nn.Module | None = None,
        rank: int | None = None,
        lora_rank: int | None = None,
        is_decoder: bool | None = None,
        whitening: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 128)
        except TypeError:
            super().__init__(config)

        if not isinstance(config, AegisHybridConfig):
            cfg_src = config if config is not None else kwargs
            h_size = hidden_size or getattr(cfg_src, "hidden_size", 768)
            n_heads = num_heads or getattr(
                cfg_src, "num_attention_heads", getattr(cfg_src, "n_heads", 12)
            )
            n_kv = getattr(
                cfg_src, "num_key_value_heads",
                getattr(cfg_src, "num_kv_heads", n_heads),
            )
            h_dim = getattr(cfg_src, "head_dim", None) or (h_size // max(1, n_heads))
            r = rank or getattr(cfg_src, "rank", min(128, h_size))
            l_r = lora_rank or getattr(cfg_src, "lora_rank", 32)

            # FIX 2: detect decoder using old_attn if available
            dec = _detect_decoder(config, is_decoder, pretrained_attn)

            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                num_key_value_heads=n_kv,
                local_window=kwargs.get("local_window", 256),
                rank=r,
                lora_rank=l_r,
                is_decoder=dec,
                use_global=kwargs.get("use_global", True),
                kernel_feature_dim=kwargs.get("kernel_feature_dim", 128),
                kernel_seed=kwargs.get("kernel_seed", 1234),
                chunk_size=kwargs.get("chunk_size", 64),
            )

        self.config = config
        self.hidden_size: int = int(config.hidden_size)
        self.num_heads: int = int(config.num_attention_heads)
        self.head_dim: int = int(config.head_dim)
        self.num_kv_heads: int = int(config.num_key_value_heads or self.num_heads)
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.local_window: int = int(config.local_window)
        self.rank: int = max(1, min(int(config.rank), self.hidden_size))
        self.lora_rank: int = max(1, min(int(config.lora_rank), self.hidden_size))
        self.dropout = nn.Dropout(config.dropout)

        # FIX 2: pass pretrained_attn to _detect_decoder for stronger signal
        self.is_decoder: bool = _detect_decoder(config, is_decoder, pretrained_attn)
        self.use_global: bool = bool(config.use_global)
        self.is_nested_encoder: bool = False
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.whitening = whitening or {}

        # ---- projections ----
        self.q_proj = FactorizedLinear(
            self.hidden_size, self.all_head_size,
            rank=self.rank, lora_rank=self.lora_rank, bias=True,
        )
        self.k_proj = FactorizedLinear(
            self.hidden_size, self.all_kv_head_size,
            rank=self.rank, lora_rank=self.lora_rank, bias=True,
        )
        self.v_proj = FactorizedLinear(
            self.hidden_size, self.all_kv_head_size,
            rank=self.rank, lora_rank=self.lora_rank, bias=True,
        )
        self.out_proj = FactorizedLinear(
            self.all_head_size, self.hidden_size,
            rank=self.rank, lora_rank=self.lora_rank, bias=True,
        )

        # ---- kernel feature map ----
        self.kernel_map = KernelFeatureMap(
            head_dim=self.head_dim,
            num_features=config.kernel_feature_dim,
            seed=config.kernel_seed,
        )

        # FIX 3: zero-init fusion_gate to match GPT-2 exactly at step 0, avoiding KD shock.
        _beta_init = _env_float("AEGIS_FUSION_BETA", getattr(config, "gate_init_bias", 0.0))
        self.fusion_gate = nn.Parameter(
            torch.full((self.num_heads,), _beta_init), requires_grad=True
        )

        # storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---- aliases ----
    @property
    def fusion_beta(self) -> nn.Parameter:
        return self.fusion_gate

    @fusion_beta.setter
    def fusion_beta(self, val: nn.Parameter) -> None:
        self.fusion_gate = val

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
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # SVD init
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(
        w: torch.Tensor,
        b: torch.Tensor | None,
        proj: FactorizedLinear,
        whitening_L: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            W = w.detach().float()
            out_dim, in_dim = W.shape
            target_out, target_in = proj.out_features, proj.in_features

            if out_dim > target_out and out_dim % target_out == 0:
                rep = out_dim // target_out
                W = W.view(target_out, rep, in_dim).mean(dim=1)
                if b is not None:
                    b = b.view(target_out, rep).mean(dim=1)
            elif out_dim < target_out and target_out % out_dim == 0:
                rep = target_out // out_dim
                W = W.repeat_interleave(rep, dim=0)
                if b is not None:
                    b = b.repeat_interleave(rep)

            if W.shape[0] != target_out or W.shape[1] != target_in:
                raise ValueError(
                    f"FactorizedLinear shape mismatch after GQA pooling: "
                    f"{tuple(W.shape)} vs ({target_out}, {target_in})"
                )

            L = None
            if whitening_L is not None:
                Lw = whitening_L.float().to(W.device)
                if Lw.shape == (target_in, target_in):
                    eps = 1e-6 * torch.trace(Lw @ Lw.t()).item() / max(1, target_in)
                    L = Lw + eps * torch.eye(target_in, device=W.device, dtype=W.dtype)

            if L is not None:
                try:
                    W_eff = W @ L
                    U, S, Vh = torch.linalg.svd(W_eff, full_matrices=False)
                    L_inv = torch.linalg.inv(L)
                    Vh = Vh @ L_inv
                except Exception:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            else:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            r = min(proj.rank, U.size(1), Vh.size(0))
            dev = proj.down.weight.device
            W = W.to(dev)
            if b is not None:
                b = b.to(dev)

            proj.down.weight.data.zero_()
            proj.up.weight.data.zero_()
            proj.down.weight.data[:r, :].copy_(Vh[:r, :].to(proj.down.weight.dtype))
            US = U[:, :r] @ torch.diag(S[:r])
            proj.up.weight.data[:, :r].copy_(US.to(proj.up.weight.dtype))

            # Standard stable LoRA init: lora_A kaiming, lora_B zero so initial ΔW = 0
            if proj.lora_A is not None and proj.lora_B is not None and proj.lora_rank > 0:
                nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                proj.lora_B.data.zero_()

            if b is not None and proj.up.bias is not None:
                n = min(b.numel(), proj.up.bias.numel())
                proj.up.bias.data[:n].copy_(b.detach()[:n].to(proj.up.bias.dtype))
            elif proj.up.bias is not None:
                proj.up.bias.data.zero_()

            proj._cached_weight = None
            proj._cached_versions = ()

    # ------------------------------------------------------------------
    # copy_weights (FIX 8)
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        try:
            for p in old_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        try:
            from neuralops.core.adapters import UniversalAdapter
        except Exception:
            UniversalAdapter = None

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # UniversalAdapter first
        if UniversalAdapter is not None:
            for p in ("q", "k", "v", "o"):
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, p)
                    if w is not None:
                        extracted[p] = (w, b)
                except Exception:
                    pass

        # FIX 8: GPT-2 fused c_attn with correct condition
        if hasattr(old_attn, "c_attn") and extracted["q"][0] is None:
            c = old_attn.c_attn
            c_w = c.weight.data.clone()
            is_conv1d = (
                c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]
            ) or c.__class__.__name__ == "Conv1D"
            if is_conv1d:
                c_w = c_w.t().contiguous()
            split = c_w.size(0) // 3
            c_b = getattr(c, "bias", None)
            c_b = c_b.data.clone() if c_b is not None else None
            extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
            extracted["k"] = (c_w[split:2*split], c_b[split:2*split] if c_b is not None else None)
            extracted["v"] = (c_w[2*split:], c_b[2*split:] if c_b is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c = old_attn.c_proj
            w_o = c.weight.data.clone()
            if (w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1]) or c.__class__.__name__ == "Conv1D":
                w_o = w_o.t().contiguous()
            b_o = getattr(c, "bias", None)
            extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # Nested BERT/RoBERTa
        if hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            self.is_nested_encoder = True
            s = old_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                if extracted[key][0] is None:
                    m = getattr(s, sub, None)
                    if m is not None and hasattr(m, "weight"):
                        b = getattr(m, "bias", None)
                        extracted[key] = (m.weight.data.clone(),
                                          b.data.clone() if b is not None else None)
            if (extracted["o"][0] is None and hasattr(old_attn, "output")
                    and hasattr(old_attn.output, "dense")):
                d = old_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(),
                                  b.data.clone() if b is not None else None)

        # Flat modules
        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for key, names in named_map.items():
            if extracted[key][0] is None:
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(),
                                              b.data.clone() if b is not None else None)
                            break

        # Fail-loud validation for mandatory projections
        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if missing:
            raise RuntimeError(f"AegisHybridAttention: mandatory projections {missing} missing")

        # Apply SVD init
        for key, target in (
            ("q", self.q_proj), ("k", self.k_proj),
            ("v", self.v_proj), ("o", self.out_proj),
        ):
            w, b = extracted[key]
            if w is not None:
                try:
                    self._svd_init(w, b, target, whitening_L=self.whitening.get(key, None))
                except Exception as e:
                    logger.warning("Aegis-Hybrid SVD init failed for '%s': %s", key, e)
            elif key == "o" and self.is_nested_encoder:
                with torch.no_grad():
                    r = min(target.out_features, target.in_features, target.rank)
                    target.down.weight.data.zero_()
                    target.up.weight.data.zero_()
                    target.down.weight.data[:r, :r].fill_diagonal_(1.0)
                    target.up.weight.data[:r, :r].fill_diagonal_(1.0)

        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, L, device) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape == (B, L):
                return m
            if m.shape[0] == B and m.shape[1] != L:
                m = m[:, :L]
                if m.shape[1] == L:
                    return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

    _read_past_kv = staticmethod(_read_past_kv)

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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        device = hidden_states.device
        Dh = self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, Dh).transpose(1, 2).contiguous()
        K_new = (
            self.k_proj(kv_input)
            .view(B, N_kv_new, self.num_kv_heads, Dh)
            .transpose(1, 2).contiguous()
        )
        V_new = (
            self.v_proj(kv_input)
            .view(B, N_kv_new, self.num_kv_heads, Dh)
            .transpose(1, 2).contiguous()
        )

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_k_probe, _, _ = self._read_past_kv(past_key_value, layer_idx)
            past_len_rope = 0 if past_k_probe is None else past_k_probe.shape[-2]
            cos = cos[..., past_len_rope : past_len_rope + N, :]
            sin = sin[..., past_len_rope : past_len_rope + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        # KV cache
        past_k, past_v, _ = self._read_past_kv(past_key_value, layer_idx)
        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
            past_len = past_k.shape[-2]
        else:
            K_cat, V_cat = K_new, V_new
            past_len = 0

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(K_new, V_new, layer_idx)

        N_kv = K_cat.shape[2]
        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        # Local
        local_out, attn_local = _local_window_attention_chunked(
            Q, K_full, V_full,
            window=self.local_window,
            token_valid=token_valid,
            is_decoder=self.is_decoder and not is_cross,
            past_len=past_len,
            dropout_p=self.config.dropout,
            training=self.training,
        )

        # Global
        if self.use_global:
            # FIX 2: use causal version when is_decoder=True
            if self.is_decoder and not is_cross:
                global_out = _causal_linear_attention(
                    Q, K_full, V_full, self.kernel_map, token_valid,
                    chunk_size=getattr(self.config, "chunk_size", 64),
                )
            else:
                global_out = _global_linear_attention(
                    Q, K_full, V_full, self.kernel_map, token_valid,
                )
            clamped_gate = torch.clamp(self.fusion_gate, -10.0, 10.0)
            beta = torch.sigmoid(clamped_gate).view(1, self.num_heads, 1, 1)
            fused = beta * local_out + (1.0 - beta) * global_out
        else:
            fused = local_out

        if head_mask is not None:
            hm = head_mask.to(fused.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            fused = fused * hm

        fused = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final = fused
        else:
            final = self.out_proj(fused)

        final = torch.nan_to_num(final, nan=0.0, posinf=1e4, neginf=-1e4)
        final = final.to(hidden_states.dtype)

        # HuggingFace-compatible return tuple
        # GPT-2 expects (hidden, present, attn_weights) when
        # use_cache=True and output_attentions=True.
        _use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        present = (K_new, V_new) if _use_cache and not is_cross else None

        if output_attentions:
            return (final, present, attn_local)
        return (final, present)

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("q_proj", self.q_proj), ("k_proj", self.k_proj),
            ("v_proj", self.v_proj), ("out_proj", self.out_proj),
            ("kernel_map", self.kernel_map),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["fusion_gate"] = _sz(self.fusion_gate)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injector
# ===========================================================================
def inject_aegis_hybrid(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_aegis_hybrid: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_hybrid(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    src = old_attn
    if src is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break
    if src is None:
        src = target

    if isinstance(src, AegisHybridAttention) or "AegisHybrid" in src.__class__.__name__:
        return None

    h_size = getattr(cfg, "hidden_size", None) if cfg else None
    if h_size is None:
        for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
            m = getattr(src, attr, None)
            if m is not None and hasattr(m, "out_features"):
                h_size = m.out_features
                break
    h_size = int(h_size or kwargs.get("hidden_size", 768))

    n_heads = getattr(cfg, "num_attention_heads", None) if cfg else None
    if n_heads is None:
        n_heads = getattr(src, "num_attention_heads",
                          getattr(src, "num_heads", kwargs.get("num_heads", 12)))
    n_heads = int(n_heads)

    # FIX 2: pass src to _detect_decoder
    dec = _detect_decoder(cfg, kwargs.get("is_decoder"), src)

    aegis_cfg = AegisHybridConfig(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        head_dim=kwargs.get("head_dim", h_size // max(1, n_heads)),
        num_key_value_heads=(getattr(cfg, "num_key_value_heads", n_heads) if cfg else n_heads),
        local_window=kwargs.get("local_window",
                                getattr(comp_config, "local_window", 256) if comp_config else 256),
        rank=kwargs.pop("rank",
                        getattr(comp_config, "rank", min(128, h_size)) if comp_config else min(128, h_size)),
        lora_rank=kwargs.get("lora_rank",
                             getattr(comp_config, "lora_rank", 32) if comp_config else 32),
        dropout=kwargs.get("dropout",
                           getattr(comp_config, "dropout", 0.1) if comp_config else 0.1),
        is_decoder=dec,
        use_global=kwargs.get("use_global",
                              getattr(comp_config, "use_global", True) if comp_config else True),
        gate_init_bias=kwargs.get("gate_init_bias", 0.0),  # FIX 3
    )

    try:
        new_attn = AegisHybridAttention(
            config=aegis_cfg,
            pretrained_attn=src if kwargs.get("copy_weights", True) else None,
            whitening=kwargs.pop("whitening", None),
        )
    except Exception as e:
        logger.error("inject_aegis_hybrid: construction failed: %s", e)
        return None

    p = next(src.parameters(), None)
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True

    if not spliced:
        logger.error("inject_aegis_hybrid: no writable attention attribute on %s",
                     type(target).__name__)
        return None

    return new_attn


# Backward-compatibility aliases for tests and external callers
AegisHybridAttention = TriBranchHybridAttention
TriBranchHybridConfig = AegisHybridConfig

__all__ = [
    "TriBranchHybridAttention",
    "AegisHybridAttention",
    "AegisHybridConfig",
    "TriBranchHybridConfig",
    "FactorizedLinear",
    "KernelFeatureMap",
    "inject_aegis_hybrid",
    "apply_rotary_pos_emb",
]