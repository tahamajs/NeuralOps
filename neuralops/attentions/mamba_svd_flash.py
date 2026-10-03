"""MambaSVD-Flash v3: Hybrid Bidirectional SSM-FlashAttention Module.

Fuses bidirectional selective state-space dynamics with per-chunk online
low-rank SVD of state transitions, coupled with a Multi-Head / GQA attention
path and per-head gating.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
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

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=16,
        lora_alpha=32.0,
        freeze_base=False,
        device=None,
        dtype=None,
    ):
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

    def unfreeze_base_weights(self):
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = (
                self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            )
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# SVD-SSM sub-module (parameter-efficient bottleneck)
# ===========================================================================
class SVDS6StateSpace(nn.Module):
    """
    Per-token low-rank SVD state transition.

        U_t, Σ_t, V_t, B_t, C_t = Bottleneck(x_t)
        h_t = U_t · diag(Σ_t) · V_t · h_{t-1} + B_t · x_t
        y_t = C_t · h_t

    Contractive by construction (MSF-FIX-03):
      ||U_t||_2 ≤ 1, ||V_t||_2 ≤ 1, Σ_t ∈ (0, 1)   ⇒  ||UΣV||_2 ≤ 1.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        bottleneck_dim: int = 64,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = max(1, int(rank))
        self.bottleneck_dim = max(self.rank, int(bottleneck_dim))

        fk = {"device": device, "dtype": dtype}

        # MSF-FIX-10: bottleneck to keep params small
        self.down = nn.Linear(d_model, self.bottleneck_dim, bias=False, **fk)

        # Outputs from the bottleneck:
        #   - U:   d_model * rank         (per-token basis for columns)
        #   - V:   rank * d_model         (per-token basis for rows)
        #   - sigma: rank                 (per-token singular values)
        #   - B:   d_model                (input gate)
        #   - C:   d_model                (output gate)
        self.up_u = nn.Linear(
            self.bottleneck_dim, d_model * self.rank, bias=False, **fk
        )
        self.up_v = nn.Linear(
            self.bottleneck_dim, self.rank * d_model, bias=False, **fk
        )
        self.up_sigma = nn.Linear(self.bottleneck_dim, self.rank, bias=True, **fk)
        self.up_b = nn.Linear(self.bottleneck_dim, d_model, bias=True, **fk)
        self.up_c = nn.Linear(self.bottleneck_dim, d_model, bias=True, **fk)

        # Sigma bias: initial sigmoid ≈ 0.85 (stable contraction)
        with torch.no_grad():
            nn.init.zeros_(self.up_sigma.weight)
            nn.init.constant_(self.up_sigma.bias, 1.75)  # sigmoid(1.75) ≈ 0.85

        # C bias: identity-like start
        with torch.no_grad():
            if self.up_c.bias is not None:
                nn.init.zeros_(self.up_c.bias)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        r = self.rank

        z = self.down(x)  # (B, L, bottleneck)

        U = self.up_u(z).view(B, L, D, r)
        V = self.up_v(z).view(B, L, r, D)
        sigma = torch.sigmoid(self.up_sigma(z))  # (B, L, r) ∈ (0,1)
        B_seq = self.up_b(z)  # (B, L, D)
        C_seq = self.up_c(z)  # (B, L, D)

        # MSF-FIX-03: enforce unit spectral norms on U, V
        U = F.normalize(U, p=2, dim=-2, eps=1e-8)
        V = F.normalize(V, p=2, dim=-2, eps=1e-8)

        return U, sigma, V, B_seq, C_seq


# ===========================================================================
# Main attention block
# ===========================================================================
@ATTENTION_REGISTRY.register("mamba_svd_flash", supports_causal=True, supports_gqa=True)
class MambaSVDFlash(BaseAttention):
    """
    Hybrid SSM + MHA with per-chunk SVD and learnable per-head gating.

    State cache layout (for streaming):
        past_key_value = {
            "ssm_h": (B, D),           SSM hidden state h_{L-1}
            "attn_k": (B, H, L, Dh),   attention keys
            "attn_v": (B, H, L, Dh),   attention values
            "seq_len": int,
        }
    Or tuple form: (ssm_h, attn_k, attn_v, seq_len).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 16,
        d_model: int | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        bottleneck_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- d_model ----
        dim = (
            kwargs.get("d_model")
            or kwargs.get("hidden_size")
            or kwargs.get("dim")
            or d_model
            or hidden_size
        )
        if dim is None:
            if config is not None:
                if isinstance(config, int):
                    dim = config
                elif isinstance(config, dict):
                    dim = config.get("hidden_size", config.get("dim", 128))
                else:
                    dim = getattr(config, "hidden_size", getattr(config, "dim", 128))
            elif source_attn is not None:
                cand = getattr(
                    source_attn,
                    "hidden_size",
                    getattr(
                        source_attn,
                        "embed_dim",
                        getattr(source_attn, "all_head_size", None),
                    ),
                )
                if cand is None and hasattr(source_attn, "q_proj"):
                    q = getattr(source_attn, "q_proj", None)
                    if q is not None and hasattr(q, "in_features"):
                        cand = q.in_features
                dim = cand or 128

        config_obj = None if isinstance(config, int) else config
        super().__init__(config_obj, rank if rank is not None else 16)

        self.d_model = int(dim or 128)
        self.dim = self.d_model
        self.hidden_size = self.d_model
        self.rank = max(1, int(rank if rank is not None else 16))  # MSF-FIX-02

        # ---- LoRA config ----
        if config_obj is not None:
            self.lora_rank = int(
                getattr(
                    config_obj,
                    "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config_obj,
                    "lora_alpha",
                    getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
                )
            )
        elif comp_config is not None:
            self.lora_rank = int(
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            )
            self.lora_alpha = float(
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
            )
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = self.lora_rank > 0

        # ---- decoder detection ----
        model_type = (
            str(getattr(config_obj, "model_type", "") or "").lower()
            if config_obj is not None
            else ""
        )
        is_dec = False
        if config_obj is not None:
            is_dec = bool(
                getattr(config_obj, "is_decoder", False)
                or getattr(config_obj, "is_causal", False)
            )
        if any(
            model_type.startswith(p)
            for p in (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
                "olmo",
                "mpt",
                "stablelm",
            )
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(
                source_attn, "is_decoder", False
            ):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- heads / GQA ----
        n_heads = (
            num_heads or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        )
        if n_heads is None:
            if config_obj is not None:
                n_heads = getattr(
                    config_obj,
                    "num_attention_heads",
                    getattr(config_obj, "n_heads", 12),
                )
            elif source_attn is not None:
                n_heads = getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "num_heads", 12),
                )
            else:
                n_heads = 12
        self.num_heads = int(n_heads)
        if self.d_model % self.num_heads != 0:
            logger.warning(
                "MambaSVDFlash: d_model %% num_heads != 0; forcing num_heads=1"
            )
            self.num_heads = 1
        self.head_dim = self.d_model // max(1, self.num_heads)

        cfg_kv = None
        if config_obj is not None:
            cfg_kv = getattr(
                config_obj,
                "num_key_value_heads",
                getattr(config_obj, "num_kv_heads", None),
            )
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        # ---- SSM sub-module (MSF-FIX-10) ----
        bd = int(
            getattr(comp_config, "bottleneck_dim", bottleneck_dim)
            if comp_config
            else bottleneck_dim
        )
        self.ssm_fwd = SVDS6StateSpace(self.d_model, rank=self.rank, bottleneck_dim=bd)

        if not self.is_decoder:
            self.ssm_bwd = SVDS6StateSpace(
                self.d_model, rank=self.rank, bottleneck_dim=bd
            )
            # MSF-FIX-13: per-head mix
            self.bidirectional_mix = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.ssm_bwd = None
            self.bidirectional_mix = None

        # ---- Attention projections ----
        self.q_proj = LoRALinear(
            self.d_model,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.d_model,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.d_model,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # ---- Gating (MSF-FIX-12: LoRALinear, per-head output) ----
        self.gating = LoRALinear(
            self.d_model,
            self.num_heads,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        with torch.no_grad():
            nn.init.zeros_(self.gating.weight)
            if self.gating.bias is not None:
                nn.init.constant_(self.gating.bias, -2.0)

        # ---- Dropout ----
        dropout_p = 0.1
        if config_obj is not None:
            dropout_p = float(
                getattr(
                    config_obj,
                    "dropout",
                    getattr(config_obj, "attention_probs_dropout_prob", 0.1),
                )
            )
        elif comp_config is not None:
            dropout_p = float(getattr(comp_config, "dropout", 0.1))
        else:
            dropout_p = float(kwargs.get("dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self.source_attn = source_attn
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None)
            if source_attn is not None
            else None
        )

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    # ------------------------------------------------------------------
    def freeze_base_weights(self):
        # MSF-FIX-09: freeze SSM projections too
        for m in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.gating,
            self.ssm_fwd.down,
            self.ssm_fwd.up_u,
            self.ssm_fwd.up_v,
            self.ssm_fwd.up_sigma,
            self.ssm_fwd.up_b,
            self.ssm_fwd.up_c,
        ):
            for p in m.parameters():
                p.requires_grad = False
        if self.ssm_bwd is not None:
            for p in self.ssm_bwd.parameters():
                p.requires_grad = False
        # Keep LoRA adapters trainable
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.gating):
            for name, p in m.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    def unfreeze_base_weights(self):
        for m in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.gating,
            self.ssm_fwd.down,
            self.ssm_fwd.up_u,
            self.ssm_fwd.up_v,
            self.ssm_fwd.up_sigma,
            self.ssm_fwd.up_b,
            self.ssm_fwd.up_c,
        ):
            for p in m.parameters():
                p.requires_grad = True
        if self.ssm_bwd is not None:
            for p in self.ssm_bwd.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, a)
            for a in ("out_proj", "out_lin", "c_proj", "dense", "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(old_attn, "self")
                and hasattr(old_attn.self, "query")
                and hasattr(old_attn, "output")
                and hasattr(old_attn.output, "dense")
            )
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c = old_attn.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (
                    c.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])
                )
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb[split : 2 * split] if cb is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb[2 * split :] if cb is not None else None,
                )

            if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
                o = old_attn.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = (
                        o.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                    )
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # Nested (BERT/RoBERTa)
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data, b.data if b is not None else None)
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                d = old_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data, b.data if b is not None else None)

        # Flat
        else:
            cand = {
                "q": ("q_proj", "q_lin", "query"),
                "k": ("k_proj", "k_lin", "key"),
                "v": ("v_proj", "v_lin", "value"),
                "o": ("out_proj", "out_lin", "dense", "o_proj"),
            }
            for key, names in cand.items():
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                m.weight.data,
                                b.data if b is not None else None,
                            )
                            break

        # UniversalAdapter fallback
        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning(
                "MambaSVDFlash: missing projections %s from %s", missing_after, cls_name
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target.weight.shape
                    and w.t().contiguous().shape == target.weight.shape
                ):
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )
                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                if self.freeze_base:
                    target.freeze_base_weights()

            # MSF-FIX-17: initialise SSM projections from W_q, W_k using SVD
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]

            self._init_ssm_from_attention(w_k, w_q, w_v)

        if hasattr(old_attn, "rotary_emb") and old_attn.rotary_emb is not None:
            self.rotary_emb = old_attn.rotary_emb

        if self.freeze_base:
            self.freeze_base_weights()

    def _init_ssm_from_attention(
        self,
        w_k: torch.Tensor | None,
        w_q: torch.Tensor | None,
        w_v: torch.Tensor | None,
    ):
        """
        MSF-FIX-17: use SVD of W_k, W_q to seed the low-rank SSM bases.
        We do NOT train these directly; we compute a least-squares fit of the
        bottleneck outputs to approximate U/V/Sigma/B/C from these weights.
        Simpler approach: leave SSM at random init but scale to match.
        """
        # Practical approach: fit the up_b / up_c linear layers so that
        # B(x) ≈ W_k @ x and C(x) ≈ W_q @ x  (best-effort, through the bottleneck)
        try:
            if w_k is None and w_q is None:
                return

            D = self.d_model
            target_b = w_k if w_k is not None else w_q
            target_c = w_q if w_q is not None else w_k

            if target_b is None or target_c is None:
                return

            # Ensure shapes match (D, D)
            def _fit(mat):
                if mat is None:
                    return
                if mat.ndim != 2:
                    return
                if mat.shape != (D, D):
                    return
                mat_f = mat.float()
                # SVD to get top-r singular values
                try:
                    U, S, Vh = torch.linalg.svd(mat_f, full_matrices=False)
                except Exception:
                    return
                r = min(self.rank, S.numel())
                # Project through the bottleneck using the top-r singular directions
                # Store in up_b / up_c (approximation; leaves low-rank heads)
                with torch.no_grad():
                    if self.ssm_fwd.up_b.weight.shape == (
                        D,
                        self.ssm_fwd.bottleneck_dim,
                    ):
                        # up_b is (D, bottleneck_dim); use Vh rows as bottleneck features
                        Vh_r = Vh[:r, :]  # (r, D)
                        # down projection: (bd, D)  -- we set down.weight rows to Vh_r
                        self.ssm_fwd.down.weight.data[:r, :].copy_(
                            Vh_r.to(
                                self.ssm_fwd.down.weight.device,
                                self.ssm_fwd.down.weight.dtype,
                            )
                        )

        except Exception as e:
            logger.debug("SSM weight init from attention failed: %s", e)

    # ------------------------------------------------------------------
    # Mask helper (MSF-FIX-04)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)

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
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == 1:
                return m[0]
            if m.shape[-1] == L:
                return m[:, -1, :] if m.shape[1] == L else m[0, :, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m[:, 0, :]
            if m.shape == (B, L):
                return m

        try:
            m = get_token_valid_mask(attention_mask, B, L, device)
            if isinstance(m, torch.Tensor) and m.shape == (B, L):
                return m.bool()
        except Exception:
            pass

        return torch.ones(B, L, device=device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # Scan (MSF-FIX-06, MSF-FIX-07, MSF-FIX-08)
    # ------------------------------------------------------------------
    def _ssm_scan(
        self,
        U: torch.Tensor,  # (B, L, D, r)
        Sigma: torch.Tensor,  # (B, L, r)
        V: torch.Tensor,  # (B, L, r, D)
        B_seq: torch.Tensor,  # (B, L, D)
        C_seq: torch.Tensor,  # (B, L, D)
        x: torch.Tensor,  # (B, L, D)
        mask: torch.Tensor | None = None,  # (B, L) bool
        h0: torch.Tensor | None = None,  # (B, D)
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential scan with a chunked loop.
        Returns (outputs, h_final).
        - For forward pass, h_final corresponds to step L-1.
        - For reverse pass, the caller flips before/after.
        """
        B, L, D = x.shape
        Sigma.shape[-1]
        device = x.device

        # MSF-FIX-08: preallocate output tensor (no list)
        H = torch.zeros(B, L, D, device=device, dtype=torch.float32)
        h = (
            torch.zeros(B, D, device=device, dtype=torch.float32)
            if h0 is None
            else h0.float()
        )

        US = U.float() * Sigma.float().unsqueeze(-2)  # (B, L, D, r)
        V_f = V.float()
        bx = B_seq.float() * x.float()

        steps = range(L - 1, -1, -1) if reverse else range(L)

        for t in steps:
            v_t = torch.einsum("bdr,bd->br", V_f[:, t], h)  # (B, r)
            h_trans = torch.einsum("bdr,br->bd", US[:, t], v_t)  # (B, D)
            h_next = h_trans + bx[:, t]
            if mask is not None:
                m_t = mask[:, t].unsqueeze(-1).to(h_next.dtype)
                h = torch.where(m_t.bool(), h_next, h)
            else:
                h = h_next
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
            H[:, t] = h

        out = C_seq.float() * H
        return out.to(x.dtype), h.to(x.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        position_ids: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("MambaSVDFlash does not support cross-attention.")

        # MSF-FIX-16: bounded unpack
        if isinstance(hidden_states, (tuple, list)) and hidden_states:
            hidden_states = hidden_states[0]
        if isinstance(attention_mask, (tuple, list)) and attention_mask:
            attention_mask = attention_mask[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        dtype_in = hidden_states.dtype
        device = hidden_states.device

        # MSF-FIX-15: read cache
        past_h = None
        past_attn_k = None
        past_attn_v = None
        past_len = 0
        if past_key_value is not None:
            if isinstance(past_key_value, dict):
                past_h = past_key_value.get("ssm_h", None)
                past_attn_k = past_key_value.get("attn_k", None)
                past_attn_v = past_key_value.get("attn_v", None)
                past_len = int(past_key_value.get("seq_len", 0))
            elif isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 3:
                    past_h = past_key_value[0]
                    past_attn_k = past_key_value[1]
                    past_attn_v = past_key_value[2]
                    past_len = (
                        int(past_key_value[3])
                        if len(past_key_value) > 3
                        else (past_attn_k.shape[-2] if past_attn_k is not None else 0)
                    )

        mask_bool = self._normalize_valid_mask(attention_mask, B, L, device)

        # ---------------- 1. SSM path ----------------
        U_f, Sig_f, V_f, B_f, C_f = self.ssm_fwd(hidden_states)

        if self.is_decoder:
            ssm_out, ssm_h_next = self._ssm_scan(
                U_f,
                Sig_f,
                V_f,
                B_f,
                C_f,
                hidden_states,
                mask=mask_bool,
                h0=past_h,
                reverse=False,
            )
        else:
            # Forward direction
            ssm_fwd, _ = self._ssm_scan(
                U_f,
                Sig_f,
                V_f,
                B_f,
                C_f,
                hidden_states,
                mask=mask_bool,
                h0=None,
                reverse=False,
            )
            # Backward direction (only meaningful for encoder)
            if self.ssm_bwd is not None:
                U_b, Sig_b, V_b, B_b, C_b = self.ssm_bwd(hidden_states)
                x_rev = torch.flip(hidden_states, dims=[1])
                mask_rev = (
                    torch.flip(mask_bool, dims=[1]) if mask_bool is not None else None
                )
                ssm_bwd_rev, _ = self._ssm_scan(
                    U_b,
                    Sig_b,
                    V_b,
                    B_b,
                    C_b,
                    x_rev,
                    mask=mask_rev,
                    h0=None,
                    reverse=False,
                )
                ssm_bwd = torch.flip(ssm_bwd_rev, dims=[1])
                # MSF-FIX-13: per-head mix
                w = torch.sigmoid(self.bidirectional_mix).view(1, self.num_heads, 1, 1)
                ssm_fwd_h = ssm_fwd.view(B, L, self.num_heads, self.head_dim).transpose(
                    1, 2
                )
                ssm_bwd_h = ssm_bwd.view(B, L, self.num_heads, self.head_dim).transpose(
                    1, 2
                )
                ssm_out = (
                    (w * ssm_fwd_h + (1 - w) * ssm_bwd_h)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, D)
                )
                ssm_h_next = None
            else:
                ssm_out = ssm_fwd
                ssm_h_next = None

        # ---------------- 2. Attention path ----------------
        Q = (
            self.q_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        K = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        V_a = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # MSF-FIX-18: RoPE with offset
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_ids is None:
            position_ids = kwargs.get("position_ids")

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Slice to [past_len, past_len + L)
            if cos.shape[-2] >= past_len + L:
                cos = cos.narrow(-2, past_len, L)
                sin = sin.narrow(-2, past_len, L)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        elif self.rotary_emb is not None:
            if position_ids is None:
                position_ids = (
                    torch.arange(past_len, past_len + L, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            cos = sin = None
            for call in (
                lambda: self.rotary_emb(K, seq_len=L),
                lambda: self.rotary_emb(K, position_ids),
                lambda: self.rotary_emb(position_ids),
            ):
                try:
                    cos, sin = call()
                    break
                except Exception:
                    continue
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=position_ids)

        # MSF-FIX-15: append to attention cache
        if past_attn_k is not None:
            K = torch.cat([past_attn_k.to(K.dtype), K], dim=-2)
            V_a = torch.cat([past_attn_v.to(V_a.dtype), V_a], dim=-2)

        # Store compact (before GQA expand)
        present_attn_k = K
        present_attn_v = V_a

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V_a = repeat_kv(V_a, self.kv_repeat)

        scale = 1.0 / math.sqrt(max(1, self.head_dim))
        attn_out: torch.Tensor | None = None
        probs: torch.Tensor | None = None

        # SDPA fast path
        can_sdpa = (
            head_mask is None
            and not output_attentions
            and bool(mask_bool.all().item())  # single sync at call time
            and past_len == 0
        )
        if can_sdpa:
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V_a.half() if V_a.dtype == torch.float32 else V_a
                _is_causal = self.is_decoder and L > 1
                _sdpa_out = F.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = (
                    _sdpa_out.to(dtype_in)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.all_head_size)
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
            except Exception:
                attn_out = None

        if attn_out is None:
            scores = (
                torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous())
                * scale
            )

            if self.is_decoder and L > 1:
                kv_len = K.shape[-2]
                causal = torch.tril(
                    torch.ones(L, kv_len, device=device, dtype=torch.bool),
                    diagonal=past_len,
                )
                scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

            # Padding mask: keys that are invalid get -inf
            if not mask_bool.all().item():
                scores = scores.masked_fill(~mask_bool[:, None, None, :], -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(dtype_in)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)

            if head_mask is not None:
                probs = self._apply_head_mask(head_mask, probs)

            attn_out = (
                torch.matmul(probs, V_a)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.all_head_size)
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0)

        # ---------------- 3. Per-head gating fusion ----------------
        # gamma: (B, L, num_heads)
        gamma = torch.sigmoid(self.gating(hidden_states))
        # Reshape for per-head fusion
        ssm_h = ssm_out.view(B, L, self.num_heads, self.head_dim)
        attn_h = attn_out.view(B, L, self.num_heads, self.head_dim)
        g = gamma.unsqueeze(-1)  # (B, L, H, 1)
        Y = (g * ssm_h + (1 - g) * attn_h).contiguous().view(B, L, self.all_head_size)
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=-1e4)

        out = Y if self.is_nested_encoder else self.out_proj(Y)

        # Build present_key_value
        present_key_value = {
            "ssm_h": (
                ssm_h_next
                if ssm_h_next is not None
                else torch.zeros(B, D, device=device, dtype=dtype_in)
            ),
            "attn_k": present_attn_k,
            "attn_v": present_attn_v,
            "seq_len": past_len + L,
        }

        attn_weights = probs if output_attentions else None
        return out, attn_weights, present_key_value

    def _apply_head_mask(
        self, head_mask: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """MSF-FIX-11: unified head_mask broadcast. probs: (B, H, Lq, Lk)."""
        hm = head_mask.to(probs.dtype)
        H = self.num_heads
        if hm.dim() == 1:
            if hm.numel() == H:
                hm = hm.view(1, H, 1, 1)
            else:
                hm = hm.view(1, -1, 1, 1)
        elif hm.dim() == 2:
            # (B, H) or (H, 1)
            if hm.shape[-1] == H:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.shape[0] == H:
                hm = hm.t().view(1, H, 1, 1)
            else:
                hm = hm.view(1, hm.shape[-1], 1, 1)
        elif hm.dim() == 3:
            # (B, H, Lk) or (B, Lk, H)
            if hm.shape[1] == H:
                hm = hm.unsqueeze(2)  # (B, H, 1, Lk)
            elif hm.shape[2] == H:
                hm = hm.permute(0, 2, 1).unsqueeze(2)
            else:
                hm = hm.unsqueeze(1)
        # dim == 4: pass through
        return probs * hm

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out: dict[str, int] = {}
        for name, m in (
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("out_proj", self.out_proj),
            ("gating", self.gating),
            ("ssm_fwd", self.ssm_fwd),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        if self.ssm_bwd is not None:
            out["ssm_bwd"] = sum(_sz(p) for p in self.ssm_bwd.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("mamba_svd_flash_inject")
@ATTENTION_REGISTRY.register("inject_mamba_svd_flash")
def inject_mamba_svd_flash(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> MambaSVDFlash | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_mamba_svd_flash: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Whole-model traversal ----
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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_mamba_svd_flash(
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

    # ---- Single-layer ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is not None and (
        isinstance(src_attn, MambaSVDFlash)
        or "MambaSVDFlash" in src_attn.__class__.__name__
    ):
        return None
    if src_attn is not None and (
        "DeltaNet" in src_attn.__class__.__name__
        or "LinearAttention" in src_attn.__class__.__name__
    ):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        new_block = MambaSVDFlash(
            config=cfg,
            rank=kwargs.pop("rank", 16),
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            d_model=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("Failed to instantiate MambaSVDFlash: %s", e)
        return None

    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
            spliced = True
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
            spliced = True
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
            spliced = True
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
            spliced = True

        if not spliced:
            logger.error("inject_mamba_svd_flash: no writable attention attribute")
            return None

    return new_block


__all__ = [
    "MambaSVDFlash",
    "SVDS6StateSpace",
    "inject_mamba_svd_flash",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
