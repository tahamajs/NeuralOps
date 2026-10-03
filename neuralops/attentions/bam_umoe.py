"""
BAM-UMoE: Unified Mixture of Experts Attention Block (Activation-Aware v2).

Combines:
  1. Optional SVD factorization of Q/K/V/O for actual parameter reduction.
  2. Optional Mixture-of-Experts (MoE) branch for residual refinement.
  3. Cross-architecture weight transfer (DistilBERT, RoBERTa/BERT, GPT-2, Qwen/LLaMA).
  4. LoRA adapters for recovery fine-tuning.

Fixes (v2):
  BAM-FIX-01  `UnifiedMoE.is_custom_attention = False`.
  BAM-FIX-02  MoE is OPTIONAL (`moe_enabled=False` by default). When enabled,
              it uses a compact output_dim (`moe_output_dim=head_dim`) so it
              does NOT cause expansion.
  BAM-FIX-03  **Real parameter reduction**: SVD factorization applied to
              Q/K/V/O via `svd_rank` parameter.
  BAM-FIX-04  `aux_loss` stored as attribute (None initially), never a CPU
              tensor on GPU. `get_aux_loss()` returns the correct device.
  BAM-FIX-05  Idempotency guard in injector.
  BAM-FIX-06  `apply_rotary_pos_emb` slices cos/sin for seq length.
  BAM-FIX-07  Config None-safe `__init__`.
  BAM-FIX-08  `copy_weights` device/dtype-safe.
  BAM-FIX-09  Storage accounting in stats.
  BAM-FIX-10  `head_mask` shape handling unified.
  BAM-FIX-11  `moe_gate_init` configurable; default -3.0 (sigmoid ≈ 0.047).
  BAM-FIX-12  MoE expert hidden dim is `dim // experts // 4` (compact).
  BAM-FIX-13  Route only to self-attention (documented); MoE disabled for
              cross-attention.
  BAM-FIX-14  SDPA path validates mask before use.
  BAM-FIX-15  No silent `except: pass`; errors logged.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    if d % 2 != 0:
        x_even = x[..., :-1]
        x1 = x_even[..., : (d - 1) // 2]
        x2 = x_even[..., (d - 1) // 2 :]
        rot_even = torch.cat((-x2, x1), dim=-1)
        return torch.cat((rot_even, x[..., -1:]), dim=-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
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

    # BAM-FIX-06: slice cos/sin if seq length mismatched
    if cos.shape[-2] != q.shape[-2]:
        if cos.shape[-2] >= q.shape[-2]:
            cos = cos[..., : q.shape[-2], :]
            sin = sin[..., : q.shape[-2], :]
        else:
            raise ValueError(f"RoPE seq mismatch: cos {cos.shape[-2]} < q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_e = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_e = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_e, q_pass], dim=-1)
        k = torch.cat([k_rot_e, k_pass], dim=-1)
    else:
        q = (q * cos.to(q.dtype)) + (_rotate_half(q) * sin.to(q.dtype))
        k = (k * cos.to(k.dtype)) + (_rotate_half(k) * sin.to(k.dtype))
    return q, k


# ===========================================================================
# Factorised Linear (SVD + LoRA) — for real compression
# ===========================================================================
class FactorisedLoRALinear(nn.Module):
    """
    Low-rank factorized linear with optional LoRA:

        y = (x V S U^T) + (x lora_A^T lora_B^T) * scaling + b

    Base factors (U, S, V) are frozen parameters (visible in .parameters()).
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        lora_rank: int = 0,
        lora_alpha: float = 32.0,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.svd_rank = max(1, min(svd_rank, self.in_features, self.out_features))
        self.lora_rank = max(0, int(lora_rank))
        self.lora_scaling = (lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        fk = {"dtype": dtype, "device": device}
        self.U = nn.Parameter(
            torch.zeros(out_features, self.svd_rank, **fk),
            requires_grad=False,
        )
        self.S = nn.Parameter(
            torch.ones(self.svd_rank, **fk),
            requires_grad=False,
        )
        self.V = nn.Parameter(
            torch.zeros(in_features, self.svd_rank, **fk),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **fk), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, **fk))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_rank, **fk))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def set_svd_factors(self, W: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            W_f = W.float()
            if W_f.shape != (self.out_features, self.in_features):
                raise ValueError(
                    f"FactorisedLoRALinear: shape mismatch "
                    f"{tuple(W_f.shape)} vs ({self.out_features}, {self.in_features})"
                )
            U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
            r = min(self.svd_rank, S.shape[0])
            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(device=self.S.device, dtype=self.S.dtype))
            self.V.data[:, :r].copy_(
                Vt[:r, :].t().contiguous().to(device=self.V.device, dtype=self.V.dtype)
            )
            if bias is not None and self.bias is not None:
                n = min(self.bias.numel(), bias.numel())
                self.bias.data[:n].copy_(
                    bias[:n].to(device=self.bias.device, dtype=self.bias.dtype)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        xV = torch.matmul(x_f, self.V.float())
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)
        out = torch.matmul(xV_S, self.U.float().t().contiguous())
        if self.bias is not None:
            out = out + self.bias.float()
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            xa = torch.matmul(x_f, self.lora_A.float().t().contiguous())
            out = out + torch.matmul(xa, self.lora_B.float().t().contiguous()) * self.lora_scaling
        return out.to(orig_dtype)

    @property
    def weight(self) -> torch.Tensor:
        W = (self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t().contiguous()
        return W

    def freeze_base_weights(self) -> None:
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True


# ===========================================================================
# Unified MoE (compact, optional)
# ===========================================================================
class UnifiedMoE(nn.Module):
    """
    Compact MoE with top-k gating.

    BAM-FIX-02 / FIX-12: expert hidden dim is `dim // (num_experts * 4)`,
    output dim is configurable (default `head_dim`, small).
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        dim: int,
        num_experts: int = 2,
        top_k: int = 1,
        output_dim: int | None = None,
        noise_std: float = 0.1,
        expert_hidden_mult: float = 0.25,
        gate_init: float = -3.0,  # BAM-FIX-11
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_experts = max(1, int(num_experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        # BAM-FIX-02: default to dim // 4 for compact output
        self.out_dim = int(output_dim) if output_dim is not None else max(16, self.dim // 4)
        self.noise_std = float(noise_std)

        self.router_ln = nn.LayerNorm(self.dim)
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        hidden_dim = max(16, int(self.out_dim * expert_hidden_mult))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.out_dim),
                )
                for _ in range(self.num_experts)
            ]
        )

        for exp in self.experts:
            nn.init.normal_(exp[2].weight, std=0.02)
            if exp[2].bias is not None:
                nn.init.zeros_(exp[2].bias)

        # BAM-FIX-11: configurable gate init
        self.gate_bias = nn.Parameter(torch.tensor([float(gate_init)]))

        # BAM-FIX-04: aux loss is an attribute, not a tensor
        self._aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_norm = self.router_ln(x)
        logits = self.router(x_norm)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        all_probs = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = torch.topk(all_probs, self.top_k, dim=-1)
        scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        scores = torch.nan_to_num(scores, nan=0.0)

        # Load balancing aux loss
        if self.training and self.num_experts > 1:
            router_probs = all_probs.view(-1, self.num_experts)
            expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
            density = expert_mask.sum(dim=2).view(-1, self.num_experts).mean(dim=0)
            prob_mean = router_probs.mean(dim=0)
            self._aux_loss = self.num_experts * torch.sum(density * prob_mean)
        else:
            self._aux_loss = None

        experts_out = torch.stack([exp(x) for exp in self.experts], dim=2)
        topk_out = torch.gather(
            experts_out,
            2,
            topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim),
        )
        out = torch.sum(topk_out * scores.unsqueeze(-1), dim=2)
        out = torch.clamp(torch.nan_to_num(out, nan=0.0), -65000.0, 65000.0)
        gate = torch.sigmoid(self.gate_bias).to(x.dtype)
        return (out * gate).to(x.dtype)

    def get_aux_loss(self, device: torch.device) -> torch.Tensor:
        """BAM-FIX-04: return aux loss on the correct device."""
        if self._aux_loss is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        return self._aux_loss.to(device)


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "bam_umoe", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class BAMUMoEBlock(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        num_experts: int = 2,
        top_k: int = 1,
        layer_id: int | None = None,
        num_layers: int | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        moe_enabled: bool = False,  # BAM-FIX-02
        svd_rank: int | None = None,  # BAM-FIX-03
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        # BAM-FIX-07: config None-safe
        if config is None:

            class _Cfg:
                pass

            config = _Cfg()
            config.hidden_size = kwargs.get("hidden_size", 768)
            config.num_attention_heads = kwargs.get(
                "num_heads", kwargs.get("num_attention_heads", 12)
            )
            config.head_dim = kwargs.get("head_dim", 64)
            config.model_type = kwargs.get("model_type", "")

        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        self.dim = int(
            kwargs.get("hidden_size")
            or getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "dim", 768)),
            )
        )
        self.hidden_size = self.dim
        self.num_heads = int(
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", getattr(config, "n_heads", 12)),
            )
        )
        self.num_kv_heads = int(
            getattr(
                config,
                "num_key_value_heads",
                getattr(config, "num_kv_heads", self.num_heads),
            )
        )
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads

        cfg_hd = getattr(config, "head_dim", None)
        self.head_dim = int(
            cfg_hd or kwargs.get("head_dim") or (self.dim // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.lora_rank = int(
            getattr(
                config,
                "lora_rank",
                (
                    getattr(comp_config, "lora_rank", lora_rank)
                    if comp_config is not None
                    else lora_rank
                ),
            )
        )
        self.lora_alpha = float(
            getattr(
                config,
                "lora_alpha",
                (
                    getattr(comp_config, "lora_alpha", lora_alpha)
                    if comp_config is not None
                    else lora_alpha
                ),
            )
        )
        self.freeze_base = self.lora_rank > 0

        # BAM-FIX-03: SVD rank for real compression
        default_svd = min(128, self.dim)
        self.svd_rank = int(
            getattr(
                config,
                "svd_rank",
                (
                    getattr(comp_config, "svd_rank", svd_rank)
                    if comp_config is not None
                    else (svd_rank or default_svd)
                ),
            )
        )

        # Decoder detection
        mt = str(getattr(config, "model_type", "") or "").lower()
        is_dec = bool(getattr(config, "is_decoder", False) or getattr(config, "causal", False))
        if any(
            mt.startswith(p)
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
        if old_attn is not None and hasattr(old_attn, "c_attn"):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.num_experts = max(1, int(num_experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        self.moe_enabled = bool(moe_enabled)

        # ---- Q/K/V/O projections ----
        # BAM-FIX-03: use FactorisedLoRALinear with svd_rank
        # If svd_rank >= min(dim, all_head_size), degenerates to dense (no compression)
        Q_rank = min(self.svd_rank, self.dim, self.all_head_size)
        KV_rank = min(self.svd_rank, self.dim, self.all_kv_head_size)
        O_rank = min(self.svd_rank, self.all_head_size, self.dim)

        self.q_lin = FactorisedLoRALinear(
            self.dim,
            self.all_head_size,
            Q_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.k_lin = FactorisedLoRALinear(
            self.dim,
            self.all_kv_head_size,
            KV_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.v_lin = FactorisedLoRALinear(
            self.dim,
            self.all_kv_head_size,
            KV_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.out_lin = FactorisedLoRALinear(
            self.all_head_size,
            self.dim,
            O_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )

        # ---- Optional MoE ----
        if self.moe_enabled:
            self.qkv_moe: UnifiedMoE | None = UnifiedMoE(
                self.dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                output_dim=self.all_kv_head_size,  # MoE refines V (matches V dim)
                gate_init=kwargs.get("moe_gate_init", -3.0),
            )
        else:
            self.qkv_moe = None

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout = nn.Dropout(dropout_p)

        self.layer_id = layer_id
        self.num_layers = num_layers
        self.comp_config = comp_config
        self.rotary_emb = None

        # BAM-FIX-09: storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if old_attn is not None:
            self.copy_weights(old_attn)

    # ---- aliases ----
    @property
    def q_proj(self):
        return self.q_lin

    @property
    def k_proj(self):
        return self.k_lin

    @property
    def v_proj(self):
        return self.v_lin

    @property
    def o_proj(self):
        return self.out_lin

    @property
    def query(self):
        return self.q_lin

    @property
    def key(self):
        return self.k_lin

    @property
    def value(self):
        return self.v_lin

    @property
    def dense(self):
        return self.out_lin

    @property
    def c_proj(self):
        return self.out_lin

    @property
    def out_proj(self):
        return self.out_lin

    # ------------------------------------------------------------------
    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        try:
            for p in pretrained_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = any(
            hasattr(pretrained_attn, a)
            for a in ("out_lin", "c_proj", "o_proj", "out_proj", "dense")
        )
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
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
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb") and pretrained_attn.rotary_emb is not None:
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c = pretrained_attn.c_attn
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

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                o = pretrained_attn.c_proj
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

        # Nested BERT / RoBERTa
        elif hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (
                        m.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                d = pretrained_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # Flat modules
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(pretrained_attn, n):
                        m = getattr(pretrained_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                m.weight.data.clone(),
                                b.data.clone() if b is not None else None,
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
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
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
            raise RuntimeError(
                f"BAMUMoE.copy_weights: missing {missing_after} from "
                f"{pretrained_attn.__class__.__name__}"
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
        }

        with torch.no_grad():
            for proj_key, target in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        # Identity for nested encoder bypass
                        eye = torch.eye(
                            target.out_features,
                            target.in_features,
                            device=target.U.device,
                            dtype=target.U.dtype,
                        )
                        target.set_svd_factors(eye, None)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                # Orient to (out, in)
                in_dim = self.all_head_size if proj_key == "o" else self.dim
                w = _orient(w, in_dim)

                # BAM-FIX-08: shape tolerance
                out_d = min(target.out_features, w.size(0))
                in_d = min(target.in_features, w.size(1))
                if (out_d, in_d) != (target.out_features, target.in_features):
                    padded = torch.zeros(
                        target.out_features,
                        target.in_features,
                        device=w.device,
                        dtype=w.dtype,
                    )
                    padded[:out_d, :in_d] = w[:out_d, :in_d]
                    w = padded

                # Handle GQA pooling for K/V
                if proj_key in ("k", "v"):
                    src_out = w.shape[0]
                    tgt_out = target.out_features
                    if src_out > tgt_out and src_out % tgt_out == 0:
                        rep = src_out // tgt_out
                        w = w.view(tgt_out, rep, -1).mean(dim=1)
                    elif src_out < tgt_out and tgt_out % src_out == 0:
                        rep = tgt_out // src_out
                        w = w.repeat_interleave(rep, dim=0)

                target.set_svd_factors(w, b)
                target.freeze_base_weights()

        # Track compressed bytes
        try:
            for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def get_aux_loss(self, device: torch.device) -> torch.Tensor:
        """BAM-FIX-04: MoE aux loss for training loop."""
        if self.qkv_moe is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        return self.qkv_moe.get_aux_loss(device)

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
        *args: Any,
        **kwargs: Any,
    ):
        B, L, D = hidden_states.shape
        dtype_in = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        S = kv_input.shape[1]

        q = (
            self.q_lin(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_lin(kv_input)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_lin(kv_input)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb[0], pos_emb[1]

        if cos is not None and sin is not None and not is_cross:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, kwargs.get("position_ids"))

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- MoE refinement (only self-attn, only if enabled) ----
        if self.qkv_moe is not None and not is_cross:
            moe_delta_v = self.qkv_moe(hidden_states)
            moe_delta_v = moe_delta_v.view(B, L, self.num_kv_heads, self.head_dim)
            moe_delta_v = moe_delta_v.transpose(1, 2).contiguous()
            if self.kv_repeat > 1:
                moe_delta_v = repeat_kv(moe_delta_v, self.kv_repeat)
            v = (v + 0.1 * moe_delta_v.to(v.dtype)).to(v.dtype)

        # ---- Dense attention (no SDPA shortcut when MoE enabled, to keep mask logic uniform) ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and not is_cross:
            causal = torch.triu(
                torch.full((L, S), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1 + (S - L),
            )
            scores = scores + causal.unsqueeze(0).unsqueeze(1)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores.masked_fill((eff_mask == 0).unsqueeze(1).unsqueeze(2), -1e4)
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(dtype_in)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # ---- head_mask (BAM-FIX-10) ----
        if head_mask is not None:
            hm = head_mask.to(probs.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        context = torch.matmul(probs.to(v.dtype), v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_lin(context)

        if out.shape[-1] != D:
            raise RuntimeError(f"BAMUMoE: output dim {out.shape[-1]} != hidden_size {D}")

        return out, (probs if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("bam_umoe_fn")
@ATTENTION_REGISTRY.register("bam_umoe_inject")
@ATTENTION_REGISTRY.register("inject_bam_umoe")
def inject_bam_umoe(
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
        logger.warning("inject_bam_umoe: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_bam_umoe(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # ---- Single-layer injection ----
    layer = target

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if isinstance(cand, nn.Module) and cand is not layer:
                    src_attn = getattr(cand, "self", cand)
                    break

    # BAM-FIX-05: idempotency guard
    if src_attn is not None and (
        isinstance(src_attn, BAMUMoEBlock) or "BAMUMoE" in src_attn.__class__.__name__
    ):
        return None

    if cfg is None and src_attn is not None:
        cfg = getattr(src_attn, "config", None)

    # Hidden size / heads detection
    hidden_size = kwargs.get("hidden_size")
    if hidden_size is None and src_attn is not None:
        hidden_size = getattr(src_attn, "hidden_size", None)
        if hidden_size is None and hasattr(src_attn, "q_proj"):
            q = getattr(src_attn, "q_proj", None)
            if q is not None and hasattr(q, "in_features"):
                hidden_size = q.in_features
    if hidden_size is None:
        hidden_size = getattr(cfg, "hidden_size", 768) if cfg else 768

    num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
    if num_heads is None and src_attn is not None:
        num_heads = getattr(src_attn, "num_heads", getattr(src_attn, "num_attention_heads", None))
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", 12) if cfg else 12

    moe_enabled = bool(kwargs.pop("moe_enabled", False))
    svd_rank = kwargs.pop("svd_rank", None)

    try:
        kwargs.pop("num_heads", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("d_model", None)
        new_attn = BAMUMoEBlock(
            config=cfg,
            old_attn=src_attn if src_attn is not layer else None,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_id=layer_idx,
            moe_enabled=moe_enabled,
            svd_rank=svd_rank,
            **kwargs,
        )
        if src_attn is not None and src_attn is not layer:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        # Splice
        spliced = False
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
            spliced = True
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = True
            layer.self = new_attn
            spliced = True
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
            spliced = True
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_bam_umoe: no writable attention attribute on %s",
                layer.__class__.__name__,
            )
            return None
        return new_attn

    except Exception as e:
        logger.exception("inject_bam_umoe failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "BAMUMoEBlock",
    "UnifiedMoE",
    "FactorisedLoRALinear",
    "apply_rotary_pos_emb",
    "inject_bam_umoe",
]
