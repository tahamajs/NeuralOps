"""
SpectronAttention: Importance-Aware Low-Rank + Sparse Attention (Activation-Aware).

Dynamically allocates non-uniform low-rank capacities across attention projection
matrices under a global parameter budget via Lagrangian spectral entropy
optimization, augmented by trainable magnitude-pruned sparse residuals.

Mathematical Mechanism:
    For projection matrices {W_i} with singular values {σ_ik}:
    1. Lagrangian Spectral Allocation (on ACTIVATION-WHITENED weights):
         min_{r_i} Σ (1 - Σ_{k≤r_i} σ²_ik / Σ_k σ²_ik)
         s.t.     Σ r_i (d_in + d_out) ≤ B_target
       Solved via bisection on λ.

    2. Activation-Aware Factorization:
         C_X = E[x x^T]                  (EMA covariance)
         C_X = L L^T                     (Cholesky whitening)
         W_w = W L
         U Σ V^T = SVD(W_w)
         W ≈ (U_r Σ_r) (L^{-1} V_r^T)    (whitened low-rank)

    3. Optional Sparse Residual:
         y = x W_down^T W_up^T + x S^T + b
         with ‖S‖_0 ≤ (1 - ρ) · numel(W)

"""

from __future__ import annotations

import logging
import math
import os
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)
logger = log


# ======================================================================
# 1. Config helper
# ======================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ======================================================================
# 2. Spectral utilities
# ======================================================================
def spectral_entropy(singular_values: np.ndarray) -> float:
    s2 = singular_values.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return 0.0
    p = s2 / total
    p = p[p > 1e-5]
    return float(-np.sum(p * np.log(p)))


def cumulative_entropy_fraction(s: np.ndarray, max_rank: int) -> np.ndarray:
    s2 = s.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s2[:max_rank]) / total
    return cum


# ======================================================================
# 3. Lagrangian rank allocator (ON WHITENED weights)
# ======================================================================
def allocate_ranks_spectron(
    weight_matrices: list[np.ndarray],
    target_budget: int,
    names: list[str],
    max_rank_factor: float = 0.75,
    lambda_range: tuple[float, float] = (1e-5, 5.0),
    steps: int = 100,
    whitening_factors: list[np.ndarray | None] | None = None,
) -> dict[str, int]:
    """
    Lagrangian optimization allocating rank per weight matrix under a
    parameter budget.

    SP-FIX-01: if `whitening_factors` is given, the SVD is computed on the
    whitened weight W @ L, so the allocation reflects the *activation-aware*
    spectral profile — not the raw weight spectrum.
    """
    cost_tables, param_tables = [], []
    for i, W in enumerate(weight_matrices):
        d_out, d_in = W.shape
        max_r = min(d_in, d_out, max(1, int(max_rank_factor * min(d_in, d_out))))

        W_eff = W
        if whitening_factors is not None and whitening_factors[i] is not None:
            L = whitening_factors[i]
            if L.shape == (d_in, d_in):
                W_eff = W @ L

        s = np.linalg.svd(W_eff.astype(np.float64), compute_uv=False)
        cum_ent = cumulative_entropy_fraction(s, max_r)
        cost = 1.0 - cum_ent
        cost_tables.append(cost)
        param_tables.append((np.arange(1, max_r + 1) * (d_in + d_out)).astype(np.float64))

    low, high = lambda_range
    best_ranks = None
    for _ in range(steps):
        lam = 0.5 * (low + high)
        ranks = []
        total_params = 0.0
        for i in range(len(weight_matrices)):
            obj = cost_tables[i] + lam * param_tables[i]
            idx = int(np.argmin(obj))
            ranks.append(idx + 1)
            total_params += param_tables[i][idx]
        if total_params <= target_budget:
            best_ranks = ranks.copy()
            high = lam
        else:
            low = lam

    if best_ranks is None:
        best_ranks = [1] * len(weight_matrices)
        log.warning("Spectron: budget too tight → fallback to rank=1 everywhere.")
    return {n: r for n, r in zip(names, best_ranks)}


# ======================================================================
# 4. Low-Rank + Sparse linear layer
# ======================================================================
class LowRankSparseLinear(nn.Module):
    """
    y = up(down(x)) + F.linear(x, S) + b

    S is a trainable sparse residual. Its L1 gradient encourages zero
    entries; periodic `prune_sparse` enforces hard threshold.

    SP-FIX-02: bias stored as a proper nn.Parameter.
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        down: nn.Linear,
        up: nn.Linear,
        sparse_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.down = down
        self.up = up
        self.sparse_weight = nn.Parameter(sparse_weight)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

        self.in_features = down.in_features
        self.out_features = up.out_features

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight) + self.sparse_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SP-FIX-11: dtype-safe
        down_w = self.down.weight.to(x.dtype)
        up_w = self.up.weight.to(x.dtype)
        up_b = self.up.bias.to(x.dtype) if self.up.bias is not None else None

        x_lr = F.linear(F.linear(x, down_w), up_w, up_b)
        x_sp = F.linear(x, self.sparse_weight.to(x.dtype))
        out = x_lr + x_sp
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def prune_sparse(self, sparsity: float) -> None:
        """Hard threshold: keep only top (1 - sparsity) fraction of entries."""
        with torch.no_grad():
            w = self.sparse_weight.data
            if w.numel() == 0:
                return
            keep = int(w.numel() * (1.0 - sparsity))
            if keep <= 0:
                w.zero_()
                return
            if keep >= w.numel():
                return
            flat = w.abs().view(-1)
            kth = torch.kthvalue(flat, w.numel() - keep).values
            w.mul_(w.abs() >= kth)

    def sparse_l1_loss(self) -> torch.Tensor:
        return self.sparse_weight.abs().sum()


# ======================================================================
# 5. RoPE helpers
# ======================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 : rotary_dim]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

    q_rot_emb = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_emb = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_emb, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_emb
    k_out = torch.cat([k_rot_emb, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_emb
    return q_out, k_out


def _unpack_weight_bias(proj: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if proj is None:
        return None, None
    if hasattr(proj, "base_layer"):
        proj = proj.base_layer
    w = getattr(proj, "weight", None)
    b = getattr(proj, "bias", None)
    if w is not None:
        w = w.detach()
        cls_p = getattr(proj, "__class__", type(proj)).__name__
        if (
            hasattr(proj, "nf")
            or "Conv1D" in cls_p
            or (w.ndim == 2 and w.shape[0] < w.shape[1] and "Linear" not in cls_p)
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.detach()
    return w, b


# ======================================================================
# 6. Activation collection (EMA covariance)
# ======================================================================
def _autocast_off(x: torch.Tensor):
    import contextlib

    if hasattr(torch, "autocast") and x.device.type in ("cuda", "cpu"):
        try:
            return torch.autocast(device_type=x.device.type, enabled=False)
        except Exception:
            pass
    return contextlib.nullcontext()


# ======================================================================
# 7. SpectronAttention
# ======================================================================
@ATTENTION_REGISTRY.register("spectron", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("spectron_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("spectron_attn", supports_causal=True, supports_gqa=True)
class SpectronAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        old_attn: Any | None = None,
        dropout: float = 0.1,
        layer_idx: int = 0,
        ranks: dict[str, int] | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        add_sparse: bool = False,
        sparsity_level: float = 0.7,
        whitening: dict[str, torch.Tensor] | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        model_config = config or kwargs.get("model_config")
        super().__init__(model_config)

        torch.manual_seed(seed)

        self.hidden_size = int(
            _cfg_get(model_config, "hidden_size", kwargs.get("hidden_size", 768))
        )
        self.num_heads = int(
            _cfg_get(
                model_config,
                "num_attention_heads",
                _cfg_get(model_config, "num_heads", kwargs.get("num_attention_heads", 12)),
            )
        )
        self.head_dim = int(
            _cfg_get(
                model_config,
                "head_dim",
                kwargs.get("head_dim", self.hidden_size // max(1, self.num_heads)),
            )
        )

        cfg_kv = _cfg_get(
            model_config,
            "num_key_value_heads",
            _cfg_get(model_config, "num_kv_heads", None),
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.q_head_size = self.num_heads * self.head_dim
        self.all_head_size = self.q_head_size

        # SP-FIX-04: explicit decoder detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(model_config, "model_type", "") or "").lower()
            _dec_pfx = (
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
            self.is_decoder = (
                bool(_cfg_get(model_config, "is_decoder", False))
                or bool(_cfg_get(model_config, "causal", False))
                or any(_mt.startswith(p) for p in _dec_pfx)
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder = False

        # Ranks (with global clamp)
        default_r = int(_cfg_get(model_config, "svd_rank", _cfg_get(model_config, "rank", 64)))
        default_r = max(1, min(default_r, self.hidden_size, self.all_head_size))
        self.ranks: dict[str, int] = {}
        for k in ("q", "k", "v", "o"):
            val = ranks.get(k, default_r) if ranks else default_r
            self.ranks[k] = max(1, min(int(val), self.hidden_size, self.all_head_size))

        self.add_sparse = bool(add_sparse)
        self.sparsity_level = float(sparsity_level)
        self.lora_rank = max(0, int(lora_rank)) if lora_rank else 0
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        # Whitening factors per projection (from calibration)
        self.whitening = whitening or {}

        # ---- factorised base projections (SVD) ----
        self.q_down = nn.Linear(self.hidden_size, self.ranks["q"], bias=False)
        self.q_up = nn.Linear(self.ranks["q"], self.q_head_size, bias=True)
        self.k_down = nn.Linear(self.hidden_size, self.ranks["k"], bias=False)
        self.k_up = nn.Linear(self.ranks["k"], self.kv_head_size, bias=True)
        self.v_down = nn.Linear(self.hidden_size, self.ranks["v"], bias=False)
        self.v_up = nn.Linear(self.ranks["v"], self.kv_head_size, bias=True)
        self.o_down = nn.Linear(self.q_head_size, self.ranks["o"], bias=False)
        self.o_up = nn.Linear(self.ranks["o"], self.hidden_size, bias=True)

        for mod in (
            self.q_down,
            self.q_up,
            self.k_down,
            self.k_up,
            self.v_down,
            self.v_up,
            self.o_down,
            self.o_up,
        ):
            nn.init.orthogonal_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
            mod.weight.requires_grad = False
            if mod.bias is not None:
                mod.bias.requires_grad = False

        # ---- LoRA adapters ----
        if self.lora_rank > 0:
            self.q_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.q_lora_B = nn.Parameter(torch.zeros(self.q_head_size, self.lora_rank))
            self.k_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.k_lora_B = nn.Parameter(torch.zeros(self.kv_head_size, self.lora_rank))
            self.v_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.v_lora_B = nn.Parameter(torch.zeros(self.kv_head_size, self.lora_rank))
            self.o_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.q_head_size))
            self.o_lora_B = nn.Parameter(torch.zeros(self.hidden_size, self.lora_rank))
            for p_a in (self.q_lora_A, self.k_lora_A, self.v_lora_A, self.o_lora_A):
                nn.init.kaiming_uniform_(p_a, a=math.sqrt(5))
        else:
            self.register_parameter("q_lora_A", None)
            self.register_parameter("q_lora_B", None)
            self.register_parameter("k_lora_A", None)
            self.register_parameter("k_lora_B", None)
            self.register_parameter("v_lora_A", None)
            self.register_parameter("v_lora_B", None)
            self.register_parameter("o_lora_A", None)
            self.register_parameter("o_lora_B", None)

        # ---- optional sparse residuals (init from zero) ----
        if self.add_sparse:
            self.q_sparse = nn.Parameter(torch.zeros(self.q_head_size, self.hidden_size))
            self.k_sparse = nn.Parameter(torch.zeros(self.kv_head_size, self.hidden_size))
            self.v_sparse = nn.Parameter(torch.zeros(self.kv_head_size, self.hidden_size))
            self.o_sparse = nn.Parameter(torch.zeros(self.hidden_size, self.q_head_size))
        else:
            self.register_parameter("q_sparse", None)
            self.register_parameter("k_sparse", None)
            self.register_parameter("v_sparse", None)
            self.register_parameter("o_sparse", None)

        dropout_p = _cfg_get(
            model_config,
            "dropout",
            _cfg_get(
                model_config,
                "hidden_dropout_prob",
                _cfg_get(model_config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout = nn.Dropout(float(dropout_p) if dropout_p is not None else 0.1)

        src_attn = old_attn or kwargs.get("pretrained_attn")
        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # SP-FIX-01: activation-aware factorization
    # ------------------------------------------------------------------
    def _factorise_and_set(
        self,
        W: torch.Tensor,
        bias: torch.Tensor | None,
        down: nn.Linear,
        up: nn.Linear,
        rank: int,
        whitening_L: torch.Tensor | None = None,
    ) -> None:
        """
        Factories W into down/up via activation-aware SVD:

            if whitening_L is None:
                U Σ V^T = SVD(W)
                down = √Σ V^T
                up   = U √Σ
            else:
                U Σ V^T = SVD(W L)
                down = √Σ V^T L^{-1}
                up   = U √Σ
        """
        with torch.no_grad():
            W_f = W.float()
            d_out, d_in = W_f.shape
            r = min(rank, d_out, d_in, down.out_features, up.in_features)
            r = max(r, 1)

            L = None
            if whitening_L is not None and whitening_L.shape == (d_in, d_in):
                L = whitening_L.float().to(W_f.device)

            if L is not None:
                eps = 1e-6 * torch.trace(L @ L.t()).item() / max(1, d_in)
                eye = torch.eye(d_in, device=W_f.device, dtype=W_f.dtype)
                try:
                    L_reg = L + eps * eye  # small regularization
                    W_eff = W_f @ L_reg
                    U, s, Vt = torch.linalg.svd(W_eff, full_matrices=False)
                    L_inv = torch.linalg.inv(L_reg)
                    Vt = Vt @ L_inv  # bring back to original coordinate
                except Exception as e:
                    log.debug("Whitening failed on factorise (%s); fallback.", e)
                    U, s, Vt = torch.linalg.svd(W_f, full_matrices=False)
            else:
                U, s, Vt = torch.linalg.svd(W_f, full_matrices=False)

            sqrt_s = torch.sqrt(torch.clamp(s[:r], min=1e-12))
            down_w = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(down.weight.device, down.weight.dtype)
            up_w = (U[:, :r] * sqrt_s.unsqueeze(0)).to(up.weight.device, up.weight.dtype)

            down.weight.data.zero_()
            up.weight.data.zero_()
            down.weight.data[:r, :].copy_(down_w)
            up.weight.data[:, :r].copy_(up_w)
            if bias is not None and up.bias is not None:
                up.bias.data.copy_(bias.to(up.bias.device, up.bias.dtype))

    # ------------------------------------------------------------------
    # copy_weights with activation-aware factorization
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_proj = any(
                hasattr(old_attn, a)
                for a in (
                    "dense",
                    "out_proj",
                    "c_proj",
                    "out_lin",
                    "o_proj",
                    "wo",
                    "o",
                    "proj",
                )
            )
            if src_cls_name in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            ) or (hasattr(old_attn, "query") and not has_out_proj):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
            else:
                src = old_attn

            # GPT-2 fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_w, c_b = _unpack_weight_bias(src.c_attn)
                if c_w is not None:
                    if c_w.shape[0] == self.hidden_size and c_w.shape[1] != self.hidden_size:
                        c_w = c_w.t().contiguous()
                    q_dim = self.q_head_size
                    k_dim = self.hidden_size
                    v_dim = self.hidden_size
                    w_q = c_w[:q_dim, :]
                    w_k = c_w[q_dim : q_dim + k_dim, :]
                    w_v = c_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                    if c_b is not None:
                        b_q = c_b[:q_dim]
                        b_k = c_b[q_dim : q_dim + k_dim]
                        b_v = c_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w, out_b = _unpack_weight_bias(src.c_proj)
                if out_w is not None:
                    if out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                        out_w = out_w.t().contiguous()
                    w_o, b_o = out_w, out_b

            # BERT / RoBERTa nested
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Qwen / LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            # UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = extracted.get("q_weight")
                        b_q = extracted.get("q_bias")
                        w_k = extracted.get("k_weight")
                        b_k = extracted.get("k_bias")
                        w_v = extracted.get("v_weight")
                        b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as e:
                    log.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None:
                missing.append("query")
            if w_k is None:
                missing.append("key")
            if w_v is None:
                missing.append("value")
            if missing:
                logger.warning("Spectron: missing %s; keeping random init.", missing)
                return

            def _orient(w, out_dim, in_dim):
                if w.shape == (out_dim, in_dim):
                    return w
                if w.shape == (in_dim, out_dim):
                    return w.t().contiguous()
                return w

            # ---- activation-aware factorization using self.whitening ----
            w_q = _orient(w_q, self.q_head_size, self.hidden_size)
            self._factorise_and_set(
                w_q,
                b_q,
                self.q_down,
                self.q_up,
                self.ranks["q"],
                whitening_L=self.whitening.get("q", None),
            )

            # K with GQA pooling
            if w_k.shape[0] == self.kv_head_size:
                w_k = _orient(w_k, self.kv_head_size, self.hidden_size)
                self._factorise_and_set(
                    w_k,
                    b_k,
                    self.k_down,
                    self.k_up,
                    self.ranks["k"],
                    whitening_L=self.whitening.get("k", None),
                )
            else:
                rep = max(1, w_k.shape[0] // self.kv_head_size)
                w_k_pooled = (
                    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                b_k_pooled = (
                    b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    if b_k is not None
                    else None
                )
                self._factorise_and_set(
                    w_k_pooled,
                    b_k_pooled,
                    self.k_down,
                    self.k_up,
                    self.ranks["k"],
                    whitening_L=self.whitening.get("k", None),
                )

            # V with GQA pooling
            if w_v.shape[0] == self.kv_head_size:
                w_v = _orient(w_v, self.kv_head_size, self.hidden_size)
                self._factorise_and_set(
                    w_v,
                    b_v,
                    self.v_down,
                    self.v_up,
                    self.ranks["v"],
                    whitening_L=self.whitening.get("v", None),
                )
            else:
                rep = max(1, w_v.shape[0] // self.kv_head_size)
                w_v_pooled = (
                    w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                b_v_pooled = (
                    b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    if b_v is not None
                    else None
                )
                self._factorise_and_set(
                    w_v_pooled,
                    b_v_pooled,
                    self.v_down,
                    self.v_up,
                    self.ranks["v"],
                    whitening_L=self.whitening.get("v", None),
                )

            # Output projection
            if w_o is not None and not self.is_nested_encoder:
                w_o = _orient(w_o, self.hidden_size, self.q_head_size)
                self._factorise_and_set(
                    w_o,
                    b_o,
                    self.o_down,
                    self.o_up,
                    self.ranks["o"],
                    whitening_L=self.whitening.get("o", None),
                )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _linear_proj(
        self,
        x: torch.Tensor,
        down: nn.Linear,
        up: nn.Linear,
        lora_A: nn.Parameter | None,
        lora_B: nn.Parameter | None,
        sparse_w: nn.Parameter | None,
    ) -> torch.Tensor:
        out = up(down(x))
        if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
            a = lora_A.to(x.dtype)
            b = lora_B.to(x.dtype)
            out = out + F.linear(F.linear(x, a), b) * self.scaling
        if self.add_sparse and sparse_w is not None:
            out = out + F.linear(x, sparse_w.to(x.dtype))
        return out

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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        Q = self._linear_proj(
            hidden_states,
            self.q_down,
            self.q_up,
            self.q_lora_A,
            self.q_lora_B,
            self.q_sparse,
        )
        K = self._linear_proj(
            kv_input,
            self.k_down,
            self.k_up,
            self.k_lora_A,
            self.k_lora_B,
            self.k_sparse,
        )
        V = self._linear_proj(
            kv_input,
            self.v_down,
            self.v_up,
            self.v_lora_A,
            self.v_lora_B,
            self.v_sparse,
        )

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = K.view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V = V.view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len : past_len + N, :]
            sin = sin[..., past_len : past_len + N, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA
        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)

        # SP-FIX-04: correct causal condition
        is_causal = self.is_decoder and not is_cross and N > 1

        # SP-FIX-05: removed padding/truncation hack
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if is_causal:
            causal = (
                torch.triu(
                    torch.full(
                        (N, kv_len),
                        float("-inf"),
                        device=scores.device,
                        dtype=torch.float32,
                    ),
                    diagonal=1 + max(0, kv_len - N),
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            hm = head_mask.to(attn_weights.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            attn_weights = attn_weights * hm

        context = (
            torch.matmul(attn_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(B, N, self.all_head_size)
        )
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self._linear_proj(
                context,
                self.o_down,
                self.o_up,
                self.o_lora_A,
                self.o_lora_B,
                self.o_sparse,
            )

        return output, (attn_weights if output_attentions else None)

    # ------------------------------------------------------------------
    def total_sparse_l1(self) -> torch.Tensor:
        """Sum of L1 norms of all sparse residuals. Add to training loss."""
        if not self.add_sparse:
            return torch.zeros((), device=self.q_down.weight.device)
        total = torch.zeros((), device=self.q_down.weight.device)
        for s in (self.q_sparse, self.k_sparse, self.v_sparse, self.o_sparse):
            if s is not None:
                total = total + s.abs().sum()
        return total

    def prune_all_sparse(self, sparsity: float) -> None:
        with torch.no_grad():
            for s in (self.q_sparse, self.k_sparse, self.v_sparse, self.o_sparse):
                if s is None or s.numel() == 0:
                    continue
                keep = int(s.numel() * (1.0 - sparsity))
                if keep <= 0:
                    s.zero_()
                    continue
                if keep >= s.numel():
                    continue
                flat = s.abs().view(-1)
                kth = torch.kthvalue(flat, s.numel() - keep).values
                s.mul_(s.abs() >= kth)


# ======================================================================
# 8. Multi-architecture weight gathering (SP-FIX-06)
# ======================================================================
def gather_attention_weights(model: Any) -> tuple[list[np.ndarray], list[str]]:
    """
    Return flattened list of Q/K/V/O weight matrices and their layer names.
    Supports DistilBERT, BERT/RoBERTa, GPT-2, LLaMA/Qwen.
    """
    weights: list[np.ndarray] = []
    names: list[str] = []

    def _collect_attention_block(attn, layer_id):
        # GPT-2 fused c_attn
        if hasattr(attn, "c_attn") and hasattr(attn, "c_proj"):
            w = attn.c_attn.weight.data.cpu().numpy()
            if w.shape[0] < w.shape[1]:
                w = w.T
            # split into q, k, v
            q_d = w.shape[0] // 3
            for tag, part in zip(("q", "k", "v"), (w[:q_d], w[q_d : 2 * q_d], w[2 * q_d :])):
                weights.append(part)
                names.append(f"layer{layer_id}_{tag}")
            weights.append(attn.c_proj.weight.data.cpu().numpy())
            names.append(f"layer{layer_id}_o")
            return

        # BERT / RoBERTa nested
        inner = getattr(attn, "self", attn)
        for proj, attr in (("q", "query"), ("k", "key"), ("v", "value")):
            if hasattr(inner, attr):
                w = getattr(inner, attr).weight.data.cpu().numpy()
                weights.append(w)
                names.append(f"layer{layer_id}_{proj}")

        # output
        for attr in ("dense", "out_proj", "o_proj", "out_lin"):
            if hasattr(attn, attr):
                weights.append(getattr(attn, attr).weight.data.cpu().numpy())
                names.append(f"layer{layer_id}_o")
                break

        # Flat modules
        for proj, names_ in (
            ("q", ("q_proj", "q_lin")),
            ("k", ("k_proj", "k_lin")),
            ("v", ("v_proj", "v_lin")),
        ):
            if not any(n.endswith(f"_{proj}") for n in names if f"_{proj}" in n):
                for nm in names_:
                    if hasattr(attn, nm):
                        weights.append(getattr(attn, nm).weight.data.cpu().numpy())
                        names.append(f"layer{layer_id}_{proj}")
                        break

    # Try whole-model traversal
    layer_lists = []
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
        obj = model
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_lists.append(obj)

    for layer_list in layer_lists:
        for lid, layer in enumerate(layer_list):
            for attr in ("attention", "attn", "self_attn"):
                if hasattr(layer, attr):
                    _collect_attention_block(getattr(layer, attr), lid)
                    break
        if weights:
            break

    return weights, names


# ======================================================================
# 9. Injector
# ======================================================================
@ATTENTION_REGISTRY.register("spectron_inject")
@ATTENTION_REGISTRY.register("spectron_attention_inject")
@ATTENTION_REGISTRY.register("inject_spectron_attention")
@ATTENTION_REGISTRY.register("inject_spectron")
def inject_spectron(
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
        logger.warning("inject_spectron: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
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
            res = inject_spectron(
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

    # Single-layer injection
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None:
        src_attn = target

    # SP-FIX-09: idempotency guard
    if isinstance(src_attn, SpectronAttention) or "Spectron" in src_attn.__class__.__name__:
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    h_size = _cfg_get(cfg, "hidden_size", _cfg_get(cfg, "dim", kwargs.get("hidden_size", 768)))
    rank = _cfg_get(
        comp_config,
        "svd_rank",
        _cfg_get(comp_config, "rank", kwargs.pop("rank", min(64, h_size))),
    )
    ranks = kwargs.pop("ranks", None)
    if ranks is None:
        ranks = {"q": rank, "k": rank, "v": rank, "o": rank}
    lr = _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
    dropout = _cfg_get(comp_config, "dropout", kwargs.get("dropout", 0.1))
    whitening = kwargs.pop("whitening", None)

    try:
        new_attn = SpectronAttention(
            config=cfg,
            old_attn=src_attn,
            ranks=ranks,
            lora_rank=lr,
            dropout=dropout,
            whitening=whitening,
            **kwargs,
        )
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.error("Spectron inject failed: %s", e)
        return None

    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn

    return new_attn


# ======================================================================
# 10. Generic sparse pruning callback (SP-FIX-08)
# ======================================================================
try:
    from transformers import TrainerCallback as _HFTrainerCallback
except ImportError:
    _HFTrainerCallback = object


class SparsePruningCallback(_HFTrainerCallback):
    """
    Periodically hard-thresholds the sparse residual of every
    SpectronAttention module in the model.
    """

    def __init__(self, model: Any, prune_freq: int = 200, final_sparsity: float = 0.7):
        self.model = model
        self.prune_freq = prune_freq
        self.final_sparsity = final_sparsity

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.prune_freq == 0:
            for m in self.model.modules():
                if isinstance(m, SpectronAttention) and m.add_sparse:
                    m.prune_all_sparse(self.final_sparsity)


# ======================================================================
# 11. GLUE helpers (unchanged)
# ======================================================================
GLUE_TASK_KEYS: dict[str, tuple[str, str | None]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue_local(task: str) -> Any:
    from datasets import Dataset, DatasetDict

    from neuralops.utils.paths import get_glue_dir

    base = str(get_glue_dir())
    path = os.path.join(base, task)
    splits = {}
    for f in os.listdir(path):
        if f.endswith(".parquet"):
            s = f.split("-")[0]
            splits[s] = Dataset.from_parquet(os.path.join(path, f))
    return DatasetDict(splits)


def get_num_labels_and_type(task: str) -> tuple[int, bool]:
    if task == "stsb":
        return 1, True
    if task == "cola":
        return 2, False
    return {"sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2}[task], False


def preprocess(examples, tokenizer, s1, s2, max_len, is_reg):
    if s2 is None:
        t = tokenizer(examples[s1], truncation=True, padding="max_length", max_length=max_len)
    else:
        t = tokenizer(
            examples[s1],
            examples[s2],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
    t["labels"] = [float(l) for l in examples["label"]] if is_reg else examples["label"]
    return t


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((logits.argmax(-1) == labels).mean().item())}


__all__ = [
    "SpectronAttention",
    "LowRankSparseLinear",
    "SparsePruningCallback",
    "allocate_ranks_spectron",
    "gather_attention_weights",
    "inject_spectron",
    "spectral_entropy",
    "cumulative_entropy_fraction",
]
