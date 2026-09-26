"""
SpectronAttention: Importance-Aware Low-Rank + Sparse Attention.

Dynamically allocates non-uniform low-rank capacities across attention projection matrices under a global parameter
budget via Lagrangian spectral entropy optimization, optionally augmented by trainable magnitude-pruned sparse residuals.

Mathematical Mechanism:
    For projection matrices $\{W_i\}_{i=1}^M$ with singular values $\{\sigma_{i, k}\}$:
    1. **Lagrangian Spectral Allocation:**
       $$\min_{\{r_i\}} \sum_{i=1}^M \left(1 - \frac{\sum_{k=1}^{r_i} \sigma_{i, k}^2}{\sum_{k=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_{i, k}^2}\right) \quad \text{subject to} \quad \sum_{i=1}^M r_i (d_{i, \text{in}} + d_{i, \text{out}}) \le B_{\text{target}}$$
       Solved via bisection on Lagrange multiplier $\lambda$:
       $$r_i^*(\lambda) = \operatorname{arg\,min}_{r} \left[\left(1 - \frac{\sum_{k=1}^r \sigma_{i, k}^2}{\sum_k \sigma_{i, k}^2}\right) + \lambda \cdot r (d_{i, \text{in}} + d_{i, \text{out}})\right]$$
    2. **Low-Rank Factorization with Sparse Residual:**
       $$W_i = W_i^{\text{up}} W_i^{\text{down}} + S_i, \quad \text{where } \|S_i\|_0 \le (1 - \rho) \cdot \operatorname{numel}(W_i)$$
       $$y = x (W_i^{\text{down}})^T (W_i^{\text{up}})^T + x S_i^T + b_i$$
    3. **Multi-Head Attention Computation:**
       $$\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$

Computational Complexity:
    - Projection computation: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} r_i + r_i d_{\text{out}}))$ for dense low-rank path + sparse FLOPs.
    - Scaled dot-product attention: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Per-layer parameter savings: Up to $(1 - \frac{B_{\text{target}}}{B_{\text{orig}}}) \times 100\%$.
"""

import argparse
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import math
import torch
from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

try:
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
except Exception:
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    EarlyStoppingCallback = object
    Trainer = object
    TrainerCallback = object
    TrainingArguments = object
    def set_seed(s): pass
warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)
logger = log


# ======================================================================
# 1. Spectral utilities
# ======================================================================


def spectral_entropy(singular_values: np.ndarray) -> float:
    r"""Compute normalized entropy of squared singular value distribution."""
    s2 = singular_values.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return 0.0
    prob = s2 / total
    prob = prob[prob > 1e-5]
    return float(-np.sum(prob * np.log(prob)))


def cumulative_entropy_fraction(s: np.ndarray, max_rank: int) -> np.ndarray:
    r"""Compute cumulative fraction of spectral energy captured by top-$k$ singular values."""
    s2 = s.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s2[:max_rank]) / total
    return cum


# ======================================================================
# 2. Lagrangian rank allocator (importance‑aware)
# ======================================================================


def allocate_ranks_spectron(
    weight_matrices: List[np.ndarray],
    target_budget: int,
    names: List[str],
    max_rank_factor: float = 0.75,
    lambda_range: Tuple[float, float] = (1e-5, 5.0),
    steps: int = 100,
) -> Dict[str, int]:
    r"""Solve constrained Lagrangian optimization allocating rank per weight matrix under a parameter budget."""
    cost_tables, param_tables = [], []
    for W in weight_matrices:
        d_out, d_in = W.shape
        max_r = min(d_in, d_out, int(max_rank_factor * min(d_in, d_out)))
        max_r = max(1, max_r)
        s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        cum_ent = cumulative_entropy_fraction(s, max_r)
        cost = 1.0 - cum_ent
        cost_tables.append(cost)
        param_tables.append(
            (np.arange(1, max_r + 1) * (d_in + d_out)).astype(np.float64)
        )

    low, high = lambda_range
    best_ranks = None
    for _ in range(steps):
        lam = (low + high) / 2.0
        ranks = []
        total_params = 0
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
        log.warning("Budget too tight, fallback to rank=1 everywhere.")
    return {n: r for n, r in zip(names, best_ranks)}


# ======================================================================
# 3. Low‑Rank + Sparse linear layer
# ======================================================================


class LowRankSparseLinear(nn.Module):
    r"""Composite linear layer: $y = (x W_{\text{down}}^T) W_{\text{up}}^T + x S^T + b$."""

    def __init__(
        self,
        down: nn.Linear,
        up: nn.Linear,
        sparse_weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    **kwargs) -> None:
        super().__init__()
        self.down: nn.Linear = down
        self.up: nn.Linear = up
        self.sparse_weight: nn.Parameter = nn.Parameter(sparse_weight)
        self.bias: Optional[torch.Tensor] = bias
        self.out_features: int = up.out_features
        self.in_features: int = down.in_features

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.up.weight, self.down.weight) + self.sparse_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute low-rank plus sparse forward projection."""
        x_lr = self.up(self.down(x))
        x_sp = F.linear(x, self.sparse_weight)
        return x_lr + x_sp

    def prune_sparse(self, sparsity: float) -> None:
        r"""Zero out smallest magnitude entries to maintain sparsity level."""
        with torch.no_grad():
            w = self.sparse_weight.data.abs()
            if w.numel() == 0:
                return
            k = int(w.numel() * (1 - sparsity))
            if k >= w.numel():
                return
            threshold = torch.kthvalue(w.view(-1), w.numel() - k).values
            mask = w >= threshold
            self.sparse_weight.data *= mask


# ======================================================================
# 4. SpectronAttention module
# ======================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
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
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support."""
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

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_embed
    k_out = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_embed
    return q_out, k_out


def _unpack_weight_bias(proj: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if proj is None:
        return None, None
    if hasattr(proj, "base_layer"):
        proj = proj.base_layer
    w = getattr(proj, "weight", None)
    b = getattr(proj, "bias", None)
    if w is not None:
        w = w.detach()
        cls_p = getattr(proj, "__class__", type(proj)).__name__
        if hasattr(proj, "nf") or "Conv1D" in cls_p or (w.ndim == 2 and w.shape[0] < w.shape[1] and "Linear" not in cls_p):
            w = w.t().contiguous()
    if b is not None:
        b = b.detach()
    return w, b


@ATTENTION_REGISTRY.register("spectron", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("spectron_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("spectron_attn", supports_causal=True, supports_gqa=True)
class SpectronAttention(BaseAttention):
    r"""Importance-aware SVD-compressed attention module with optional trainable sparse residual branch,
    trainable LoRA adapters, GQA support, RoPE support, and causal masking.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[Any] = None,
        dropout: float = 0.1,
        layer_idx: int = 0,
        ranks: Optional[Dict[str, int]] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        add_sparse: bool = False,
        sparsity_level: float = 0.7,
        **kwargs: Any,
    ) -> None:
        model_config = config or kwargs.get("model_config", None)
        super().__init__(model_config)

        # 1. Dimensions & Architecture
        self.hidden_size: int = int(_cfg_get(
            model_config, "hidden_size", kwargs.get("hidden_size", 768)
        ))
        self.num_heads: int = int(_cfg_get(
            model_config,
            "num_attention_heads",
            _cfg_get(model_config, "num_heads", kwargs.get("num_attention_heads", 12)),
        ))
        self.head_dim: int = int(_cfg_get(
            model_config, "head_dim", kwargs.get("head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        ))

        cfg_kv = _cfg_get(
            model_config, "num_key_value_heads", _cfg_get(model_config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(model_config, "model_type", "") or "").lower()
            _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
            self.is_decoder = (
                bool(_cfg_get(model_config, "is_decoder", False))
                or any(_mt.startswith(p) for p in _dec_pfx)
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder: bool = False
        default_r = int(_cfg_get(model_config, "svd_rank", _cfg_get(model_config, "rank", 64)))
        default_r = max(1, min(default_r, self.hidden_size, self.all_head_size))
        self.ranks: Dict[str, int] = {}
        for k in ("q", "k", "v", "o"):
            val = ranks.get(k, default_r) if ranks else default_r
            self.ranks[k] = max(1, min(int(val), self.hidden_size, self.all_head_size))

        self.add_sparse: bool = add_sparse
        self.sparsity_level: float = sparsity_level
        self.lora_rank = max(1, min(int(lora_rank if lora_rank is not None else 16), self.hidden_size, self.all_head_size))
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, self.lora_rank)

        # 3. Factorized projections
        self.q_down = nn.Linear(self.hidden_size, self.ranks["q"], bias=False)
        self.q_up = nn.Linear(self.ranks["q"], self.q_head_size, bias=True)
        self.k_down = nn.Linear(self.hidden_size, self.ranks["k"], bias=False)
        self.k_up = nn.Linear(self.ranks["k"], self.kv_head_size, bias=True)
        self.v_down = nn.Linear(self.hidden_size, self.ranks["v"], bias=False)
        self.v_up = nn.Linear(self.ranks["v"], self.kv_head_size, bias=True)
        self.o_down = nn.Linear(self.q_head_size, self.ranks["o"], bias=False)
        self.o_up = nn.Linear(self.ranks["o"], self.hidden_size, bias=True)

        for mod in [self.q_down, self.q_up, self.k_down, self.k_up, self.v_down, self.v_up, self.o_down, self.o_up]:
            nn.init.orthogonal_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
            mod.weight.requires_grad = False
            if mod.bias is not None:
                mod.bias.requires_grad = False

        # 4. Trainable LoRA parameters
        if self.lora_rank > 0:
            self.q_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.q_lora_B = nn.Parameter(torch.empty(self.q_head_size, self.lora_rank))
            self.k_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.k_lora_B = nn.Parameter(torch.empty(self.kv_head_size, self.lora_rank))
            self.v_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.v_lora_B = nn.Parameter(torch.empty(self.kv_head_size, self.lora_rank))
            self.o_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.q_head_size))
            self.o_lora_B = nn.Parameter(torch.empty(self.hidden_size, self.lora_rank))

            for p_a in (self.q_lora_A, self.k_lora_A, self.v_lora_A, self.o_lora_A):
                nn.init.kaiming_uniform_(p_a, a=math.sqrt(5))
            for p_b in (self.q_lora_B, self.k_lora_B, self.v_lora_B, self.o_lora_B):
                nn.init.zeros_(p_b)
        else:
            self.register_parameter("q_lora_A", None)
            self.register_parameter("q_lora_B", None)
            self.register_parameter("k_lora_A", None)
            self.register_parameter("k_lora_B", None)
            self.register_parameter("v_lora_A", None)
            self.register_parameter("v_lora_B", None)
            self.register_parameter("o_lora_A", None)
            self.register_parameter("o_lora_B", None)

        if add_sparse:
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

        src_attn = old_attn or kwargs.get("pretrained_attn", None)
        if src_attn is not None:
            self.copy_weights(src_attn)

    def _factorise_and_set(
        self, W: torch.Tensor, bias: Optional[torch.Tensor], down: nn.Linear, up: nn.Linear, rank: int
    ) -> None:
        with torch.no_grad():
            W_f = W.float()
            d_out, d_in = W_f.shape
            r = min(rank, d_out, d_in, down.out_features, up.in_features)
            r = max(r, 1)
            U, s, Vt = torch.linalg.svd(W_f, full_matrices=False)
            sqrt_s = torch.sqrt(torch.clamp(s[:r], min=1e-12))
            down_weight = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(device=down.weight.device, dtype=down.weight.dtype)
            up_weight = (U[:, :r] * sqrt_s.unsqueeze(0)).to(device=up.weight.device, dtype=up.weight.dtype)

            down.weight.data.zero_()
            up.weight.data.zero_()
            down.weight.data[:r, :].copy_(down_weight)
            up.weight.data[:, :r].copy_(up_weight)
            if bias is not None and up.bias is not None:
                up.bias.data.copy_(bias.to(device=up.bias.device, dtype=up.bias.dtype))

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Detect nested encoder container or inner self-attention
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_proj = any(hasattr(old_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention") or (
                hasattr(old_attn, "query") and not has_out_proj
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
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
                    w_o = out_w
                    b_o = out_b

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            # Fallback: UniversalAdapter
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

            # Verification of essential projections
            missing = []
            if w_q is None:
                missing.append("query/q_proj")
            if w_k is None:
                missing.append("key/k_proj")
            if w_v is None:
                missing.append("value/v_proj")
            if missing:
                logger.warning(f"Failed to extract required projections: {missing}")
                return

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Orient weights if needed
            def _orient(w: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
                if w.shape == (out_dim, in_dim):
                    return w
                if w.shape == (in_dim, out_dim):
                    return w.t().contiguous()
                return w

            w_q = _orient(w_q, self.q_head_size, self.hidden_size)
            # Factorize and set Q
            self._factorise_and_set(w_q, b_q, self.q_down, self.q_up, self.ranks["q"])

            # Factorize and set K & V with GQA pooling if needed
            if w_k.shape[0] == self.kv_head_size:
                w_k = _orient(w_k, self.kv_head_size, self.hidden_size)
                self._factorise_and_set(w_k, b_k, self.k_down, self.k_up, self.ranks["k"])
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
                self._factorise_and_set(w_k_pooled, b_k_pooled, self.k_down, self.k_up, self.ranks["k"])

            if w_v.shape[0] == self.kv_head_size:
                w_v = _orient(w_v, self.kv_head_size, self.hidden_size)
                self._factorise_and_set(w_v, b_v, self.v_down, self.v_up, self.ranks["v"])
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
                self._factorise_and_set(w_v_pooled, b_v_pooled, self.v_down, self.v_up, self.ranks["v"])

            # Factorize and set Out projection if not nested encoder
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    logger.warning("Copied out_proj weight has near-zero Frobenius norm (< 1e-4)")
                w_o = _orient(w_o, self.hidden_size, self.q_head_size)
                self._factorise_and_set(w_o, b_o, self.o_down, self.o_up, self.ranks["o"])

    def _linear_proj(
        self,
        x: torch.Tensor,
        down: nn.Linear,
        up: nn.Linear,
        lora_A: Optional[nn.Parameter],
        lora_B: Optional[nn.Parameter],
        sparse_w: Optional[nn.Parameter],
    ) -> torch.Tensor:
        out = up(down(x))
        if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
            out = out + F.linear(F.linear(x, lora_A), lora_B) * self.scaling
        if self.add_sparse and sparse_w is not None:
            out = out + F.linear(x, sparse_w)
        return out

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
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Projections with LoRA
        Q = self._linear_proj(hidden_states, self.q_down, self.q_up, self.q_lora_A, self.q_lora_B, self.q_sparse)
        K = self._linear_proj(kv_input, self.k_down, self.k_up, self.k_lora_A, self.k_lora_B, self.k_sparse)
        V = self._linear_proj(kv_input, self.v_down, self.v_up, self.v_lora_A, self.v_lora_B, self.v_sparse)

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = K.view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V = V.view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # 4. Attention scores in FP32
        scale = 1.0 / math.sqrt(self.head_dim)
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

        # 5. Masking
        if self.is_decoder and not is_cross and N > 1:
            causal_mask = torch.triu(
                torch.full((N, kv_len), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

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
        attn_weights = self.dropout(attn_weights)

        # 6. Safe head_mask application
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn_weights = attn_weights * head_mask

        # 7. Context computation & Head Merge
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # 8. Output projection (bypassed if nested encoder)
        if self.is_nested_encoder:
            output = context
        else:
            output = self._linear_proj(context, self.o_down, self.o_up, self.o_lora_A, self.o_lora_B, self.o_sparse)

        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., :hidden_states.shape[-1]]

        return (output, attn_weights if output_attentions else None)


# ======================================================================
# 5. Injector & weight gathering
# ======================================================================


def gather_attention_weights(model: Any) -> Tuple[List[np.ndarray], List[str]]:
    r"""Return flattened list of all Q/K/V/Out weight matrices and their layer names."""
    weights = []
    names = []
    layers = getattr(getattr(model, "distilbert", getattr(model, "transformer", model)), "layer", None)
    if layers is None and hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        layers = model.distilbert.transformer.layer
    if layers is not None:
        for lid, distil_layer in enumerate(layers):
            attn = getattr(distil_layer, "attention", getattr(distil_layer, "attn", None))
            if attn is not None:
                for proj, attr in [
                    ("q", "q_lin"),
                    ("k", "k_lin"),
                    ("v", "v_lin"),
                    ("out", "out_lin"),
                ]:
                    if hasattr(attn, attr):
                        w = getattr(attn, attr).weight.data.cpu().numpy()
                        weights.append(w)
                        names.append(f"layer{lid}_{proj}")
    return weights, names


def inject_spectron(
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
            res = inject_spectron(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    return inject_spectron_layer(
        layer=layer,
        model_config=cfg,
        comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
        layer_idx=layer_idx,
        total_layers=total_layers,
        old_attn=old_attn,
        **kwargs,
    )


@ATTENTION_REGISTRY.register("spectron_inject")
@ATTENTION_REGISTRY.register("spectron_attention_inject")
@ATTENTION_REGISTRY.register("inject_spectron_attention")
@ATTENTION_REGISTRY.register("inject_spectron")
def inject_spectron_layer(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Any]:
    r"""Injects SpectronAttention into a single transformer layer or whole model safely."""
    # Dual injector: if layer is whole model
    if layer is not None and (hasattr(layer, "transformer") or hasattr(layer, "distilbert") or hasattr(layer, "roberta") or (hasattr(layer, "model") and not hasattr(layer, "forward"))):
        inject_spectron(layer, config=model_config or getattr(layer, "config", None), args=comp_config, **kwargs)
        return layer

    if old_attn is None and layer is not None:
        old_attn = getattr(layer, "attention", getattr(layer, "attn", getattr(layer, "self_attn", None)))
        if old_attn is None and hasattr(layer, "self"):
            old_attn = layer
    if old_attn is None:
        return None

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    h_sz = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768)) if model_config else 768
    r = _cfg_get(comp_config, "svd_rank", _cfg_get(comp_config, "rank", kwargs.get("rank", h_sz)))
    ranks = kwargs.pop("ranks", None)
    if ranks is None:
        ranks = {"q": r, "k": r, "v": r, "o": r}

    lr = lora_rank if lora_rank is not None else _cfg_get(comp_config, "lora_rank", 16)
    dropout = _cfg_get(comp_config, "dropout", 0.1)

    try:
        new_attn = SpectronAttention(
            config=model_config,
            old_attn=old_attn,
            ranks=ranks,
            layer_idx=layer_idx,
            lora_rank=lr,
            dropout=dropout,
            **kwargs,
        )
    except Exception as e:
        log.error(f"Failed to instantiate SpectronAttention: {e}")
        return None

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
    elif hasattr(layer, "attn"):
        layer.attn = new_attn
    elif hasattr(layer, "self_attn"):
        layer.self_attn = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn

    return new_attn


# ======================================================================
# 6. Sparse pruning callback (applied during training)
# ======================================================================


class SparsePruningCallback(TrainerCallback):
    r"""Trainer callback periodically magnitude-pruning sparse residual parameters."""

    def __init__(
        self, model: Any, prune_freq: int = 200, final_sparsity: float = 0.7,
    **kwargs) -> None:
        self.model: Any = model
        self.prune_freq: int = prune_freq
        self.final_sparsity: float = final_sparsity

    def on_step_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """On step end."""
        if state.global_step > 0 and state.global_step % self.prune_freq == 0:
            for layer in self.model.distilbert.transformer.layer:
                attn = layer.attention
                if isinstance(attn, SpectronAttention) and attn.add_sparse:
                    for proj in ["q", "k", "v", "o"]:
                        sparse = getattr(attn, f"{proj}_sparse", None)
                        if sparse is not None:
                            with torch.no_grad():
                                w = sparse.data
                                k = int(w.numel() * (1 - self.final_sparsity))
                                if k >= w.numel():
                                    continue
                                threshold = torch.kthvalue(
                                    w.abs().view(-1), w.numel() - k
                                ).values
                                w[w.abs() < threshold] = 0.0


# ======================================================================
# 7. GLUE data loading
# ======================================================================

GLUE_TASK_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
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
    r"""Load GLUE benchmark dataset from local path."""
    from datasets import Dataset, DatasetDict
    from framework.utils.paths import get_glue_dir

    base = str(get_glue_dir())
    path = os.path.join(base, task)
    splits = {}
    for f in os.listdir(path):
        if f.endswith(".parquet"):
            s = f.split("-")[0]
            splits[s] = Dataset.from_parquet(os.path.join(path, f))
    return DatasetDict(splits)


def get_num_labels_and_type(task: str) -> Tuple[int, bool]:
    r"""Return number of output classes and regression flag for GLUE task."""
    if task == "stsb":
        return 1, True
    if task == "cola":
        return 2, False
    m = {"sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2}
    return m[task], False


def preprocess(
    examples: Any,
    tokenizer: Any,
    s1: str,
    s2: Optional[str],
    max_len: int,
    is_reg: bool,
) -> Dict[str, Any]:
    r"""Tokenize text sequences for classification or regression tasks."""
    if s2 is None:
        t = tokenizer(
            examples[s1], truncation=True, padding="max_length", max_length=max_len
        )
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


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    r"""Compute classification accuracy metric from evaluation predictions."""
    logits, labels = eval_pred
    return {"accuracy": float((logits.argmax(-1) == labels).mean().item())}


# ======================================================================
# 8. Main execution entry point
# ======================================================================


def main() -> None:
    r"""CLI entry point for SpectronAttention fine-tuning on GLUE tasks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="./spectron_output")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget_ratio", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--add_sparse", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--prune_freq", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_glue_local(args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True
    )
    num_labels, is_reg = get_num_labels_and_type(args.task)
    s1, s2 = GLUE_TASK_KEYS[args.task]
    tokenized = raw.map(
        lambda x: preprocess(x, tokenizer, s1, s2, args.max_len, is_reg),
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    train_ds = tokenized["train"]
    eval_ds = tokenized.get("validation", tokenized.get("validation_matched"))
    log.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, local_files_only=True
    )

    all_weights, all_names = gather_attention_weights(model)
    original_total = sum(w.size for w in all_weights)
    target = int(original_total * args.budget_ratio)
    log.info(
        f"Original attention params: {original_total:,} -> target {args.budget_ratio*100:.0f}% = {target:,}"
    )

    rank_dict = allocate_ranks_spectron(all_weights, target, all_names)
    avg_rank = float(np.mean(list(rank_dict.values())))
    log.info(f"Allocated ranks (avg {avg_rank:.1f}): {rank_dict}")

    inject_spectron(model, rank_dict, args, config)

    total_steps = (len(train_ds) // args.batch_size) * args.epochs
    warmup = int(0.06 * total_steps)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=warmup,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=False,
        report_to="none",
    )

    callbacks: List[TrainerCallback] = [EarlyStoppingCallback(early_stopping_patience=3)]
    if args.add_sparse:
        callbacks.append(
            SparsePruningCallback(
                model, prune_freq=args.prune_freq, final_sparsity=args.sparsity
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    log.info("Starting training with SpectronAttention...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, args.task))
    log.info("Training complete.")


if __name__ == "__main__":
    main()
