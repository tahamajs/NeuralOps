"""
SVD-Compressed Attention with Importance-Aware Lagrangian Rank Allocation.

Truncated Singular Value Decomposition (SVD) attention compression module with
Lagrangian spectral-entropy optimal rank budgeting and optional low-rank plus sparse residual decomposition.

Mathematical Mechanism:
    For projection weight matrices $\{W_i\}_{i=1}^M$ across all transformer attention sub-layers:
    1. **Spectral Information Loss & Parameter Cost:**
       $$E_i(r) = \frac{\sum_{j=1}^r \sigma_{i, j}^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_{i, j}^2}, \quad \operatorname{Cost}_i(r) = 1 - E_i(r)$$
       $$\operatorname{Params}_i(r) = r \cdot (d_{i, \text{in}} + d_{i, \text{out}})$$
    2. **Lagrangian Global Rank Optimization:**
       $$\min_{\{r_i\}} \sum_{i=1}^M \operatorname{Cost}_i(r_i) \quad \text{subject to} \quad \sum_{i=1}^M \operatorname{Params}_i(r_i) \le B_{\text{target}}$$
    3. **Low-Rank Factorized Multi-Head Projections:**
       $$W_i \approx W_{i}^{\text{up}} W_{i}^{\text{down}}, \quad W_{i}^{\text{down}} \in \mathbb{R}^{r_i \times d_{\text{in}}}, \, W_{i}^{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r_i}$$
       $$\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$

Computational Complexity:
    - Low-rank projection per token: $\mathcal{O}(d_{\text{in}} r_i + r_i d_{\text{out}})$.
    - Attention matrix evaluation: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} r + r d_{\text{out}}) + B \cdot H \cdot N^2 \cdot d)$.
"""

import argparse
import logging
import os
import warnings
from typing import Optional, Tuple, Any, Union, List, Dict

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
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
        TrainingArguments,
        set_seed,
    )
except Exception:
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    EarlyStoppingCallback = object
    Trainer = object
    TrainingArguments = object
    def set_seed(s): pass

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)


# =============================================================================
# 1. Helper functions for spectral analysis
# =============================================================================


def calculate_spectral_entropy(singular_values: np.ndarray) -> float:
    r"""Compute normalized spectral entropy of singular value distribution."""
    s = singular_values.astype(np.float64)
    s_sum = s.sum()
    if s_sum == 0:
        return 0.0
    prob = s / s_sum
    prob = prob[prob > 1e-5]
    return float(-np.sum(prob * np.log(prob)))


def compute_cumulative_entropy(
    singular_values: np.ndarray, max_rank: int
) -> np.ndarray:
    r"""Compute cumulative spectral energy fraction retained by top-$k$ singular values."""
    s = singular_values.astype(np.float64)
    s_sq = s**2
    total = s_sq.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s_sq[:max_rank]) / total
    return cum


# =============================================================================
# 2. Lagrangian optimisation for rank allocation
# =============================================================================


def lagrangian_rank_allocation(
    weight_matrices: List[np.ndarray],
    target_budget: int,
    layer_proj_names: List[str],
    max_rank_factor: float = 0.5,
    lambda_bracket: Tuple[float, float] = (1e-5, 10.0),
    num_lambda_steps: int = 100,
) -> Dict[str, int]:
    r"""Allocate per-projection ranks under global parameter constraint via binary search on Lagrange multiplier."""
    cost_table = []
    param_table = []

    for W in weight_matrices:
        d_in, d_out = W.shape[1], W.shape[0]
        max_r = min(d_in, d_out, int(max_rank_factor * min(d_in, d_out)))
        max_r = max(1, max_r)

        s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        cum_ent = compute_cumulative_entropy(s, max_r)
        cost = 1.0 - cum_ent
        cost_table.append(cost)

        params_per_r = np.arange(1, max_r + 1) * (d_in + d_out)
        param_table.append(params_per_r)

    low, high = lambda_bracket
    best_allocation = None

    for _ in range(num_lambda_steps):
        lam = (low + high) / 2.0
        ranks = []
        total_params = 0
        for i in range(len(weight_matrices)):
            obj = cost_table[i] + lam * param_table[i]
            best_r_idx = int(np.argmin(obj))
            r = best_r_idx + 1
            ranks.append(r)
            total_params += param_table[i][best_r_idx]

        if total_params <= target_budget:
            best_allocation = ranks.copy()
            high = lam
        else:
            low = lam

    if best_allocation is None:
        best_allocation = [1] * len(weight_matrices)
        log.warning("Could not meet budget, falling back to rank=1 everywhere.")

    rank_dict = {name: r for name, r in zip(layer_proj_names, best_allocation)}
    return rank_dict


# =============================================================================
# 3. SVD‑Compressed Attention module
# =============================================================================


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
        if hasattr(proj, "nf") or proj.__class__.__name__ == "Conv1D":
            w = w.t().contiguous()
    if b is not None:
        b = b.detach()
    return w, b


@ATTENTION_REGISTRY.register("svd_compressed", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_compressed_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_compressed_module", supports_causal=True, supports_gqa=True)
class SVDCompressedAttention(BaseAttention):
    r"""Multi-head attention with truncated SVD projections, trainable LoRA adapters,
    GQA support, RoPE support, and optional sparse residuals.
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
            self.is_decoder = (
                bool(_cfg_get(model_config, "is_decoder", False))
                or any(_mt.startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "falcon", "bloom", "opt"))
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
            if isinstance(mod, nn.Linear):
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
            self.q_sparse_weight = nn.Parameter(torch.zeros(self.q_head_size, self.hidden_size))
            self.k_sparse_weight = nn.Parameter(torch.zeros(self.kv_head_size, self.hidden_size))
            self.v_sparse_weight = nn.Parameter(torch.zeros(self.kv_head_size, self.hidden_size))
            self.o_sparse_weight = nn.Parameter(torch.zeros(self.hidden_size, self.q_head_size))
        else:
            self.register_parameter("q_sparse_weight", None)
            self.register_parameter("k_sparse_weight", None)
            self.register_parameter("v_sparse_weight", None)
            self.register_parameter("o_sparse_weight", None)

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
        src_p = next(src_attn.parameters(), None) if src_attn is not None else None
        target_dtype = kwargs.get("dtype", src_p.dtype if src_p is not None else torch.float32)
        target_device = kwargs.get("device", src_p.device if src_p is not None else None)

        if src_attn is not None:
            self.copy_weights(src_attn)

        if target_device is not None and target_dtype is not None:
            self.to(device=target_device, dtype=target_dtype)
        elif target_device is not None:
            self.to(device=target_device)
        elif target_dtype is not None:
            self.to(target_dtype)

    def _set_factorised_weights(
        self, W_tensor: torch.Tensor, bias_tensor: Optional[torch.Tensor], down: nn.Linear, up: nn.Linear, rank: int
    ) -> None:
        with torch.no_grad():
            W = W_tensor.float()
            d_out, d_in = W.shape
            r = min(rank, d_out, d_in, down.out_features, up.in_features)
            r = max(r, 1)

            U, s, VT = torch.linalg.svd(W, full_matrices=False)
            sqrt_s = torch.sqrt(torch.clamp(s[:r], min=1e-12))
            down_weight = (sqrt_s.unsqueeze(1) * VT[:r, :]).to(device=down.weight.device, dtype=down.weight.dtype)
            up_weight = (U[:, :r] * sqrt_s.unsqueeze(0)).to(device=up.weight.device, dtype=up.weight.dtype)

            down.weight.data.zero_()
            up.weight.data.zero_()
            down.weight.data[:r, :].copy_(down_weight)
            up.weight.data[:, :r].copy_(up_weight)
            if bias_tensor is not None and up.bias is not None:
                up.bias.data.copy_(bias_tensor.to(device=up.bias.device, dtype=up.bias.dtype))

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
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
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
                    k_dim = self.kv_head_size
                    v_dim = self.kv_head_size

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
                log.warning("Failed to extract required projections: %s", missing)
                return

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    log.warning(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Orient weights if needed
            def _orient(w: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
                if w.shape == (out_dim, in_dim):
                    return w
                if w.shape == (in_dim, out_dim):
                    return w.t().contiguous()
                return w

            w_q = _orient(w_q, self.q_head_size, self.hidden_size)
            # Factorize and set Q
            self._set_factorised_weights(w_q, b_q, self.q_down, self.q_up, self.ranks["q"])

            # Factorize and set K & V with GQA pooling if needed
            if w_k.shape[0] == self.kv_head_size:
                w_k = _orient(w_k, self.kv_head_size, self.hidden_size)
                self._set_factorised_weights(w_k, b_k, self.k_down, self.k_up, self.ranks["k"])
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
                self._set_factorised_weights(w_k_pooled, b_k_pooled, self.k_down, self.k_up, self.ranks["k"])

            if w_v.shape[0] == self.kv_head_size:
                w_v = _orient(w_v, self.kv_head_size, self.hidden_size)
                self._set_factorised_weights(w_v, b_v, self.v_down, self.v_up, self.ranks["v"])
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
                self._set_factorised_weights(w_v_pooled, b_v_pooled, self.v_down, self.v_up, self.ranks["v"])

            # Set Out projection if not nested encoder
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    log.warning("Copied out_proj weight has near-zero Frobenius norm (< 1e-4)")
                w_o = _orient(w_o, self.hidden_size, self.q_head_size)
                self._set_factorised_weights(w_o, b_o, self.o_down, self.o_up, self.ranks["o"])

            # Initialize sparse residuals if enabled
            if self.add_sparse:
                low_rank_q = self.q_up.weight @ self.q_down.weight
                res_q = w_q - low_rank_q
                self.q_sparse_weight.data.copy_(res_q)

                low_rank_k = self.k_up.weight @ self.k_down.weight
                res_k = (w_k if w_k.shape[0] == self.kv_head_size else w_k_pooled) - low_rank_k
                self.k_sparse_weight.data.copy_(res_k)

                low_rank_v = self.v_up.weight @ self.v_down.weight
                res_v = (w_v if w_v.shape[0] == self.kv_head_size else w_v_pooled) - low_rank_v
                self.v_sparse_weight.data.copy_(res_v)

                if w_o is not None and not self.is_nested_encoder:
                    low_rank_o = self.o_up.weight @ self.o_down.weight
                    res_o = w_o - low_rank_o
                    self.o_sparse_weight.data.copy_(res_o)

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
        # Defensive device alignment
        if self.q_down.weight.device != hidden_states.device:
            self.to(device=hidden_states.device)

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        def proj_q(x: torch.Tensor) -> torch.Tensor:
            if x.device != self.q_down.weight.device:
                x = x.to(self.q_down.weight.device)
            out = self.q_up(self.q_down(x))
            if self.lora_rank > 0 and self.q_lora_A is not None and self.q_lora_B is not None:
                out = out + F.linear(F.linear(x, self.q_lora_A), self.q_lora_B) * self.scaling
            if self.add_sparse and self.q_sparse_weight is not None:
                out = out + F.linear(x, self.q_sparse_weight)
            return out

        def proj_k(x: torch.Tensor) -> torch.Tensor:
            if x.device != self.k_down.weight.device:
                x = x.to(self.k_down.weight.device)
            out = self.k_up(self.k_down(x))
            if self.lora_rank > 0 and self.k_lora_A is not None and self.k_lora_B is not None:
                out = out + F.linear(F.linear(x, self.k_lora_A), self.k_lora_B) * self.scaling
            if self.add_sparse and self.k_sparse_weight is not None:
                out = out + F.linear(x, self.k_sparse_weight)
            return out

        def proj_v(x: torch.Tensor) -> torch.Tensor:
            if x.device != self.v_down.weight.device:
                x = x.to(self.v_down.weight.device)
            out = self.v_up(self.v_down(x))
            if self.lora_rank > 0 and self.v_lora_A is not None and self.v_lora_B is not None:
                out = out + F.linear(F.linear(x, self.v_lora_A), self.v_lora_B) * self.scaling
            if self.add_sparse and self.v_sparse_weight is not None:
                out = out + F.linear(x, self.v_sparse_weight)
            return out

        def proj_o(x: torch.Tensor) -> torch.Tensor:
            if x.device != self.o_down.weight.device:
                x = x.to(self.o_down.weight.device)
            out = self.o_up(self.o_down(x))
            if self.lora_rank > 0 and self.o_lora_A is not None and self.o_lora_B is not None:
                out = out + F.linear(F.linear(x, self.o_lora_A), self.o_lora_B) * self.scaling
            if self.add_sparse and self.o_sparse_weight is not None:
                out = out + F.linear(x, self.o_sparse_weight)
            return out

        # 1. Projections
        Q = proj_q(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = proj_k(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V = proj_v(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

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
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2).contiguous()) * scale

        # 5. Masking
        if self.is_decoder and not is_cross and N > 1:
            causal_mask = torch.triu(
                torch.full((N, kv_len), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            scores = apply_attention_mask(scores, eff_mask)

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
            output = proj_o(context)



        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., :hidden_states.shape[-1]]

        return output, (attn_weights if output_attentions else None)



# =============================================================================
# 4. Injection helpers
# =============================================================================


def gather_all_attention_weights(model: Any) -> Tuple[List[np.ndarray], List[str]]:
    r"""Gather attention projection weights and layer names across model layers."""
    weights = []
    names = []
    for layer_idx in range(len(model.distilbert.transformer.layer)):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        attn = distil_layer.attention
        for proj in ["q_lin", "k_lin", "v_lin", "out_lin"]:
            w = getattr(attn, proj).weight.data.cpu().numpy()
            weights.append(w)
            proj_short = proj.replace("_lin", "")
            names.append(f"layer{layer_idx}_{proj_short}")
    return weights, names


def inject_all_layers(
    model: Any = None,
    rank_dict: Optional[Dict[str, int]] = None,
    dropout: float = 0.1,
    add_sparse: bool = False,
    sparsity: float = 0.0,
    config: Any = None,
    **kwargs) -> Any:
    r"""Inject SVDCompressedAttention into all transformer layers of model."""
    if rank_dict is None:
        return inject_svd_compressed(model=model, model_config=config, **kwargs)
    for layer_idx in range(len(model.distilbert.transformer.layer)):
        distil_layer = model.distilbert.transformer.layer[layer_idx]
        old_attn = distil_layer.attention
        ranks = {}
        for proj in ["q", "k", "v", "out"]:
            key = f"layer{layer_idx}_{proj}"
            if proj == "out":
                ranks["o"] = rank_dict[key]
            else:
                ranks[proj] = rank_dict[key]
        new_attn = SVDCompressedAttention(
            model_config=config,
            dropout=dropout,
            layer_idx=layer_idx,
            pretrained_attn=old_attn,
            ranks=ranks,
            add_sparse=add_sparse,
            sparsity_level=sparsity,
        )
        if hasattr(distil_layer, "attention") and hasattr(distil_layer.attention, "self"):
            distil_layer.attention.self = new_attn
        else:
            distil_layer.attention = new_attn


# =============================================================================
# 5. GLUE data loading
# =============================================================================

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


def load_glue_local(task_name: str) -> Any:
    r"""Load GLUE benchmark dataset from local offline storage."""
    from datasets import Dataset, DatasetDict
    from framework.utils.paths import get_glue_dir

    base_path = str(get_glue_dir())
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Dataset not found: {task_path}")
    splits = {}
    for fname in os.listdir(task_path):
        if fname.endswith(".parquet"):
            split = fname.split("-")[0]
            splits[split] = Dataset.from_parquet(os.path.join(task_path, fname))
    if task_name == "mnli":
        for extra in ["validation_matched", "validation_mismatched"]:
            fpath = os.path.join(task_path, f"{extra}.parquet")
            if os.path.exists(fpath):
                splits[extra] = Dataset.from_parquet(fpath)
    return DatasetDict(splits)


def get_num_labels_and_is_regression(task: str) -> Tuple[int, bool]:
    r"""Return class count and regression indicator for GLUE task."""
    if task == "stsb":
        return 1, True
    if task == "cola":
        return 2, False
    mapping = {
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    }
    return mapping[task], False


def preprocess_function(
    examples: Any,
    tokenizer: Any,
    sent1_key: str,
    sent2_key: Optional[str],
    max_length: int,
    is_regression: bool,
) -> Dict[str, Any]:
    r"""Preprocess and tokenize dataset sequence examples."""
    if sent2_key is None:
        t = tokenizer(
            examples[sent1_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    else:
        t = tokenizer(
            examples[sent1_key],
            examples[sent2_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    t["labels"] = (
        [float(l) for l in examples["label"]] if is_regression else examples["label"]
    )
    return t


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    r"""Compute classification evaluation accuracy."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": float((preds == labels).mean().item())}


# =============================================================================
# 6. Main training script
# =============================================================================


def main() -> None:
    r"""Run full training workflow for SVDCompressedAttention on GLUE."""
    parser = argparse.ArgumentParser(
        description="Train DistilBERT with Importance‑Aware SVD‑Compressed Attention"
    )
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./svd_compressed_output")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=0.2,
        help="Target fraction of original attention parameters (e.g. 0.2 = 20%)",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--add_sparse",
        action="store_true",
        help="Enable low‑rank + sparse decomposition (experimental)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.7,
        help="Sparsity level for residual (if --add_sparse)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_datasets = load_glue_local(args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True
    )
    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(
            ex, tokenizer, sent1_key, sent2_key, args.max_len, is_regression
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation", tokenized.get("validation_matched"))
    log.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, local_files_only=True
    )

    all_weights, all_names = gather_all_attention_weights(model)
    original_total_params = sum(w.size for w in all_weights)
    target_budget = int(original_total_params * args.budget_ratio)
    log.info(
        f"Original attention parameters: {original_total_params:,}, "
        f"target budget ({args.budget_ratio*100:.0f}%): {target_budget:,}"
    )

    rank_dict = lagrangian_rank_allocation(
        weight_matrices=all_weights,
        target_budget=target_budget,
        layer_proj_names=all_names,
    )
    avg_rank = float(np.mean(list(rank_dict.values())))
    log.info(f"Optimised ranks across layers: average rank = {avg_rank:.1f}")

    inject_all_layers(
        model,
        rank_dict,
        args.dropout,
        add_sparse=args.add_sparse,
        sparsity=args.sparsity,
        config=config,
    )
    log.info("Injected importance‑aware SVD‑Compressed Attention.")

    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(0.06 * total_steps)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("Starting training with importance‑aware SVD‑Compressed Attention...")
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, args.task))
    log.info("Training completed.")


@ATTENTION_REGISTRY.register("svd_compressed_inject")
@ATTENTION_REGISTRY.register("svd_compressed_attention_inject")
@ATTENTION_REGISTRY.register("inject_svd_compressed")
@ATTENTION_REGISTRY.register("inject_svd_compressed_attention")
def inject_svd_compressed(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
    comp_config: Optional[Any] = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[Any] = None,
    lora_rank: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Union[SVDCompressedAttention, nn.Module]]:
    r"""Bridge helper safely injecting SVDCompressedAttention into a transformer layer or model."""
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
            res = inject_svd_compressed(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    try:
        src_attn = old_attn
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    src_attn = getattr(target, attr)
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # Double-injection guard
        if src_attn is not None:
            if isinstance(src_attn, SVDCompressedAttention):
                return None
            if "SVDCompressed" in src_attn.__class__.__name__:
                return None
            if hasattr(src_attn, "inner") and isinstance(getattr(src_attn, "inner"), SVDCompressedAttention):
                return None
            if hasattr(src_attn, "_wrapped_attention") and isinstance(getattr(src_attn, "_wrapped_attention"), SVDCompressedAttention):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, SVDCompressedAttention):
                    return None
                if "SVDCompressed" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), SVDCompressedAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), SVDCompressedAttention):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src_attn is not None:
            c_name = src_attn.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        h_sz = getattr(cfg, "hidden_size", getattr(cfg, "dim", 768)) if cfg else 768
        rank = _cfg_get(comp_config, "rank", _cfg_get(comp_config, "svd_rank", kwargs.get("rank", h_sz)))
        ranks = kwargs.pop("ranks", None)
        if ranks is None:
            ranks = {"q": rank, "k": rank, "v": rank, "o": rank}

        dropout = _cfg_get(comp_config, "dropout", 0.1)
        lr = lora_rank if lora_rank is not None else _cfg_get(comp_config, "lora_rank", 16)

        new_attn = SVDCompressedAttention(
            config=cfg,
            old_attn=src_attn,
            dropout=dropout,
            layer_idx=layer_idx,
            ranks=ranks,
            lora_rank=lr,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    target.attention = new_attn
                    new_attn.is_nested_encoder = False
                return new_attn
            for attr in ("self_attn", "attn", "self"):
                if hasattr(target, attr):
                    setattr(target, attr, new_attn)
                    return new_attn
            target.attention = new_attn
            new_attn.is_nested_encoder = False
            return new_attn
        return new_attn
    except Exception as e:
        log.warning("inject_svd_compressed failed: %s", e)
        return None
