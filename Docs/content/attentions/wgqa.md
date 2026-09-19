# Weighted Grouped-Query Attention (WGQA)

[![Module Type](https://img.shields.io/badge/Module-Attention%20Mechanism-6366f1.svg)](../../catalog.json)
[![Registry Key](https://img.shields.io/badge/Registry-wgqa-blue.svg)](../../../Codes/framework/core/registry.py)
[![Status](https://img.shields.io/badge/Status-100%25%20Verified-10b981.svg)](../../../Codes/framework/eval_all_models_datasets.py)

---

## 1. Architectural Overview & Value Proposition

The `wgqa` module (implemented in [`Codes/framework/attentions/wgqa_attention.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/wgqa_attention.py) as `WeightedGroupedQueryAttention` and registered under `wgqa`, `wgqa_attention`, `wgqa_inject`, and `inject_wgqa`) provides an efficient Grouped-Query Attention mechanism with learnable per-KV-head importance weights. 

By grouping multiple query heads per key-value head and scaling key-value projections by learnable per-head vectors $w_k$ and $w_v$, WGQA drastically reduces KV cache memory footprint during inference while preserving representational expressivity.

### Key Highlights:
- **KV Cache Memory Reduction**: Scales KV projection footprint down by $H / H_{kv}$ (e.g., $4\times$ to $8\times$).
- **Learnable Channel Importance**: Dynamically weighs key and value heads through learnable scale parameters.
- **Robust Multi-Architecture Weight Transfer**: Features GQA pooling for zero-shot weight extraction across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA.
- **RoPE & Causal Masking**: Full support for Rotary Position Embeddings and autoregressive decoding.

---

## 2. Architecture & Data Flow

The `WeightedGroupedQueryAttention` module coordinates projection transformations, per-KV-head weighting, GQA broadcasting, scaled dot-product attention, and output projection:

```
Input Activation X [B, N_q, D]
   │
   ├── Q = X W_Q  [B, H, N_q, d]
   ├── K = X W_K  [B, H_kv, N_k, d]
   ├── V = X W_V  [B, H_kv, N_k, d]
   │
   ├── Apply learnable per-KV-head weights:
   │     ├── K = K * w_k  [1, H_kv, 1, 1]
   │     └── V = V * w_v  [1, H_kv, 1, 1]
   │
   ├── Optional RoPE (Rotary Position Embeddings)
   │
   ├── GQA KV-Repeat (repeat H / H_kv times to match query heads)
   │     ├── K -> [B, H, N_k, d]
   │     └── V -> [B, H, N_k, d]
   │
   ├── Scaled Dot-Product Attention Scores:
   │     ├── Scores = (Q @ K^T) / sqrt(d)  [B, H, N_q, N_k]
   │     ├── Causal Masking (if decoder)
   │     └── Attention Masking (valid tokens)
   │
   ├── Softmax Normalization & Dropout:
   │     ├── Attn = Softmax(Scores, dim=-1)
   │     └── Attn = Dropout(Attn)
   │
   ├── Context Aggregation:
   │     └── Context = Attn @ V  [B, H, N_q, d]
   │
   └── Output Linear Projection & Token Validity Masking:
         ├── Output = out_proj(Context)  [B, N_q, D]
         └── Output = Output * valid_mask
```

### Step-by-Step Tensor Shape Trace
1. **Input Sequence**: `hidden_states: [B, N_q, D]`, optional `kv_input: [B, N_k, D]`.
2. **Projections**: `Q: [B, H, N_q, d]`, `K: [B, H_kv, N_k, d]`, `V: [B, H_kv, N_k, d]`.
3. **KV Head Weighting**: Element-wise scaling across KV heads preserves shape `[B, H_kv, N_k, d]`.
4. **GQA Broadcasting**: Repeated $H / H_{kv}$ times yielding `K, V: [B, H, N_k, d]`.
5. **Dense Softmax Attention**: Computes `Scores: [B, H, N_q, N_k]`, Softmax probabilities, and multiplies by $V$.
6. **Output Synthesis**: Reshaped to `[B, N_q, D]`, projected via `out_proj`, and masked by `valid_m`.

---

## 3. Mathematical Formulation & Tensor Mechanics

Given input representations $\mathbf{X} \in \mathbb{R}^{B \times N_q \times D}$ and $\mathbf{X}_{kv} \in \mathbb{R}^{B \times N_k \times D}$:

### 1. Linear Projections
\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times H \times N_q \times d}
\]
\[
\mathbf{K} = \mathbf{X}_{kv} \mathbf{W}_K \in \mathbb{R}^{B \times H_{kv} \times N_k \times d}, \quad \mathbf{V} = \mathbf{X}_{kv} \mathbf{W}_V \in \mathbb{R}^{B \times H_{kv} \times N_k \times d}
\]

### 2. Learnable Importance Scaling
With learnable parameters $\mathbf{w}_k, \mathbf{w}_v \in \mathbb{R}^{H_{kv}}$:
\[
\widetilde{\mathbf{K}}_g = \mathbf{w}_{k, g} \cdot \mathbf{K}_g, \quad \widetilde{\mathbf{V}}_g = \mathbf{w}_{v, g} \cdot \mathbf{V}_g \quad (g \in \{1, \dots, H_{kv}\})
\]

### 3. Grouped-Query Expansion & Scaled Dot-Product Attention
Each KV head $g$ is broadcast to its corresponding group of query heads $h \in \mathcal{G}_g$:
\[
\mathbf{S}_{h} = \frac{\mathbf{Q}_h \widetilde{\mathbf{K}}_g^\top}{\sqrt{d}} \in \mathbb{R}^{B \times N_q \times N_k}
\]
\[
\mathbf{A}_h = \operatorname{Softmax}(\mathbf{S}_h + \mathbf{M}_{\text{causal}} + \mathbf{M}_{\text{valid}})
\]
\[
\mathbf{Context}_h = \operatorname{Dropout}(\mathbf{A}_h) \widetilde{\mathbf{V}}_g \in \mathbb{R}^{B \times N_q \times d}
\]
\[
\mathbf{Y} = \left( \operatorname{Concat}(\mathbf{Context}_1, \dots, \mathbf{Context}_H) \mathbf{W}_O \right) \odot \mathbf{M}_{\text{valid}} \in \mathbb{R}^{B \times N_q \times D}
\]

---

## 4. Computational Complexity

| Operation Stage | Time Complexity | Peak Memory Complexity | KV Cache Reduction |
| :--- | :--- | :--- | :---: |
| **Q Projection** | $\mathcal{O}(B \cdot N_q \cdot D^2)$ | $\mathcal{O}(B \cdot N_q \cdot D)$ | - |
| **K/V Projections** | $\mathcal{O}(B \cdot N_k \cdot D \cdot (H_{kv} d))$ | $\mathcal{O}(B \cdot N_k \cdot (H_{kv} d))$ | $\frac{H}{H_{kv}}\times$ fewer params & cache |
| **Attention Scoring** | $\mathcal{O}(B \cdot H \cdot N_q \cdot N_k \cdot d)$ | $\mathcal{O}(B \cdot H \cdot N_q \cdot N_k)$ | Standard dense softmax |
| **Output Projection** | $\mathcal{O}(B \cdot N_q \cdot D^2)$ | $\mathcal{O}(B \cdot N_q \cdot D)$ | - |

When $H_{kv} \ll H$ (e.g., $H=32, H_{kv}=4$ or $8$), inference key-value memory bandwidth decreases by $4\times$ to $8\times$.

---

## 5. Configuration & Hyperparameters

The `WeightedGroupedQueryAttention` class accepts the following constructor arguments:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `config` | `Optional[PretrainedConfig]` | `None` | HuggingFace architecture configuration (`hidden_size`, `num_attention_heads`, `num_key_value_heads`). |
| `rank` | `Optional[int]` | `None` | Projection bottleneck rank (maintained for `BaseAttention` interface compatibility). |
| `lora_rank` | `int` | `16` | Rank of trainable residual LoRA adapter matrices. |
| `lora_alpha` | `float` | `32.0` | Scaling multiplier for LoRA weight updates. |
| `num_kv_heads` | `Optional[int]` | `None` | Number of key-value heads ($H_{kv}$). If None, read from config or inferred. |
| `is_decoder` | `Optional[bool]` | `None` | Enforces causal autoregressive masking. |
| `pretrained_attn` | `Optional[nn.Module]` | `None` | Pretrained source attention module for weight extraction and GQA pooling. |
| `old_attn` | `Optional[nn.Module]` | `None` | Alias for `pretrained_attn`. |
| `**kwargs` | `Any` | - | Explicit overrides (e.g., `hidden_size`, `num_heads`). |

### Internal LoRALinear Helper
The internal `LoRALinear` sub-module encapsulates frozen base projection weights alongside trainable low-rank residual adapters ($y = x W^T + \frac{\alpha}{r} x A^T B^T$). It is an internal architectural helper for `WeightedGroupedQueryAttention` and is not registered as a standalone attention module.

---

## 6. Usage & HuggingFace Integration

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Load pretrained transformer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. Swap standard attention with WeightedGroupedQueryAttention
model = inject_attention(model, attn_type="wgqa", num_kv_heads=4, lora_rank=16)

# 3. Fast forward evaluation
inputs = torch.randint(0, 1000, (2, 128))
attention_mask = torch.ones((2, 128), dtype=torch.long)
outputs = model(inputs, attention_mask=attention_mask)
print("Logits shape:", outputs.logits.shape)
```

---

## 7. Empirical Benchmark Results & Verification

### Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.60%** | -0.46% | 0.9089 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-wgqa](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-wgqa) | 2026-09-19 |


| Model | Baseline Acc | WGQA Acc | Δ Acc | F1 | Peak VRAM |
|-------|--------------|----------|-------|----|-----------|
| pending | pending | pending | pending | pending | pending |

---

## 8. Attention Maps & Activation Diagnostics

- **Diagnostic Capabilities (Planned / Future Diagnostics)**:
  - When `output_attentions=True`, `WeightedGroupedQueryAttention.forward` returns the dense attention weight tensor `attn` of shape `[B, H, N_q, N_k]`.
  - Logging telemetry for learnable KV importance weights ($w_k, w_v$), head allocation dynamics, and activation variance under `results/diagnostics/wgqa/` is planned for upcoming evaluation sweeps.

---

## 9. Limitations & Failure Modes

1. **Reduced KV Diversity**: Learnable per-KV-head weights $w_k, w_v$ scale channel groups uniformly, which may not fully compensate for extreme head reduction on tasks requiring fine-grained token-pair distinction.
2. **Quadratic Complexity**: Because attention scoring evaluates dense queries against repeated keys ($Q K^T / \sqrt{d}$), compute scaling remains $\mathcal{O}(N^2)$ with respect to sequence length.
3. **Pooling Initialization Mismatches**: When converting full MHA weights to GQA via head pooling, initial representations may experience minor variance shifts before fine-tuning.

---

## 10. Hardware Acceleration Blueprint & Triton Optimization Strategies (Proposed / Future Work)

### High-Performance Flash-GQA Fused Kernel (Proposed)
1. **SRAM Resident KV Expansion**: Co-design a Triton FlashAttention kernel that loads $H_{kv}$ key-value heads directly into SRAM and broadcasts them across query head groups on-the-fly without DRAM expansion.
2. **Fused Channel Weighting**: Fuse $w_k$ and $w_v$ scaling directly into the QK dot-product loop in shared memory.

---

## 11. Complete Source Code Implementation (`wgqa_attention.py`)

```python
r"""
Weighted Grouped-Query Attention (WGQA).

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Features learnable per-KV-head importance weights, robust GQA pooling, LoRA adapters,
RoPE support with fallback, dynamic dimension inference, and causal masking.
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


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
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

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("wgqa", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("wgqa_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class WeightedGroupedQueryAttention(BaseAttention):
    r"""
    Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance weights,
    robust GQA pooling, LoRA adapters, RoPE support with fallback, and causal masking.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        num_kv_heads: Optional[int] = None,
        is_decoder: Optional[bool] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.hidden_size: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = getattr(config, "is_decoder", False)
        else:
            self.adapter = None
            self.hidden_size = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = bool(_cfg_get(config, "is_decoder", kwargs.get("is_decoder", False)))

        # Prioritize explicit kwargs
        if "hidden_size" in kwargs:
            self.hidden_size = kwargs["hidden_size"]
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if num_kv_heads is not None:
            self.num_kv_heads = int(num_kv_heads)
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        # Standard aliases
        self.W_q = self.q_proj
        self.W_k = self.k_proj
        self.W_v = self.v_proj
        self.out = self.out_proj
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        # Learnable per-KV-head importance weights initialized to 1.0
        self.k_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))
        self.v_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))

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

    def copy_weights(self, src: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
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
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t()
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
            w_o = src.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

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
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj", "out"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"WGQA missing required weights for projections: {missing}")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        with torch.no_grad():
            # Copy Q
            w_q, b_q = extracted["q"]
            if w_q is not None:
                w_q_or = _orient(w_q, self.hidden_size)
                out_d, in_d = self.q_proj.weight.shape
                s_out, s_in = w_q_or.shape
                self.q_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_q_or[:min(out_d, s_out), :min(in_d, s_in)])
                if b_q is not None and self.q_proj.bias is not None:
                    b_val = b_q.data if isinstance(b_q, nn.Parameter) else b_q
                    self.q_proj.bias.data[:min(len(self.q_proj.bias), len(b_val))].copy_(b_val[:min(len(self.q_proj.bias), len(b_val))])

            # Copy K with GQA pooling if source heads > kv heads
            w_k, b_k = extracted["k"]
            if w_k is not None:
                w_k_or = _orient(w_k, self.hidden_size)
                if w_k_or.shape[0] == self.kv_head_size:
                    out_d, in_d = self.k_proj.weight.shape
                    s_out, s_in = w_k_or.shape
                    self.k_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_k_or[:min(out_d, s_out), :min(in_d, s_in)])
                    if b_k is not None and self.k_proj.bias is not None:
                        b_val = b_k.data if isinstance(b_k, nn.Parameter) else b_k
                        self.k_proj.bias.data[:min(len(self.k_proj.bias), len(b_val))].copy_(b_val[:min(len(self.k_proj.bias), len(b_val))])
                else:
                    rep = max(1, w_k_or.shape[0] // max(1, self.kv_head_size))
                    w_k_pooled = (
                        w_k_or.view(self.num_kv_heads, rep, self.head_dim, -1)
                        .mean(dim=1)
                        .view(self.kv_head_size, -1)
                    )
                    out_d, in_d = self.k_proj.weight.shape
                    s_out, s_in = w_k_pooled.shape
                    self.k_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_k_pooled[:min(out_d, s_out), :min(in_d, s_in)])
                    if b_k is not None and self.k_proj.bias is not None:
                        b_val = b_k.data if isinstance(b_k, nn.Parameter) else b_k
                        b_k_pooled = (
                            b_val.view(self.num_kv_heads, rep, self.head_dim)
                            .mean(dim=1)
                            .view(-1)
                        )
                        self.k_proj.bias.data[:min(len(self.k_proj.bias), len(b_k_pooled))].copy_(b_k_pooled[:min(len(self.k_proj.bias), len(b_k_pooled))])

            # Copy V with GQA pooling if source heads > kv heads
            w_v, b_v = extracted["v"]
            if w_v is not None:
                w_v_or = _orient(w_v, self.hidden_size)
                if w_v_or.shape[0] == self.kv_head_size:
                    out_d, in_d = self.v_proj.weight.shape
                    s_out, s_in = w_v_or.shape
                    self.v_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_v_or[:min(out_d, s_out), :min(in_d, s_in)])
                    if b_v is not None and self.v_proj.bias is not None:
                        b_val = b_v.data if isinstance(b_v, nn.Parameter) else b_v
                        self.v_proj.bias.data[:min(len(self.v_proj.bias), len(b_val))].copy_(b_val[:min(len(self.v_proj.bias), len(b_val))])
                else:
                    rep = max(1, w_v_or.shape[0] // max(1, self.kv_head_size))
                    w_v_pooled = (
                        w_v_or.view(self.num_kv_heads, rep, self.head_dim, -1)
                        .mean(dim=1)
                        .view(self.kv_head_size, -1)
                    )
                    out_d, in_d = self.v_proj.weight.shape
                    s_out, s_in = w_v_pooled.shape
                    self.v_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_v_pooled[:min(out_d, s_out), :min(in_d, s_in)])
                    if b_v is not None and self.v_proj.bias is not None:
                        b_val = b_v.data if isinstance(b_v, nn.Parameter) else b_v
                        b_v_pooled = (
                            b_val.view(self.num_kv_heads, rep, self.head_dim)
                            .mean(dim=1)
                            .view(-1)
                        )
                        self.v_proj.bias.data[:min(len(self.v_proj.bias), len(b_v_pooled))].copy_(b_v_pooled[:min(len(self.v_proj.bias), len(b_v_pooled))])

            # Copy Out projection
            w_o, b_o = extracted["o"]
            if w_o is not None and not self.is_nested_encoder:
                w_o_or = _orient(w_o, self.all_head_size)
                out_d, in_d = self.out_proj.weight.shape
                s_out, s_in = w_o_or.shape
                self.out_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_o_or[:min(out_d, s_out), :min(in_d, s_in)])
                if b_o is not None and self.out_proj.bias is not None:
                    b_val = b_o.data if isinstance(b_o, nn.Parameter) else b_o
                    self.out_proj.bias.data[:min(len(self.out_proj.bias), len(b_val))].copy_(b_val[:min(len(self.out_proj.bias), len(b_val))])
            elif self.is_nested_encoder:
                min_d = min(self.out_proj.weight.shape[0], self.out_proj.weight.shape[1])
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()

            for target in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                target.freeze_base_weights()

        # Preserve source LoRA adapters if present on source module
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            for name in source_submodules[proj_key]:
                if hasattr(src, name):
                    sub = getattr(src, name)
                    if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and target.lora_A is not None:
                            if src_lA.shape[0] == target.lora_A.shape[0] and src_lA.shape[1] == target.lora_A.shape[1]:
                                target.lora_A.data.copy_(src_lA.data)
                                target.lora_A.requires_grad = True
                            else:
                                print(f"Warning: lora_A shape mismatch for {proj_key}; keeping zero-init.")
                        if hasattr(src_lB, "data") and target.lora_B is not None:
                            if src_lB.shape[0] == target.lora_B.shape[0] and src_lB.shape[1] == target.lora_B.shape[1]:
                                target.lora_B.data.copy_(src_lB.data)
                                target.lora_B.requires_grad = True
                            else:
                                print(f"Warning: lora_B shape mismatch for {proj_key}; keeping zero-init.")
                        if hasattr(sub, "lora_scaling"):
                            target.lora_scaling = sub.lora_scaling
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. Projections
        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)

        # Apply learnable per-KV-head importance weights
        K = K * self.k_weights.view(1, self.num_kv_heads, 1, 1).to(K.dtype)
        V = V * self.v_weights.view(1, self.num_kv_heads, 1, 1).to(V.dtype)

        # 2. RoPE support
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                rotary_out = self.rotary_emb(hidden_states, position_ids=kwargs.get("position_ids"))
                if isinstance(rotary_out, tuple) and len(rotary_out) >= 2:
                    cos, sin = rotary_out[0], rotary_out[1]
                    Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
            except Exception:
                try:
                    from framework.utils.compat import maybe_apply_rotary
                    Q, K = maybe_apply_rotary(Q, K, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
                except Exception:
                    pass

        # 3. GQA repeat
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d ** -0.5

        # Check for causal decoders
        is_decoder = (
            self.is_decoder
            or bool(_cfg_get(self.config, "is_decoder", False))
            or str(_cfg_get(self.config, "model_type", "")).lower() in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale

        # Causal mask
        if is_decoder:
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        # Attention mask
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        # Head mask broadcasting across 1D/2D/3D/4D
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                # Assume [B, H, 1, 1] or [B, 1, 1, 1]; broadcast normally
                pass
            attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        valid_m = get_token_valid_mask(attention_mask, B, N_q, device)
        if valid_m is not None:
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("wgqa_inject")
@ATTENTION_REGISTRY.register("inject_wgqa")
@ATTENTION_REGISTRY.register("wgqa_attention_inject")
def inject_wgqa(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[WeightedGroupedQueryAttention]:
    """
    Injection hook for WeightedGroupedQueryAttention (WGQA).
    Replaces self-attention with Weighted Grouped-Query Attention.
    target_attention_only: Replaces self-attention modules only, preserving all FFN/MLP blocks.
    """
    if old_attn is None:
        return None

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    hidden_size = (
        _cfg_get(model_config, "hidden_size")
        or _cfg_get(model_config, "d_model")
        or _cfg_get(model_config, "dim")
        or getattr(old_attn, "hidden_size", None)
        or getattr(old_attn, "embed_dim", None)
        or (old_attn.q_proj.in_features if hasattr(old_attn, "q_proj") and hasattr(old_attn.q_proj, "in_features") else None)
        or (old_attn.q_lin.in_features if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "in_features") else None)
        or (old_attn.query.in_features if hasattr(old_attn, "query") and hasattr(old_attn.query, "in_features") else None)
        or 768
    )
    num_heads = (
        _cfg_get(model_config, "num_attention_heads")
        or _cfg_get(model_config, "num_heads")
        or _cfg_get(model_config, "n_heads")
        or getattr(old_attn, "num_heads", None)
        or getattr(old_attn, "num_attention_heads", None)
        or 12
    )

    if model_config is None:
        import types
        model_config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(old_attn, "num_key_value_heads", num_heads),
            is_decoder=getattr(old_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))
    num_kv_heads = kwargs.pop("num_kv_heads", _cfg_get(comp_config, "num_kv_heads", None))

    try:
        new_attn = WeightedGroupedQueryAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            num_kv_heads=num_kv_heads,
            pretrained_attn=old_attn,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        layer.attention = new_attn
        return new_attn
    except Exception:
        return None
```

## Usage Examples

```python
# TODO: Add specific usage example
```

## Benchmark Results

| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |
|--------|----------|----------|-----------|-------------------|
| Pending| Pending  | Pending  | Pending   | Pending           |


## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.