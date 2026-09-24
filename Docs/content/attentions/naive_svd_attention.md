# Naive SVD Attention: Truncated SVD Baseline

## 1. Title & Overview
`NaiveSVDAttention` is an uncalibrated low-rank matrix approximation baseline for multi-head attention weight matrices. It applies standard Singular Value Decomposition (SVD) directly to attention projections ($W_Q, W_K, W_V, W_O$) truncating at rank $r$ without post-hoc fine-tuning, activation-aware scaling, or LoRA recovery adapters.

- **Registry Aliases:** `naive_svd_inject`, `inject_naive_svd`
- **Purpose:** Serves as the negative baseline to measure the necessity of activation weighting and fine-tuning recovery.

## 2. Mathematical Formulation
Given pretrained weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
$$W = U \Sigma V^T$$
Truncating to rank $r \ll \min(d_{\text{out}}, d_{\text{in}})$:
$$\hat{W} = U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T = A B$$
where $A = U_{:, :r} \Sigma_{:r, :r}^{1/2}$ and $B = \Sigma_{:r, :r}^{1/2} V_{:, :r}^T$.

## 3. Architecture & Data Flow
1. SVD factorization is performed once ahead of inference.
2. Injected linear projections factorize: $x \mapsto (x B^T) A^T$.
3. Standard scaled dot-product attention is computed with reduced projection parameters.

## 4. Configuration & Hyperparameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `rank` | `int` | `64` | Truncation rank for low-rank projection. |
| `energy` | `float` | `1.0` | Target singular value cumulative energy threshold. |
| `lora_rank` | `int` | `0` | Set to 0 to strictly enforce non-adapted baseline. |
| `dropout` | `float` | `0.1` | Dropout probability on attention maps. |

## 5. Complete Source Code Implementation
```python
"""
Naive SVD baseline: plain truncated SVD without calibration, Fisher weighting, or whitening.
Used as an uncalibrated low-rank baseline in comparative evaluations.

Paper comparison reference:
Standard truncated SVD approximation W ~ U[:, :r] @ diag(S[:r]) @ V^T[:r, :].
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.minimal_svd import MinimalSVDAttention

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("naive_svd", supports_causal=True, supports_gqa=True)
class NaiveSVDAttention(MinimalSVDAttention):
    r"""
    Naive truncated SVD baseline attention module.
    Inherits production-hardened multi-architecture routing from MinimalSVDAttention
    with uncalibrated plain truncated SVD initialization and no recovery adapters.
    """
    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        energy: float = 1.0,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: int = 0,  # Zero recovery parameters for naive baseline
        lora_alpha: float = 1.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            model_config=model_config,
            energy=energy,
            dropout=dropout,
            pretrained_attn=pretrained_attn,
            rank=rank,
            lora_rank=0,
            lora_alpha=1.0,
            is_decoder=is_decoder,
            **kwargs,
        )


@ATTENTION_REGISTRY.register("naive_svd_inject")
@ATTENTION_REGISTRY.register("inject_naive_svd")
def inject_naive_svd(
    layer: nn.Module,
    comp_config: Any = None,
    old_attn: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
) -> Optional[nn.Module]:
    """Factory hook to instantiate and inject NaiveSVDAttention into transformer layers."""
    rank = getattr(comp_config, "rank", 64) if comp_config else 64
    new_attn = NaiveSVDAttention(
        model_config=model_config,
        rank=rank,
        energy=1.0,
        dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
    )
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", None))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error("copy_weights failed for naive_svd: %s", e)
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
            new_attn.is_nested_encoder = False
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True
    else:
        new_attn.is_nested_encoder = False
    return new_attn

```

## 6. Usage Examples
```python
from framework.core.attention_utils import inject_attention
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
inject_attention(model, module_name="naive_svd_inject", rank=64)
```

## 7. Benchmark Results
| Model | Dataset | Status | Baseline Acc | Injected Acc | Latency | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DistilBERT | SST-2 | PASS | 91.06% | 51.30% | 1.45 ms | 480 MB |

## 8. Attention Maps & Activations
Demonstrates severe attention drift and entropy collapse without activation-aware calibration.

## 9. Hugging Face Model Link
Not uploaded (Baseline only).

## 10. Limitations & Failure Modes
- Severe degradation on task accuracy without recovery fine-tuning or activation weighting.

## 11. Future Work
- Use ASVD or Fisher-Weighted SVD for production compression.
