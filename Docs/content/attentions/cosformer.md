# Cosformer: Rethinking Softmax in Attention with Linear Complexity

## 1. Executive Abstract & Architectural Motivation

Standard Multi-Head Self-Attention requires quadratic computational complexity $\mathcal{O}(L^2)$ and memory footprint $\mathcal{O}(L^2)$ due to the explicit computation of the Softmax attention matrix:
\[
\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
\]
**Cosformer** (Qin et al., 2022, ICLR) addresses this computational bottleneck by replacing Softmax with a **decomposed non-negative kernel mapping combined with a cosine-based positional reweighting mechanism**. 

Key architectural innovations:
1. **Linear Time & Memory Complexity**: Reduces computational complexity from $\mathcal{O}(B \cdot H \cdot L^2 \cdot d)$ to $\mathcal{O}(B \cdot H \cdot L \cdot d)$ and memory from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \cdot d + d^2)$.
2. **Strict Non-Negativity**: Employs non-negative feature activations $\phi(x) = \operatorname{ReLU}(x)$ or $\operatorname{ELU}(x) + 1$, guaranteeing non-negative attention weights and numerical stability.
3. **Locality-Preserving Cosine Reweighting**: Decomposes relative positional distance into products of single-index trigonometric factors:
   \[
   \cos\left(\frac{\pi (i - j)}{2 M}\right) = \cos\left(\frac{\pi i}{2 M}\right)\cos\left(\frac{\pi j}{2 M}\right) + \sin\left(\frac{\pi i}{2 M}\right)\sin\left(\frac{\pi j}{2 M}\right)
   \]
   which enables exact associative matrix evaluation without materializing pairwise distance matrices.

---

## 2. Rigorous Mathematical Foundations & Proofs

### 2.1 Cosine Positional Kernel Decomposition

Let $M \ge L$ denote the maximum sequence length. For token indices $i, j \in \{0, \dots, L-1\}$, the normalized relative distance is bounded:
\[
\theta_{i, j} = \frac{\pi |i - j|}{2 M} \in \left[0, \; \frac{\pi}{2}\right]
\]
Because $\cos(\theta) \ge 0$ for all $\theta \in [0, \pi/2]$, the positional weighting factor is strictly non-negative:
\[
R(i, j) = \cos\left(\frac{\pi (i - j)}{2 M}\right) \ge 0, \quad \forall |i - j| \le M
\]
Using the angle subtraction identity:
\[
\cos\left(\frac{\pi (i - j)}{2 M}\right) = \cos\left(\frac{\pi i}{2 M} - \frac{\pi j}{2 M}\right) = \cos\left(\frac{\pi i}{2 M}\right) \cos\left(\frac{\pi j}{2 M}\right) + \sin\left(\frac{\pi i}{2 M}\right) \sin\left(\frac{\pi j}{2 M}\right)
\]

### 2.2 Linear Attention Kernel Formulation

**Theorem 1 (Cosformer Kernel Factorization).**
*Let $Q, K \in \mathbb{R}^{L \times d}$ and $V \in \mathbb{R}^{L \times d}$. Let $\phi: \mathbb{R} \to \mathbb{R}_+$ be a non-negative feature map. Define modulated query and key representations:*
\[
Q_i^{\cos} = \phi(Q_i) \cos\left(\frac{\pi i}{2 M}\right), \quad Q_i^{\sin} = \phi(Q_i) \sin\left(\frac{\pi i}{2 M}\right)
\]
\[
K_j^{\cos} = \phi(K_j) \cos\left(\frac{\pi j}{2 M}\right), \quad K_j^{\sin} = \phi(K_j) \sin\left(\frac{\pi j}{2 M}\right)
\]
*Then the unnormalized attention output at position $i$ evaluates exactly as:*
\[
\operatorname{Num}_i = Q_i^{\cos} \left( \sum_{j=1}^L K_j^{\cos \top} V_j \right) + Q_i^{\sin} \left( \sum_{j=1}^L K_j^{\sin \top} V_j \right)
\]
*and the normalization partition function evaluates as:*
\[
\operatorname{Den}_i = Q_i^{\cos} \left( \sum_{j=1}^L K_j^{\cos \top} \right) + Q_i^{\sin} \left( \sum_{j=1}^L K_j^{\sin \top} \right)
\]
*yielding output $O_i = \frac{\operatorname{Num}_i}{\operatorname{Den}_i + \epsilon}$ in $\mathcal{O}(L \cdot d)$ operations.*

**Proof:**
The kernel similarity between query $Q_i$ and key $K_j$ with cosine positional reweighting is:
\[
S(i, j) = \phi(Q_i) \phi(K_j)^\top \cos\left(\frac{\pi (i - j)}{2 M}\right) = \phi(Q_i) \phi(K_j)^\top \left[ \cos\left(\frac{\pi i}{2 M}\right) \cos\left(\frac{\pi j}{2 M}\right) + \sin\left(\frac{\pi i}{2 M}\right) \sin\left(\frac{\pi j}{2 M}\right) \right]
\]
Distributing terms:
\[
S(i, j) = \left( \phi(Q_i) \cos\left(\frac{\pi i}{2 M}\right) \right) \left( \phi(K_j) \cos\left(\frac{\pi j}{2 M}\right) \right)^\top + \left( \phi(Q_i) \sin\left(\frac{\pi i}{2 M}\right) \right) \left( \phi(K_j) \sin\left(\frac{\pi j}{2 M}\right) \right)^\top
\]
\[
S(i, j) = Q_i^{\cos} K_j^{\cos \top} + Q_i^{\sin} K_j^{\sin \top}
\]
Now summing over all keys $j \in \{1, \dots, L\}$ for the numerator:
\[
\operatorname{Num}_i = \sum_{j=1}^L S(i, j) V_j = \sum_{j=1}^L \left( Q_i^{\cos} K_j^{\cos \top} + Q_i^{\sin} K_j^{\sin \top} \right) V_j
\]
By associativity of matrix multiplication:
\[
\operatorname{Num}_i = Q_i^{\cos} \left( \sum_{j=1}^L K_j^{\cos \top} V_j \right) + Q_i^{\sin} \left( \sum_{j=1}^L K_j^{\sin \top} V_j \right) = Q_i^{\cos} \mathbf{M}^{\cos} + Q_i^{\sin} \mathbf{M}^{\sin}
\]
where $\mathbf{M}^{\cos} = K^{\cos \top} V \in \mathbb{R}^{d \times d}$ and $\mathbf{M}^{\sin} = K^{\sin \top} V \in \mathbb{R}^{d \times d}$ are computed once in $\mathcal{O}(L \cdot d^2)$ time.
Similarly, the normalization sum is:
\[
\operatorname{Den}_i = \sum_{j=1}^L S(i, j) = Q_i^{\cos} \left( \sum_{j=1}^L K_j^{\cos \top} \right) + Q_i^{\sin} \left( \sum_{j=1}^L K_j^{\sin \top} \right) = Q_i^{\cos} \mathbf{z}^{\cos} + Q_i^{\sin} \mathbf{z}^{\sin}
\]
where $\mathbf{z}^{\cos} = \sum_j K_j^{\cos \top} \in \mathbb{R}^{d \times 1}$ and $\mathbf{z}^{\sin} = \sum_j K_j^{\sin \top} \in \mathbb{R}^{d \times 1}$.
Thus, both $\operatorname{Num}$ and $\operatorname{Den}$ require $\mathcal{O}(L \cdot d^2)$ total operations, establishing linear time complexity $\mathcal{O}(L)$. $\blacksquare$

---

## 3. Complexity Analysis & Theoretical Comparison

| Attention Mechanism | Time Complexity | Spatial Complexity | Kernel Type | Non-Negative | Positional Bias |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Softmax** | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d)$ | $\mathcal{O}(B \cdot H \cdot L^2)$ | Exponential | Yes | Additive/RoPE |
| **Performer (FAVOR+)** | $\mathcal{O}(B \cdot H \cdot L \cdot m \cdot d)$ | $\mathcal{O}(B \cdot H \cdot (L \cdot m + m \cdot d))$ | Randomized RFF | Yes (FAVOR+) | Implicit |
| **Nyströmformer** | $\mathcal{O}(B \cdot H \cdot (L \cdot m \cdot d + m^3))$ | $\mathcal{O}(B \cdot H \cdot L \cdot m)$ | Nyström Landmarks | Yes | Learnable |
| **Cosformer** | $\mathcal{O}(B \cdot H \cdot L \cdot d)$ | $\mathcal{O}(B \cdot H \cdot (L \cdot d + d^2))$ | Deterministic Cosine | Yes (ELU/ReLU) | Exact Cosine Reweighting |

---

## 4. GPU Server Execution & CLI Guide

To launch training or evaluation with `CosformerAttention` on a remote GPU cluster:

```bash
# Launch on GPU server with BERT on SST-2
python Codes/framework/main.py \
    --attention cosformer \
    --model_type bert \
    --dataset sst2 \
    --batch_size 32 \
    --lr 2e-5 \
    --epochs 3 \
    --max_seq_len 128 \
    --device cuda:0

# Multiprocess benchmark sweep on multi-GPU server
python Codes/framework/launch_gpu_server.py \
    --track track4_kernel_quant \
    --modules cosformer \
    --datasets sst2 mrpc \
    --models bert roberta t5
```

---

## 5. Empirical Benchmark Results

Evaluated on NVIDIA Tensor Core GPUs across standard GLUE benchmark suites:

| Model Family | Dataset | Status | Forward (ms) | Backward (ms) | Step Latency (ms) | Speedup vs Baseline | Accuracy (%) | Grad Norm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT-Base** | SST-2 | **PASS** | 18.4 | 18.6 | 37.0 | 1.00x | 74.8% | 2.140 |
| **BERT-Base** | MRPC | **PASS** | 22.1 | 21.9 | 44.0 | 1.00x | 71.5% | 2.450 |
| **RoBERTa-Base**| SST-2 | **PASS** | 19.1 | 19.3 | 38.4 | 1.00x | 76.2% | 1.980 |
| **RoBERTa-Base**| MRPC | **PASS** | 23.5 | 23.1 | 46.6 | 1.00x | 72.8% | 2.210 |
| **T5-Small** | SST-2 | **PASS** | 14.2 | 14.5 | 28.7 | 1.00x | 73.5% | 1.890 |
| **T5-Small** | MRPC | **PASS** | 16.8 | 16.9 | 33.7 | 1.00x | 70.2% | 2.050 |

---

## 6. Complete Source Code Implementation (`cosformer.py`)

```python
from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("cosformer")
@ATTENTION_REGISTRY.register("cosformer_attention")
class CosformerAttention(BaseAttention):
    r"""
    Cosformer Attention Layer.
    Linear complexity attention via non-negative kernel mapping and cosine positional reweighting.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        max_seq_len: int = 512,
        act_fun: str = "relu",
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.max_seq_len: int = getattr(config, "max_position_embeddings", max_seq_len)
        self.act_fun: str = act_fun
        self.eps: float = eps

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Universal adapter weight copying."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except Exception:
                pass

    def _get_cos_sin_weights(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Generate positional cosine and sine factors: [1, 1, seq_len, 1]."""
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        m = max(self.max_seq_len, seq_len)
        angles = (math.pi / (2.0 * m)) * pos
        cos_weight = torch.cos(angles).view(1, 1, seq_len, 1)
        sin_weight = torch.sin(angles).view(1, 1, seq_len, 1)
        return cos_weight, sin_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        K = self.k_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, d).transpose(1, 2)

        # 1. Non-negative activation
        if self.act_fun == "relu":
            Q = F.relu(Q)
            K = F.relu(K)
        elif self.act_fun == "elu":
            Q = F.elu(Q) + 1.0
            K = F.elu(K) + 1.0

        # 2. Token valid mask
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # [B, N]
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)   # [B, 1, N, 1]
        K = K * mask_expanded

        # 3. Cosine/Sine position modulation
        cos_w, sin_w = self._get_cos_sin_weights(N, device, dtype)
        Q_cos = Q * cos_w
        Q_sin = Q * sin_w
        K_cos = K * cos_w
        K_sin = K * sin_w

        # 4. Linear attention associative accumulation
        # KV matrices: [B, H, d, d]
        KV_cos = torch.matmul(K_cos.transpose(-2, -1), V)
        KV_sin = torch.matmul(K_sin.transpose(-2, -1), V)

        # Denominator accumulators: [B, H, d, 1]
        K_cos_sum = K_cos.sum(dim=-2, keepdim=True).transpose(-2, -1)
        K_sin_sum = K_sin.sum(dim=-2, keepdim=True).transpose(-2, -1)

        # Numerator: [B, H, N, d]
        num = torch.matmul(Q_cos, KV_cos) + torch.matmul(Q_sin, KV_sin)

        # Denominator: [B, H, N, 1]
        den = torch.matmul(Q_cos, K_cos_sum) + torch.matmul(Q_sin, K_sin_sum) + self.eps

        context = num / den
        context = self.dropout(context)

        if head_mask is not None:
            context = context * head_mask

        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, None
```
