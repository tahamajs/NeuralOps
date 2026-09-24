# Universal Rotary Position Embedding (UnifiedRoPE): Positional Invariance across Attention & SSM Branches

[![Module Type](https://img.shields.io/badge/Module-Attention%20Mechanism-6366f1.svg)](../../catalog.json)
[![Registry Key](https://img.shields.io/badge/Registry-unified__rope-blue.svg)](../../../Codes/framework/core/registry.py)
[![Status](https://img.shields.io/badge/Status-Verified-10b981.svg)](../../../Codes/framework/attentions/unified_rope.py)

## 1. Executive Abstract & Architectural Motivation

Universal Rotary Position Embedding (`UnifiedRoPE`) provides a unified, hardware-efficient rotational positional encoding framework designed for modern transformer and hybrid architectures (spanning flat MHA in DistilBERT, nested encoder self-attention in BERT/RoBERTa, causal decoders in GPT-2, and Grouped-Query Attention in Qwen/LLaMA). In standard Transformers, Rotary Position Embeddings (RoPE, Su et al., 2021) encode sequence positions by rotating Query and Key vectors in complex 2D planes, capturing relative token distances naturally without requiring additive positional bias matrices.

However, cross-architecture migration often suffers from projection mismatches, heterogeneous key-value head dimensions (GQA), and lack of parameter-efficient adaptation. `UnifiedRoPE` integrates:
1. **Universal Adapter & Projection Extraction**: Robust weight harvesting across `c_attn`, `query/key/value`, `q_proj/k_proj/v_proj`, and `q_lin/k_lin/v_lin`.
2. **LoRA-Augmented Projections (`LoRALinear`)**: Base projection weights remain frozen while low-rank adapter matrices ($A, B$) enable efficient downstream fine-tuning.
3. **Precomputed Rotational Frequency Buffers**: Unified inverse frequency matrices and rotational cache buffers (`cos_cached`, `sin_cached`), supporting odd head dimensions safely and dynamic cache resizing.
4. **Grouped-Query Attention (GQA)**: Native `repeat_kv` inter-group broadcasting when $H_Q \ne H_{KV}$.
5. **Universal Masking & Multi-Dimensional Head Masking**: Compatibility with 2D, 3D, and 4D attention masks as well as 1D to 4D head mask broadcasting.

---

## 2. Mathematical Formulations & Derivations

Let $X \in \mathbb{R}^{B \times L \times D}$ denote the input activation tensor where $B$ is the batch size, $L$ is sequence length, and $D$ is the hidden dimension.

### 2.1 Multi-Head / Grouped-Query Projections
Let $H$ be the number of query attention heads and $H_{kv}$ be the number of key-value heads. The head dimension is $d_{\text{head}} = \lfloor D / H \rfloor$.
The projections are computed via `LoRALinear` layers:
$$Q = \operatorname{LoRALinear}_Q(X) \in \mathbb{R}^{B \times L \times (H \cdot d_{\text{head}})}$$
$$K = \operatorname{LoRALinear}_K(X) \in \mathbb{R}^{B \times L \times (H_{kv} \cdot d_{\text{head}})}$$
$$V = \operatorname{LoRALinear}_V(X) \in \mathbb{R}^{B \times L \times (H_{kv} \cdot d_{\text{head}})}$$

Reshaping into multi-head format yields:
$$Q \in \mathbb{R}^{B \times H \times L \times d_{\text{head}}}, \quad K \in \mathbb{R}^{B \times H_{kv} \times L \times d_{\text{head}}}, \quad V \in \mathbb{R}^{B \times H_{kv} \times L \times d_{\text{head}}}$$

### 2.2 Inverse Frequency Vector Precomputation
Let $d_{\text{rot}} = 2 \cdot \lfloor d_{\text{head}} / 2 \rfloor$ denote the even rotary dimension. The inverse frequency parameters $\theta_i$ for frequency channel $i \in \{0, 1, \dots, d_{\text{rot}}/2 - 1\}$ with frequency base $b$ (default $10,000$) are:
$$\theta_i = b^{-\frac{2i}{d_{\text{rot}}}} \in \mathbb{R}^{d_{\text{rot}}/2}$$

### 2.3 Positional Frequency Matrix Construction
For token sequence position $m \in \{0, 1, \dots, L-1\}$ and frequency channel $i$:
$$\text{freqs}_{m, i} = m \cdot \theta_i \in \mathbb{R}^{L \times (d_{\text{rot}}/2)}$$
Duplicating frequencies across both half-dimensions yields the angle matrix $\Theta \in \mathbb{R}^{L \times d_{\text{rot}}}$:
$$\Theta = [\text{freqs} \,\|\, \text{freqs}]$$
The precomputed cached cosine and sine tables are:
$$C_{\text{cache}} = \cos(\Theta) \in \mathbb{R}^{L \times d_{\text{rot}}}, \quad S_{\text{cache}} = \sin(\Theta) \in \mathbb{R}^{L \times d_{\text{rot}}}$$

### 2.4 Vector Half-Rotation (`_rotate_half`) Operator
For a tensor slice $u \in \mathbb{R}^{d_{\text{rot}}}$ split into equal halves $u_1 = u[:d_{\text{rot}}/2]$ and $u_2 = u[d_{\text{rot}}/2:d_{\text{rot}}]$:
$$\mathcal{R}_{\text{half}}(u) = [-u_2 \,\|\, u_1]$$
If $d_{\text{head}} > d_{\text{rot}}$ (odd head dimensions), the trailing pass-through feature $u_{d_{\text{rot}}:]}$ is concatenated unrotated:
$$\mathcal{R}_{\text{half, full}}(x) = [-x[..., d_{\text{rot}}/2 : d_{\text{rot}}] \,\|\, x[..., :d_{\text{rot}}/2] \,\|\, x[..., d_{\text{rot}}:]]$$

### 2.5 RoPE Embedding Application
For query tensor $Q$ and key tensor $K$, rotary embedding is applied up to $d_{\text{rot}}$:
$$Q_{\text{rot}} = (Q_{..., :d_{\text{rot}}} \odot C) + (\mathcal{R}_{\text{half}}(Q_{..., :d_{\text{rot}}}) \odot S)$$
$$K_{\text{rot}} = (K_{..., :d_{\text{rot}}} \odot C) + (\mathcal{R}_{\text{half}}(K_{..., :d_{\text{rot}}}) \odot S)$$
Passing any remaining dimensions without alteration:
$$Q \leftarrow [Q_{\text{rot}} \,\|\, Q_{..., d_{\text{rot}}:}], \quad K \leftarrow [K_{\text{rot}} \,\|\, K_{..., d_{\text{rot}}:]]$$

### 2.6 Relative Positional Distance Invariant
For token positions $m$ and $n$ along head channel $h$:
$$\langle \operatorname{RoPE}(Q_{m, h}), \operatorname{RoPE}(K_{n, h}) \rangle = Q_{m, h}^\top R_{m-n} K_{n, h}$$
where $R_{m-n}$ is an orthogonal block-diagonal 2D rotation matrix whose inner product depends exclusively on the relative offset $(m - n)$.

### 2.7 Grouped-Query Attention (GQA) Expansion
When $H_{kv} < H$, key and value heads are replicated along head dimension by factor $G = H / H_{kv}$:
$$K \leftarrow \operatorname{repeat\_kv}(K, G) \in \mathbb{R}^{B \times H \times L \times d_{\text{head}}}$$
$$V \leftarrow \operatorname{repeat\_kv}(V, G) \in \mathbb{R}^{B \times H \times L \times d_{\text{head}}}$$

### 2.8 Scaled Dot-Product Attention & Output Synthesis
1. Attention Logits:
   $$S = \frac{Q K^\top}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{B \times H \times L \times L}$$
2. Causal / Decoder Masking (if decoder active and $L > 1$):
   $$M_{\text{causal}}(i, j) = \begin{cases} 0, & i \ge j \\ -\infty, & i < j \end{cases}, \quad S \leftarrow S + M_{\text{causal}}$$
3. Attention Mask Integration:
   $$S \leftarrow \operatorname{apply\_attention\_mask}(S, M_{\text{attn}})$$
4. Normalization and Dropout:
   $$A = \operatorname{Dropout}(\operatorname{softmax}(S, \text{dim}=-1))$$
5. Head Mask Modulation:
   $$A \leftarrow A \odot M_{\text{head}}$$
6. Context Aggregation & Out Projection:
   $$\text{Context} = A V \in \mathbb{R}^{B \times H \times L \times d_{\text{head}}} \xrightarrow{\text{permute \& view}} \mathbb{R}^{B \times L \times (H \cdot d_{\text{head}})}$$
   $$\text{Output} = \operatorname{LoRALinear}_O(\text{Context}) \in \mathbb{R}^{B \times L \times D}$$

---

## 3. Architecture & Data Flow

The `UnifiedRoPE` module coordinates input projection with LoRA adaptation, dynamic RoPE rotary embedding, GQA broadcast, and masked attention scoring:

```
Input Hidden States X [B, L, D]
   │
   ├── LoRALinear Projections:
   │     ├─ Q_proj(X) -> [B, L, H * d_head]    -> View & Transpose -> [B, H, L, d_head]
   │     ├─ K_proj(X) -> [B, L, H_kv * d_head] -> View & Transpose -> [B, H_kv, L, d_head]
   │     └─ V_proj(X) -> [B, L, H_kv * d_head] -> View & Transpose -> [B, H_kv, L, d_head]
   │
   ├── Rotary Position Embedding (RoPE):
   │     ├─ Check Cache: if L > max_seq_len, dynamic resize _build_cache(L)
   │     ├─ cos, sin = cos_cached[:L, :d_rot], sin_cached[:L, :d_rot]
   │     ├─ Q_rot = (Q * cos) + (_rotate_half(Q) * sin)
   │     └─ K_rot = (K * cos) + (_rotate_half(K) * sin)
   │
   ├── GQA Expansion (if H_kv != H):
   │     ├─ K = repeat_kv(K, num_key_value_groups) -> [B, H, L, d_head]
   │     └─ V = repeat_kv(V, num_key_value_groups) -> [B, H, L, d_head]
   │
   ├── Scaled Dot-Product Scoring:
   │     ├─ Scores = (Q @ K.T) / sqrt(d_head)      -> [B, H, L, L]
   │     ├─ Causal Mask (if is_decoder and L > 1)
   │     └─ Attention Mask (apply_attention_mask: 2D, 3D, or 4D)
   │
   ├── Probability & Head Masking:
   │     ├─ Attn_Weights = Dropout(Softmax(Scores, dim=-1))
   │     └─ Head Mask (1D, 2D, 3D, 4D broadcasting)
   │
   ├── Context Aggregation & Synthesis:
   │     ├─ Context = Attn_Weights @ V             -> [B, H, L, d_head]
   │     ├─ Reshape & Transpose                    -> [B, L, H * d_head]
   │     └─ Output = LoRALinear_O(Context)         -> [B, L, D]
   │
   └── Return (Output, Attn_Weights if output_attentions else None)
```

### Tensor Shape Transformation Table

| Execution Stage | Input Tensor Shape | Transformation / Operation | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Input Hidden States** | `[B, L, D]` | Received hidden activations | `[B, L, D]` |
| **Q, K, V Projections** | `[B, L, D]` | `q_proj`, `k_proj`, `v_proj` | `Q: [B, L, H*d]`, `K,V: [B, L, H_kv*d]` |
| **Head Partitioning** | `Q, K, V` | `.view().transpose(1, 2)` | `Q: [B, H, L, d]`, `K,V: [B, H_kv, L, d]` |
| **Rotary Embedding** | `Q, K`, `cos`, `sin` | `(x * cos) + (_rotate_half(x) * sin)` | `Q: [B, H, L, d]`, `K: [B, H_kv, L, d]` |
| **GQA KV Broadcast** | `K, V` | `repeat_kv(..., H // H_kv)` | `K, V: [B, H, L, d]` |
| **Logit Matmul** | `Q, K` | `(Q @ K.T) / sqrt(d_head)` | `Scores: [B, H, L, L]` |
| **Causal & Masking** | `Scores`, `mask` | Causal upper-tri + `apply_attention_mask` | `[B, H, L, L]` |
| **Softmax & Dropout** | `Scores` | `F.softmax + nan_to_num + dropout` | `Attn_Weights: [B, H, L, L]` |
| **Head Masking** | `Attn_Weights`, `hm` | Broadcast 1D-4D head mask | `[B, H, L, L]` |
| **Context Aggregation** | `Attn_Weights, V` | `Attn_Weights @ V` | `[B, H, L, d]` |
| **Transpose & Merge** | `Context` | `.transpose(1, 2).contiguous().view()` | `[B, L, H * d]` |
| **Out Projection** | `Context` | `out_proj(Context)` (or identity in nested) | `[B, L, D]` |

---

## 4. Analytical Complexity & Hardware Profiling

### Computational Workload (FLOPs & MACs)
Let $B$ be batch size, $L$ sequence length, $D$ hidden dimension, $H$ number of heads, and $d = D/H$ head dimension.

1. **Precomputed Cache Construction (Once or on sequence extension)**:
   - Frequency inner products: $L \cdot (d_{\text{rot}}/2)$ FLOPs.
   - Trigonometric evaluation ($\cos, \sin$): $2 L \cdot d_{\text{rot}}$ FLOPs.
2. **LoRA Projections**:
   - Frozen base projections: $3 \times 2 B L D^2 = 6 B L D^2$ FLOPs.
   - Trainable LoRA rank $r$: $3 \times 4 B L D r = 12 B L D r$ FLOPs.
3. **Forward RoPE Rotation**:
   - Hadamard multiplications ($Q \odot \cos$, $\mathcal{R}(Q) \odot \sin$, $K \odot \cos$, $\mathcal{R}(K) \odot \sin$): $4 B H L d_{\text{rot}} = 4 B L d_{\text{rot}}$ FLOPs.
   - Elementwise additions: $2 B L d_{\text{rot}}$ FLOPs.
4. **Attention Scoring & Output**:
   - $Q K^\top$: $2 B H L^2 d = 2 B L^2 D$ FLOPs.
   - Softmax & Context $A V$: $2 B L^2 D$ FLOPs.
   - Out projection: $2 B L D^2 + 4 B L D r$ FLOPs.

### Memory Footprint & Parameter Requirements
- **Cache Storage**: $2 \cdot L_{\max} \cdot d_{\text{rot}} \cdot \text{sizeof(dtype)}$ bytes (persistent=False registered buffers).
  For $L_{\max} = 4096, d_{\text{rot}} = 64, \text{FP32}$: $2 \times 4096 \times 64 \times 4 \approx \mathbf{2.1 \text{ MB}}$.
- **LoRA Parameters**: $4 \times (D \cdot r + r \cdot D) = 8 D r$ trainable parameters.

---

## 5. Architectural Bottlenecks, Edge Cases & Failure Modes

1. **Dynamic Sequence Extension Spikes**:
   When runtime sequence length $L > \text{max\_seq\_len}$, `_build_cache(seq_len)` re-allocates and recomputes sinusoidal buffers online on the device, inducing a temporary CUDA memory allocation and synchronization stall.
2. **Odd Head Dimension Handling**:
   Standard RoPE splits feature dimensions evenly ($d/2$). In architectures with odd head dimensions (e.g. $d=65$), naive slicing causes index truncation. `UnifiedRoPE` enforces $d_{\text{rot}} = 2 \cdot \lfloor d/2 \rfloor$ and passes unrotated residual dimensions untouched.
3. **Nested Encoder Container vs Flat MHA**:
   In HuggingFace BERT/RoBERTa architectures, self-attention output projection (`dense`) resides in `BertSelfOutput`, separate from `BertSelfAttention`. `copy_weights` detects nested containers via `self.is_nested_encoder`, bypassing redundant out projection when already managed by the parent module.
4. **Extreme Low-Rank LoRA Truncation**:
   When `lora_rank` is set too low ($r < 4$), adapter expressive capacity may collapse during parameter-efficient fine-tuning on high-entropy reasoning tasks.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design

> [!NOTE]
> **Implementation Status**: *Proposed / Future Work*.
> Production execution utilizes native PyTorch matrix operations and fused rotary embedding. The Triton blueprint below is documented as a hardware acceleration pathway for fused block-level rotary attention kernels.

A proposed Triton kernel fuses Query/Key low-rank scaling, half-rotation index mapping, and Hadamard cosine/sine multiplication into SRAM registers before HBM writeback:

```
                    Global Memory (HBM)
   ┌─────────────────────────────────────────────────────────────┐
   │ Input: Q, K [B, H, L, d_head]                               │
   │ Precomputed Caches: cos_cache [L, d_rot], sin_cache [L, d_rot]│
   └──────────────────────────────┬──────────────────────────────┘
                                  │ Vectorized 2D Block Tile Loading
                                  ▼
 ┌────────────────────────────────────────────────────────────────┐
 │                     Triton SRAM Scratchpad                     │
 │ 1. Load Tile Q_first [BLOCK_M, d_rot/2] & Q_sec [BLOCK_M, d/2] │
 │ 2. Load Positional Caches cos_tile, sin_tile [BLOCK_M, d_rot]  │
 │ 3. Compute Rotated Halves in SRAM Registers:                   │
 │      Q_rot1 = Q_first * cos_first - Q_sec * sin_first          │
 │      Q_rot2 = Q_sec * cos_sec + Q_first * sin_sec              │
 │ └──────────────────────────────┬───────────────────────────────┘
                                  │ Stream Rotated Output to HBM
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Rotated Tensors: Q_rot, K_rot [B, H, L, d_head]             │
   └─────────────────────────────────────────────────────────────┘
```

---

## 7. Empirical Benchmark Results & Comparative Profiling

### Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **88.99%** | -2.07% | 0.8950 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-unified-rope](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-unified-rope) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **64.70%** | -17.10% | 0.6474 | 1.00x | 0 | [svd-linear-attention-distilbert-mnli-full-unified-rope](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-mnli-full-unified-rope) | 2026-09-19 |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **93.00%** | -1.30% | 0.9260 | 1.00x | 0 | [svd-linear-attention-distilbert-agnews-full-unified-rope](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-agnews-full-unified-rope) | 2026-09-19 |


### 7.1 Language Modeling Quality & Throughput
*Pending actual benchmark results.* Benchmark evaluations across GLUE (SST-2, MRPC, MNLI) and LongBench suites are scheduled for execution on standardized GPU clusters.

| Benchmark / Metric | Empirical Value | Baseline (Full MHA) | Relative Delta |
| :--- | :--- | :--- | :--- |
| **WikiText-2 Perplexity ($PPL \downarrow$)** | *Pending actual benchmark results* | 11.20 | — |
| **C4 Perplexity ($PPL \downarrow$)** | *Pending actual benchmark results* | 14.60 | — |
| **Inference Throughput ($tok/sec \uparrow$)** | *Pending actual benchmark results* | 1,250 tok/s | — |
| **Hardware Kernel Status** | Native PyTorch SDPA | Baseline SDPA | — |

### 7.2 Sequence Length Scaling Profiling ($L=512 \to 8192$)
*Pending actual benchmark results.*

| Context Length ($L$) | Inference Latency ($ms$) | Peak VRAM ($MB$) | Speedup vs Dense MHA |
| :--- | :--- | :--- | :--- |
| **$L = 512$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 1024$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 2048$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 4096$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 8192$** | *Pending actual benchmark results* | *Pending* | — |

---

## 8. Complete Source Code Implementation (`unified_rope.py`)

```python
r"""
Universal Rotary Position Embedding Attention (UnifiedRoPE).
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor safely handling odd dims."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if 2 * half < d:
        return torch.cat((rotated, x[..., 2 * half :]), dim=-1)
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
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("compression_unified_rope", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("unified_rope", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("unified_rope_attention", supports_causal=True, supports_gqa=True)
class UnifiedRoPE(BaseAttention):
    r"""Universal Rotary Position Embedding Attention module."""

    supports_gqa: bool = True
    supports_causal: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        base: int = 10000,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, rank=rank)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        if dim is None:
            if config is not None and not isinstance(config, int):
                dim = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", None)))
            if dim is None and src is not None:
                dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
                if dim is None and hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                    dim = src.q_proj.in_features
                elif dim is None and hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                    dim = src.q_lin.in_features
                elif dim is None and hasattr(src, "query") and hasattr(src.query, "in_features"):
                    dim = src.query.in_features
                elif dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                    dim = src.c_attn.weight.shape[0] if src.c_attn.weight.shape[0] != 3 * src.c_attn.weight.shape[1] else src.c_attn.weight.shape[1]
            if dim is None:
                dim = kwargs.get("hidden_size", 768)

        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None)))
            if num_heads is None and src is not None:
                num_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", getattr(src, "n_head", None)))
            if num_heads is None:
                num_heads = 12

        self.hidden_size: int = int(dim)
        self.num_heads: int = max(1, int(num_heads))
        self.head_dim: int = self.hidden_size // self.num_heads
        self.max_seq_len: int = max_seq_len
        self.base: int = base
        self.scale: float = 1.0 / math.sqrt(self.head_dim)
        self.lora_rank: int = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha: float = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        cfg_kv = (
            _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
            if config is not None
            else None
        )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(_cfg_get(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.q_proj = LoRALinear(
            self.hidden_size, self.q_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_proj = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.v_proj = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.hidden_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", dropout))
            if config
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)

        rotary_dim = 2 * (self.head_dim // 2)
        self.rotary_dim = rotary_dim
        if rotary_dim > 0:
            inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        else:
            inv_freq = torch.empty(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _build_cache(self, seq_len: int) -> None:
        """Precomputes cosine and sine frequency tables up to seq_len."""
        if self.rotary_dim == 0 or self.inv_freq.numel() == 0:
            self.register_buffer("cos_cached", torch.ones(seq_len, 0, device=self.inv_freq.device), persistent=False)
            self.register_buffer("sin_cached", torch.zeros(seq_len, 0, device=self.inv_freq.device), persistent=False)
            return
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies RoPE rotation: [B, H, L, D_head]."""
        if self.rotary_dim == 0 or self.cos_cached.shape[-1] == 0:
            return x

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        rotary_dim = min(self.rotary_dim, self.cos_cached.shape[-1], x.shape[-1])
        rotary_dim = 2 * (rotary_dim // 2)
        if rotary_dim == 0:
            return x

        cos = self.cos_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin = self.sin_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)

        if rotary_dim < x.shape[-1]:
            x_rot = x[..., :rotary_dim]
            x_pass = x[..., rotary_dim:]
            rotated = (x_rot * cos) + (_rotate_half(x_rot) * sin)
            return torch.cat((rotated, x_pass), dim=-1)
        else:
            return (x * cos) + (_rotate_half(x) * sin)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None

                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None
                elif w_o is None:
                    for attr in ("out_proj", "out_lin", "dense", "c_proj", "wo", "proj", "output"):
                        if hasattr(src, attr):
                            proj_layer = getattr(src, attr)
                            if hasattr(proj_layer, "dense"):
                                proj_layer = getattr(proj_layer, "dense")
                            if hasattr(proj_layer, "weight"):
                                w_o = proj_layer.weight.detach()
                                b_o = proj_layer.bias.detach() if proj_layer.bias is not None else None
                                break

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None

                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None
                elif w_o is None:
                    for attr in ("out_proj", "o_proj", "dense", "c_proj", "wo", "proj", "output"):
                        if hasattr(src, attr):
                            proj_layer = getattr(src, attr)
                            if hasattr(proj_layer, "dense"):
                                proj_layer = getattr(proj_layer, "dense")
                            if hasattr(proj_layer, "weight"):
                                w_o = proj_layer.weight.detach()
                                b_o = proj_layer.bias.detach() if proj_layer.bias is not None else None
                                break

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None or w_o is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Strict validation
            missing = []
            if w_q is None:
                missing.append("Q projection")
            if w_k is None:
                missing.append("K projection")
            if w_v is None:
                missing.append("V projection")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out projection")

            if missing:
                raise RuntimeError(
                    f"UnifiedRoPE: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            def _copy_layer(target_layer, w, b, src_sub=None):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.t().to(target_layer.weight.dtype))
                    else:
                        min_0 = min(target_layer.weight.shape[0], w.shape[0])
                        min_1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.zero_()
                        target_layer.weight[:min_0, :min_1].copy_(w[:min_0, :min_1].to(target_layer.weight.dtype))
                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.copy_(b.to(target_layer.bias.dtype))
                    else:
                        min_b = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.zero_()
                        target_layer.bias[:min_b].copy_(b[:min_b].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.zero_()

                if src_sub is not None and hasattr(src_sub, "lora_A") and hasattr(src_sub, "lora_B"):
                    if src_sub.lora_A is not None and src_sub.lora_B is not None and target_layer.lora_A is not None and target_layer.lora_B is not None:
                        src_lA = getattr(src_sub.lora_A, "weight", src_sub.lora_A)
                        src_lB = getattr(src_sub.lora_B, "weight", src_sub.lora_B)
                        if hasattr(src_lA, "data") and hasattr(src_lB, "data"):
                            r_min = min(target_layer.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(target_layer.lora_A.shape[1], src_lA.shape[1])
                            target_layer.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            target_layer.lora_A.requires_grad = True

                            out_min = min(target_layer.lora_B.shape[0], src_lB.shape[0])
                            r_min_b = min(target_layer.lora_B.shape[1], src_lB.shape[1])
                            target_layer.lora_B.data[:out_min, :r_min_b].copy_(src_lB.data[:out_min, :r_min_b])
                            target_layer.lora_B.requires_grad = True
                        if hasattr(src_sub, "lora_scaling"):
                            target_layer.scaling = src_sub.lora_scaling

            sub_q = getattr(src, "q_proj", getattr(src, "q_lin", getattr(src, "query", None)))
            sub_k = getattr(src, "k_proj", getattr(src, "k_lin", getattr(src, "key", None)))
            sub_v = getattr(src, "v_proj", getattr(src, "v_lin", getattr(src, "value", None)))
            sub_o = getattr(src, "o_proj", getattr(src, "out_lin", getattr(src, "dense", getattr(src, "out_proj", None))))

            _copy_layer(self.q_proj, w_q, b_q, sub_q)
            _copy_layer(self.k_proj, w_k, b_k, sub_k)
            _copy_layer(self.v_proj, w_v, b_v, sub_v)
            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o, sub_o)

                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    raise RuntimeError(f"UnifiedRoPE: Out projection norm is near zero: {f_norm_o}")

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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """Forward pass computing RoPE attention, returning (output, attention_weights)."""
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            q = self._apply_rope(q, L)
            k = self._apply_rope(k, L)

        # GQA expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

        # Causal mask for decoders
        if self.is_decoder and L > 1:
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)
        if mask is not None:
            scores = apply_attention_mask(scores, mask)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights.to(v.dtype))

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            elif head_mask.dim() == 4:
                head_mask = head_mask
            attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)

        if self.is_nested_encoder:
            if context.shape[-1] != self.hidden_size:
                output = self.out_proj(context)
            else:
                output = context
        else:
            output = self.out_proj(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1])) if output.shape[-1] < hidden_states.shape[-1] else output[..., :hidden_states.shape[-1]]

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("unified_rope_inject")
@ATTENTION_REGISTRY.register("compression_unified_rope_inject")
@ATTENTION_REGISTRY.register("inject_unified_rope")
def inject_unified_rope(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[UnifiedRoPE]:
    """Safe injection hook for UnifiedRoPE."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_attn = UnifiedRoPE(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        layer.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_unified_rope failed at layer %d: %s", layer_idx, e)
        return None
```

---

## 9. Configuration & Hyperparameters

The `UnifiedRoPE` module and its internal `LoRALinear` projection blocks provide the following configurable initialization parameters:

| Class | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `UnifiedRoPE` | `config` | `Optional[Any]` | `None` | Pretrained model configuration namespace or HuggingFace `PretrainedConfig`. |
| `UnifiedRoPE` | `rank` | `Optional[int]` | `None` | Generic adapter / compression rank passed to `BaseAttention`. |
| `UnifiedRoPE` | `dim` | `Optional[int]` | `None` | Total hidden dimension $D$ (inferred from `config.hidden_size` or 768). |
| `UnifiedRoPE` | `num_heads` | `Optional[int]` | `None` | Number of query attention heads $H$ (inferred from `config` or 12). |
| `UnifiedRoPE` | `max_seq_len` | `int` | `4096` | Initial precomputed sequence length for trigonometric cache tables. |
| `UnifiedRoPE` | `base` | `int` | `10000` | Base frequency constant $\theta_0$ for inverse frequency computation. |
| `UnifiedRoPE` | `dropout` | `float` | `0.1` | Attention weight dropout probability applied after softmax. |
| `UnifiedRoPE` | `pretrained_attn` | `Optional[nn.Module]` | `None` | Source attention module to harvest weights from via `copy_weights`. |
| `UnifiedRoPE` | `lora_rank` | `int` | `16` | Bottleneck rank for adapter matrices in projection layers ($0$ disables LoRA). |
| `UnifiedRoPE` | `lora_alpha` | `float` | `32.0` | LoRA scaling numerator; effective multiplier is $\alpha / r$. |
| `UnifiedRoPE` | `is_decoder` | `Optional[bool]` | `None` | Enables causal lower-triangular masking for autoregressive decoders. |
| `LoRALinear` *(Internal)* | `in_features` | `int` | **Required** | Dimension of input feature space $d_{\text{in}}$. |
| `LoRALinear` *(Internal)* | `out_features` | `int` | **Required** | Dimension of output projection space $d_{\text{out}}$. |
| `LoRALinear` *(Internal)* | `bias` | `bool` | `True` | Whether to allocate and learn an additive bias vector. |
| `LoRALinear` *(Internal)* | `freeze_base` | `bool` | `True` | Freezes the base linear parameter weight and bias to preserve pretrained representations. |

---

## 10. Usage & Integration Examples

### HuggingFace Injection & Verification

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Initialize base transformer model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. Inject UnifiedRoPE dynamically into self-attention layers
model = inject_attention(model, attn_type="unified_rope", lora_rank=16)

# 3. Execute forward pass with sequence inputs and attention mask
input_ids = torch.randint(0, 1000, (2, 128))
attention_mask = torch.ones((2, 128), dtype=torch.long)
attention_mask[:, -16:] = 0  # 16 padded tokens

outputs = model(input_ids=input_ids, attention_mask=attention_mask)
print("Output logits shape:", outputs.logits.shape)
```

---

## 11. Attention Maps & Activation Diagnostics

> [!NOTE]
> **Telemetry Roadmap**: Active runtime diagnostics are *planned* for deployment in `results/diagnostics/unified_rope/`.

Planned diagnostic telemetry hooks include:
- **Rotary Phase Spectrum Tracking**: Monitoring relative rotational phase divergence and spectral dispersion across attention layers.
- **LoRA Gradient & Weight Norms**: Logging Frobenius norms of adapter matrices $\|A\|_F, \|B\|_F$ and the frozen base weights to guard against adapter gradient vanishing or explosion.
- **Attention Entropy & Sparsity Profiling**: Spectral entropy computation over softmax distributions to detect head redundancy or position collapse.
