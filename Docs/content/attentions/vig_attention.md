# ViG Attention: Bidirectional Row-Column ELU Linear Recurrent Grid Attention

## 1. Executive Abstract & Architectural Motivation

Standard Multi-Head Attention treats sequences as 1D structures, incurring $\mathcal{O}(N^2)$ quadratic complexity over length $N$. Vision Grid Attention (ViG) reformulates 1D token sequences of length $N$ into a 2D spatial grid of dimensions $H \times W$ (where $H \cdot W \ge N$). 

By factorizing attention interactions into separate row-wise and column-wise linear attention passes using non-negative feature maps $\phi(x) = \text{ELU}(x) + 1.0$, ViG computes bidirectional contextual interactions in $\mathcal{O}(N \sqrt{N})$ sequence time. Furthermore, dynamic sigmoidal gating channels adaptively blend row and column attention outputs per head, enabling vision models and 2D spatial sequence processors to achieve global context receptive fields with linear-grid efficiency.

---

## 2. Mathematical Formulations & Derivations

Given input sequence representations $X \in \mathbb{R}^{B \times N \times D}$, padding is applied if $N$ is not a perfect square to form a grid of size $H \times W$, where $H = \lfloor\sqrt{N}\rfloor$ and $W = \lceil N / H \rceil$, total length $N_{\text{pad}} = H \cdot W$.

Linear projections compute multi-head Query, Key, and Value tensors $Q, K, V \in \mathbb{R}^{B \times H_h \times N_{\text{pad}} \times d_k}$ and gating weights:

\[
\text{Gates} = \sigma(\text{Linear}(X)) \in \mathbb{R}^{B \times N_{\text{pad}} \times 2 \times H_h \times 1}
\]

Tensors are reshaped into 2D grid representations $Q, K, V \in \mathbb{R}^{B \times H_h \times H \times W \times d_k}$.

### Non-Negative Kernel Feature Mapping ($\phi$):
Query and Key states use the non-negative ELU kernel transformation:

\[
\phi(x) = \text{ELU}(x) + 1.0 > 0
\]

### Row-Wise Linear Attention Pass:
Reshaping grid dimensions into row batches $\mathbb{R}^{B \times (H_h \cdot W) \times H \times d_k}$:

\[
\phi(K_{\text{row}}) = \text{ELU}(K_{\text{row}}) + 1.0, \quad \phi(Q_{\text{row}}) = \text{ELU}(Q_{\text{row}}) + 1.0
\]
\[
M_{\text{row}} = \phi(K_{\text{row}})^\top V_{\text{row}} \in \mathbb{R}^{B \times (H_h W) \times d_k \times d_k}
\]
\[
O_{\text{row}} = \frac{\phi(Q_{\text{row}}) M_{\text{row}}}{\phi(Q_{\text{row}}) \left( \sum_{i=1}^H \phi(K_{\text{row}})_i \right)^\top + \epsilon} \in \mathbb{R}^{B \times H_h \times H \times W \times d_k}
\]

### Column-Wise Linear Attention Pass:
Reshaping grid dimensions into column batches $\mathbb{R}^{(B \cdot H_h \cdot H) \times W \times d_k}$:

\[
\phi(K_{\text{col}}) = \text{ELU}(K_{\text{col}}) + 1.0, \quad \phi(Q_{\text{col}}) = \text{ELU}(Q_{\text{col}}) + 1.0
\]
\[
M_{\text{col}} = \phi(K_{\text{col}})^\top V_{\text{col}} \in \mathbb{R}^{(B H_h H) \times d_k \times d_k}
\]
\[
O_{\text{col}} = \frac{\phi(Q_{\text{col}}) M_{\text{col}}}{\phi(Q_{\text{col}}) \left( \sum_{j=1}^W \phi(K_{\text{col}})_j \right)^\top + \epsilon} \in \mathbb{R}^{B \times H_h \times H \times W \times d_k}
\]

### Dynamic Gated Grid Fusion:
Row and column context outputs are fused using learned sigmoid gates:

\[
O_{\text{fused}} = g_{\text{row}} \odot O_{\text{row}} + g_{\text{col}} \odot O_{\text{col}} \in \mathbb{R}^{B \times H_h \times H \times W \times d_k}
\]

Flattening back to 1D and unpadding yields output $Y = \text{Linear}(O_{\text{fused}}[:, :N, :]) \in \mathbb{R}^{B \times N \times D}$.

---


## Architecture & Data Flow

The `ViGAttention` module coordinates 2D grid factoring, row and column attention computation (linear bidirectional or causal per-axis softmax), gated fusion, and output synthesis:

```
Input Activation X [B, N, D]
   │
   ├── Grid Pad to N_pad = H * W (H = floor(sqrt(N)), W = ceil(N/H))
   ├── Multi-Head Projections: Q, K, V [B, N_pad, H, d_k] (via LoRALinear with frozen base weights)
   ├── Dynamic Gate Generation: Gates = sigmoid(gate_net(X)) [B, N_pad, 2, H, 1]
   │
   ├── Reshape to 2D Spatial Grid: [B, H_h, H, W, d_k]
   ├── Dual-Axis Spatial Attention Passes:
   │     ├── Row-Wise Pass (Bidirectional ELU Linear or Causal Softmax) ──> Out_row
   │     └── Column-Wise Pass (Bidirectional ELU Linear or Causal Softmax) ──> Out_col
   ├── Dynamic Gated Fusion: Out = g_row * Out_row + g_col * Out_col
   ├── Dropout Regularization: context = dropout(Out)
   ├── Head Masking Broadcasting (1D-4D support)
   ├── Grid Unpadding (truncate to original length N)
   └── Context Projection & Output Synthesis: Y = out_proj(context) [B, N, D]
```

### Step-by-Step Tensor Shape Trace

1. **Input Representation**: Shape `[B, N, D]` where $B$ is batch size, $N$ is token count, $D$ is hidden dimension.
2. **Grid Reshaping & Projections**: Padded to $N_{\text{pad}} = H \cdot W$ tokens, projected to multi-head tensors $Q, K, V$ of shape `[B, H_h, N_{\text{pad}}, d_k]` and reshaped to 2D grid `[B, H_h, H, W, d_k]`.
3. **Axis-Decoupled Attention Evaluation**: Row-wise and column-wise passes compute spatial interactions yielding `[B, N_pad, H_h, d_k]`.
4. **Gated Blending & Synthesis**: Context vectors are fused via input-conditioned sigmoid gates, regularized with dropout, unpadded to length $N$, and mapped through `out_proj` to restore hidden dimension `[B, N, D]`.

## 3. Step-by-Step Tensor Execution Trace

```
Input: hidden_states [B, N, D]
  │
  ├──> Grid Pad to N_pad = H * W (H = floor(sqrt(N)), W = ceil(N/H)) ──> [B, N_pad, D]
  ├──> Q, K, V Projections ──> [B, H_h, N_pad, d_k]
  ├──> Gate Net Projections ──> Sigmoid ──> [B, N_pad, 2, H_h, 1]
  │
  ├──> Reshape to 2D Grid ──> [B, H_h, H, W, d_k]
  │
  ├──> Row Branch (H_h * W parallel rows of length H):
  │      ├── Bidirectional Mode:
  │      │     ├── phi(Q_row), phi(K_row) = ELU + 1.0
  │      │     ├── KV_row = phi(K_row).T @ V_row ──> [B, H_h*W, d_k, d_k]
  │      │     └── Out_row = phi(Q_row) @ KV_row / Denom ──> [B, H_h, H, W, d_k]
  │      └── Causal Decoder Mode (Autoregressive Fallback):
  │            ├── scores_r = (Q_r @ K_r.T) / sqrt(d_k)
  │            ├── Apply row lower-triangular causal mask & padding mask
  │            └── Out_row = softmax(scores_r) @ V_r ──> [B, H_h, H, W, d_k]
  │
  ├──> Column Branch (H_h * H parallel columns of length W):
  │      ├── Bidirectional Mode:
  │      │     ├── phi(Q_col), phi(K_col) = ELU + 1.0
  │      │     ├── KV_col = phi(K_col).T @ V_col ──> [B*H_h*H, d_k, d_k]
  │      │     └── Out_col = phi(Q_col) @ KV_col / Denom ──> [B, H_h, H, W, d_k]
  │      └── Causal Decoder Mode (Autoregressive Fallback):
  │            ├── scores_c = (Q_c @ K_c.T) / sqrt(d_k)
  │            ├── Apply column lower-triangular causal mask & padding mask
  │            └── Out_col = softmax(scores_c) @ V_c ──> [B, H_h, H, W, d_k]
  │
  ├──> Gated Fusion: Out = g_row * Out_row + g_col * Out_col ──> [B, N_pad, H_h, d_k]
  ├──> Dropout Regularization: Out = dropout(Out)
  ├──> Head Mask Broadcasting (supports 1D, 2D, 3D, and 4D head masks)
  ├──> Reshape Grid to 1D ──> [B, N_pad, D]
  ├──> Unpad (truncate to original length N) ──> [B, N, D]
  └──> Out Projection: out_proj(Out) ──> Final Output: [B, N, D]
```

### Detailed Tensor Transformations:

| Step | Operation | Tensor Shape | Description |
| :--- | :--- | :--- | :--- |
| 1 | Input | $[B, N, D]$ | Input sequence |
| 2 | Grid Padding | $[B, N_{\text{pad}}, D]$ | Padded to $N_{\text{pad}} = H \cdot W$ |
| 3 | $W_Q, W_K, W_V$ Projections | $[B, H_h, N_{\text{pad}}, d_k]$ | Multi-head activations (LoRALinear with frozen base) |
| 4 | Reshape to 2D Grid | $[B, H_h, H, W, d_k]$ | 2D spatial grid layout |
| 5 | Row Kernel / Causal Attention | $[B, H_h, H, W, d_k]$ | Bidirectional linear associative product or causal masked softmax |
| 6 | Column Kernel / Causal Attention | $[B, H_h, H, W, d_k]$ | Bidirectional linear associative product or causal masked softmax |
| 7 | Sigmoidal Gated Fusion | $[B, N_{\text{pad}}, H_h, d_k]$ | Dynamic blending $g_{\text{row}} O_{\text{row}} + g_{\text{col}} O_{\text{col}}$ |
| 8 | Dropout Regularization | $[B, N_{\text{pad}}, H_h, d_k]$ | Dropout applied to fused contextual states |
| 9 | Head Mask Broadcasting | $[B, N_{\text{pad}}, H_h, d_k]$ | Multi-dimensional head mask pruning applied |
| 10 | Unpad & Out Projection | $[B, N, D]$ | Truncated to $N$ and projected via $W_O$ |

---

## 4. Analytical Complexity & Hardware Profiling

### FLOP and MAC Breakdown
Let $B$ denote batch size, $N$ input sequence length, $H \times W$ the factorized 2D grid dimensions (such that $H \cdot W = N_{\text{pad}} \approx N$), $D$ the model hidden dimension, $H_h$ the number of attention heads, and $d_k = D / H_h$ head dimension. In bidirectional linear attention mode, Query, Key, and Value linear projections demand $6 B N_{\text{pad}} D^2$ FLOPs, alongside gate generation requiring $4 B N_{\text{pad}} D H_h$ FLOPs. In the spatial factorization stage, the row-wise linear attention pass computes associative outer-product states over $H_h \cdot W$ rows of length $H$, requiring $4 B N_{\text{pad}} D d_k$ FLOPs. Symmetrically, the column-wise linear pass processes $H_h \cdot H$ columns of length $W$, contributing an identical $4 B N_{\text{pad}} D d_k$ FLOPs. Following sigmoidal gated blending and dropout, final linear projection back to hidden dimension $D$ consumes $2 B N D^2$ FLOPs.

Summing these contributions yields the cumulative operational footprint:
$$\text{Total MACs} \approx 4 B N D^2 + 4 B N D d_k$$
$$\text{Total FLOPs} \approx 8 B N D^2 + 8 B N D d_k$$

Crucially, by factorizing full 2D self-attention into orthogonal 1D passes along the grid axes, the asymptotic spatial interaction complexity scales as $\mathcal{O}(N (H + W)) \approx \mathcal{O}(N \sqrt{N})$, achieving sub-quadratic scaling while maintaining global bidirectional spatial context across both grid dimensions. In causal decoder mode, per-axis causal softmax attention scales with local row/column quadratic costs $\mathcal{O}(W \cdot H^2 + H \cdot W^2) = \mathcal{O}(N(H+W)) \approx \mathcal{O}(N \sqrt{N})$ rather than full sequence quadratic $\mathcal{O}(N^2)$.

### Memory Footprint & Tensor Layout
The static model parameter footprint is governed by the multi-head projection layers and gating networks, totaling $4 D^2 + 2 D H_h$ weights (plus optional low-rank LoRA parameters). Transient activation memory is strictly bounded by $\mathcal{O}(B \cdot H_h \cdot H \cdot W \cdot d_k)$, entirely eliminating the quadratic $\mathcal{O}(B \cdot H_h \cdot N^2)$ attention matrix storage required by standard Transformers. This reduces transient peak memory from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$, drastically decreasing cache pressure on modern GPU architectures.

---

## 5. Failure Modes, Rank Collapse & Architectural Bottlenecks

### 5.1 Prime Sequence Grid Padding and Inefficient FLOP Utilization
When the sequence length $N$ is a prime number or has highly asymmetric integer factorizations (such as $N = 1009$, where $H = 31$ and $W = 33 \implies N_{\text{pad}} = 1023$), the required grid padding introduces non-negligible computational slack. These padded virtual tokens participate in row and column linear kernel accumulations, consuming arithmetic FLOPs and memory bandwidth while contributing zero semantic information. In extreme aspect ratios ($H \gg W$ or $W \gg H$), this geometric distortion skews receptive field balance and induces non-uniform gradient propagation across the row and column pathways.

### 5.2 Orthogonal Axis Decoupling and Diagonal Receptive Field Latency
Because ViG factorizes spatial attention into strictly decoupled horizontal and vertical passes, immediate single-layer interactions are restricted to tokens sharing either the same row or column coordinate. Direct cross-diagonal token interactions (i.e., between positions $(i_1, j_1)$ and $(i_2, j_2)$ where $i_1 \neq i_2$ and $j_1 \neq j_2$) cannot occur within a single layer; they require at least two cascaded Transformer layers to propagate information through intermediate row-column intersections. This structural constraint delays diagonal feature synthesis and can limit performance on complex visual reasoning tasks requiring immediate omnidirectional receptive fields.

### 5.3 Kernel Underflow and Gradient Saturation in Positive Feature Maps
The non-negative ELU kernel transformation $\phi(x) = \text{ELU}(x) + 1.0$ guarantees non-negativity for valid kernel inner-product formulation. However, for large negative pre-activations, $\phi(x) = \exp(x) \to 0$, causing key and query feature maps to underflow toward zero. When entire token vectors approach this near-zero regime, the associative accumulator matrix $M_{\text{row}} = \phi(K)^\top V$ suffers from severe numerical rank deficiency and key activation sparsity. Furthermore, the gradient $\frac{\partial \phi(x)}{\partial x} = \exp(x)$ vanishes exponentially in negative regimes, stalling optimization progress and degrading the stability of the loss landscape during early training iterations.

### 5.4 Causal Decoder Alignment & Per-Axis Softmax Behavior
Because standard 2D grid linear attention maps sequences across spatial dimensions without enforcing strict 1D temporal order across both axes simultaneously, autoregressive generation requires careful causal masking. In causal decoder mode (`is_decoder=True`), the module employs per-axis causal scaled dot-product attention along rows and columns with lower-triangular causal masks and padding masks, ensuring strict autoregression along each coordinate grid dimension while maintaining sub-quadratic complexity.

---

## 6. Hardware Acceleration Blueprint & Triton Kernel Design (Proposed / Future Work)

> **Note**: The fused Triton kernel implementation below represents a design blueprint for future hardware optimization and is not yet deployed in the production runtime path.

To maximize hardware throughput and eliminate memory bandwidth bottlenecks, a proposed fused Triton kernel executes ELU feature mapping, row and column associative matrix products, and dynamic sigmoidal gating entirely within on-chip SRAM registers, preventing intermediate grid tensor transpositions from round-tripping to DRAM.

```python
# Triton Fused Grid Row/Column Linear Attention Blueprint (Proposed / Future Work)
import triton
import triton.language as tl

@triton.jit
def _vig_grid_fused_kernel(
    Q_ptr, K_ptr, V_ptr, Gate_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qh_dim, stride_qw, stride_qd,
    H, W, d_k,
    BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_row = tl.program_id(1) # Row index
    
    # SRAM Associative Matrix State [d_k, d_k]
    kv_row = tl.zeros((d_k, d_k), dtype=tl.float32)
    
    # Compute row associative state over grid width W
    for w in range(W):
        k_val = tl.load(K_ptr + pid_bh * stride_qh + pid_row * stride_qh_dim + w * stride_qw + tl.arange(0, d_k))
        v_val = tl.load(V_ptr + pid_bh * stride_qh + pid_row * stride_qh_dim + w * stride_qw + tl.arange(0, d_k))
        
        # ELU + 1.0 Feature Map
        phi_k = tl.where(k_val > 0, k_val + 1.0, tl.exp(k_val))
        kv_row += phi_k[:, None] * v_val[None, :]
        
    # Store row output tile
```

---

## 7. Next-Generation Innovations & Synthesis Opportunities

To advance the capabilities and versatility of bidirectional grid attention architectures, several promising research directions emerge:

**Shifted Grid Window Partitions (Swin-ViG)**: Introducing cyclic partition shifts across successive Transformer blocks dynamically alternates the row and column boundaries, bridging the diagonal receptive field gap within a single stacked block and accelerating cross-spatial feature composition.

**Hybrid Grid-SSM Multidimensional Ensembles**: Integrating 2D ViG grid linear attention with continuous-time bidirectional State-Space Models replaces flattened 1D scan heuristics in Vision Mamba with geometrically grounded 2D continuous state dynamics.

**Dynamic Deformable Grid Topologies**: Parameterizing grid dimensions $H(x)$ and $W(x)$ via input-dependent routing enables the network to adaptively match arbitrary image aspect ratios and irregularly sized inputs, completely eliminating zero-padding waste and equalizing directional receptive fields.

## 8. Empirical Benchmark Results & Comparative Profiling

### Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **88.19%** | -2.87% | 0.8815 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-vig](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-vig) | 2026-09-19 |


> **Notice**: Empirical benchmark evaluation for `vig_attention` across standard sequence lengths and language modeling benchmarks is currently underway. Hardware profiling and latency numbers will be published upon test completion.

### 8.1 Language Modeling Quality & Throughput
| Benchmark / Metric | Empirical Value | Baseline (Full MHA) | Relative Delta |
| :--- | :--- | :--- | :--- |
| **WikiText-2 Perplexity ($PPL \downarrow$)** | *Pending actual benchmark results* | 11.20 | — |
| **C4 Perplexity ($PPL \downarrow$)** | *Pending actual benchmark results* | 14.60 | — |
| **Inference Throughput ($tok/sec \uparrow$)** | *Pending actual benchmark results* | 1,250 tok/s | — |
| **Hardware Kernel Acceleration** | Standard PyTorch SDPA / Linear Associative | PyTorch SDPA | Baseline |

### 8.2 Sequence Length Scaling Profiling ($L=512 \to 8192$)
| Context Length ($L$) | Inference Latency ($ms$) | Peak VRAM ($MB$) | Speedup vs Dense MHA |
| :--- | :--- | :--- | :--- |
| **$L = 512$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 1024$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 2048$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 4096$** | *Pending actual benchmark results* | *Pending* | — |
| **$L = 8192$** | *Pending actual benchmark results* | *Pending* | — |

---

## Complete Source Code Implementation (`vig_attention.py`)

```python
r"""
ViG Attention: Vision GNN-inspired 2D Grid Linear Attention with Gated Fusion.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
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


@ATTENTION_REGISTRY.register("vig", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("vig_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class ViGAttention(BaseAttention):
    r"""Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion."""

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.dim: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = getattr(config, "is_decoder", False)
        else:
            self.adapter = None
            self.dim = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", kwargs.get("d_model", 768)))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = bool(_cfg_get(config, "is_decoder", kwargs.get("is_decoder", False)))

        # Explicit overrides
        if "hidden_size" in kwargs:
            self.dim = kwargs["hidden_size"]
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
            self.head_dim = self.dim // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        # Standard aliases
        self.W_q = self.q_proj
        self.W_k = self.k_proj
        self.W_v = self.v_proj
        self.out = self.out_proj
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads)
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

    def copy_weights(self, old_attn: Any) -> None:
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
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(old_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = old_attn.c_attn.weight.data
            bias = getattr(old_attn.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.dim:
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

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            w_o = old_attn.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.dim and w_o.shape[1] == self.dim:
                w_o = w_o.t()
            b_o = getattr(old_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(old_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(old_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = old_attn.output.dense
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
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        base = getattr(layer, "base_layer", layer)
                        if hasattr(base, "weight"):
                            extracted[proj] = (base.weight.data, getattr(base, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj", "out"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"ViGAttention missing required weights for projections: {missing}")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.dim
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape
                    target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[:min(len(target.bias), len(b_val))].copy_(b_val[:min(len(target.bias), len(b_val))])
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        # Preserve source LoRA adapters
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            for name in source_submodules[proj_key]:
                if hasattr(old_attn, name):
                    sub = getattr(old_attn, name)
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

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)

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
        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        H: int = max(1, int(math.isqrt(N)))
        W: int = (N + H - 1) // H
        total: int = H * W
        pad_len: int = total - N

        if pad_len > 0:
            x = torch.cat(
                [
                    hidden_states,
                    torch.zeros(B, pad_len, D, device=device, dtype=dtype),
                ],
                dim=1,
            )
            N_pad = total
        else:
            x = hidden_states
            N_pad = N

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                valid_2d = (mask[:, 0, 0, :] > -10000.0).to(device)
            elif mask.dim() == 3:
                valid_2d = (mask[:, 0, :] > -10000.0).to(device)
            elif mask.dim() == 2:
                valid_2d = (mask > 0).to(device)
            else:
                valid_2d = mask.bool().to(device)
            if pad_len > 0:
                valid = torch.cat([valid_2d, torch.zeros(B, pad_len, dtype=torch.bool, device=device)], dim=1)
            else:
                valid = valid_2d
        else:
            valid = torch.ones(B, N_pad, dtype=torch.bool, device=device)

        Hh = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)

        # Rotary position embeddings
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            if pad_len > 0:
                cos_pad = torch.cat([cos, torch.zeros(cos.size(0), pad_len, cos.size(-1), device=cos.device, dtype=cos.dtype)], dim=1)
                sin_pad = torch.cat([sin, torch.zeros(sin.size(0), pad_len, sin.size(-1), device=sin.device, dtype=sin.dtype)], dim=1)
            else:
                cos_pad, sin_pad = cos, sin
            q, k = apply_rotary_pos_emb(q, k, cos_pad, sin_pad)
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                q, k = maybe_apply_rotary(q, k, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
            except Exception:
                pass

        # GQA expansion
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        gates = torch.sigmoid(self.gate_net(x)).view(B, N_pad, 2, Hh, 1)

        q = q.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Row Linear Attention
        q_row = q.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        k_row = k.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        v_row = v.reshape(B, Hh * W, H, Dh).transpose(2, 3)

        q_row_lin = F.elu(q_row.float()) + 1.0
        k_row_lin = F.elu(k_row.float()) + 1.0

        valid_grid = valid.reshape(B, H, W).permute(0, 2, 1)
        valid_row = (
            valid_grid.unsqueeze(1)
            .expand(B, Hh, W, H)
            .reshape(B, Hh * W, H)
            .unsqueeze(-2)
            .to(k_row_lin.dtype)
        )
        k_row_lin = k_row_lin * valid_row
        v_row_f = v_row.float() * valid_row

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        if is_decoder:
            # Per-axis causal softmax attention along rows
            # q_row: [B, Hh * W, Dh, H] -> permute to [B, Hh * W, H, Dh]
            q_r = q_row.transpose(-1, -2).float()
            k_r = k_row.transpose(-1, -2).float()
            v_r = v_row.transpose(-1, -2).float()
            scale_r = 1.0 / math.sqrt(Dh)
            scores_r = torch.matmul(q_r, k_r.transpose(-1, -2)) * scale_r
            tril_r = torch.tril(torch.ones(H, H, device=q.device, dtype=torch.bool))
            scores_r = scores_r.masked_fill(~tril_r.unsqueeze(0).unsqueeze(0), float("-inf"))
            valid_r_mask = valid_row.squeeze(-2)  # [B, Hh * W, H]
            scores_r = scores_r.masked_fill(~valid_r_mask.unsqueeze(-2), float("-inf"))
            # If an entire row is masked out, softmax produces NaNs: fill with 0.0 after nan_to_num
            attn_r = torch.softmax(scores_r, dim=-1)
            attn_r = torch.nan_to_num(attn_r, nan=0.0)
            out_row = torch.matmul(attn_r, v_r)
            out_row = out_row.transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)
        else:
            kv_row = torch.einsum("b m n d, b m n e -> b m d e", k_row_lin, v_row_f)
            z_row = k_row_lin.sum(dim=2, keepdim=True)
            denom_row = torch.einsum("b m n d, b m d -> b m n", q_row_lin, z_row.squeeze(2)).unsqueeze(-1)
            out_row = torch.einsum("b m n d, b m d e -> b m n e", q_row_lin, kv_row) / (denom_row.clamp(min=1e-5))
            out_row = out_row.transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)

        # Column Linear Attention
        q_col = q.permute(0, 1, 2, 4, 3)
        k_col = k.permute(0, 1, 2, 4, 3)
        v_col = v.permute(0, 1, 2, 4, 3)
        q_col_flat = q_col.reshape(B * Hh * H, W, Dh)
        k_col_flat = k_col.reshape(B * Hh * H, W, Dh)
        v_col_flat = v_col.reshape(B * Hh * H, W, Dh)

        q_col_lin = F.elu(q_col_flat.float()) + 1.0
        k_col_lin = F.elu(k_col_flat.float()) + 1.0

        valid_col = (
            valid.reshape(B, H, W)
            .unsqueeze(1)
            .expand(B, Hh, H, W)
            .reshape(B * Hh * H, W)
            .unsqueeze(-1)
            .to(k_col_lin.dtype)
        )
        k_col_lin = k_col_lin * valid_col
        v_col_f = v_col_flat.float() * valid_col

        if is_decoder:
            # Per-axis causal softmax attention along columns
            scale_c = 1.0 / math.sqrt(Dh)
            scores_c = torch.matmul(q_col_flat.float(), k_col_flat.float().transpose(-1, -2)) * scale_c
            tril_c = torch.tril(torch.ones(W, W, device=q.device, dtype=torch.bool))
            scores_c = scores_c.masked_fill(~tril_c.unsqueeze(0), float("-inf"))
            valid_c_mask = valid_col.squeeze(-1)  # [B * Hh * H, W]
            scores_c = scores_c.masked_fill(~valid_c_mask.unsqueeze(-2), float("-inf"))
            attn_c = torch.softmax(scores_c, dim=-1)
            attn_c = torch.nan_to_num(attn_c, nan=0.0)
            out_col = torch.matmul(attn_c, v_col_flat.float())
            out_col = out_col.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        else:
            kv_col = torch.einsum("b n d, b n e -> b d e", k_col_lin, v_col_f)
            z_col = k_col_lin.sum(dim=1, keepdim=True)
            denom_col = torch.einsum("b n d, b d -> b n", q_col_lin, z_col.squeeze(1)).unsqueeze(-1)
            out_col = torch.einsum("b n d, b d e -> b n e", q_col_lin, kv_col) / (denom_col.clamp(min=1e-5))
            out_col = out_col.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Gated fusion
        out_row = out_row.permute(0, 1, 3, 2, 4).reshape(B, Hh, N_pad, Dh).permute(0, 2, 1, 3)
        out_col = out_col.permute(0, 1, 3, 2, 4).reshape(B, Hh, N_pad, Dh).permute(0, 2, 1, 3)

        gate_row = gates[:, :, 0, :, :]
        gate_col = gates[:, :, 1, :, :]

        out = (out_row * gate_row + out_col * gate_col).to(dtype)
        out = self.dropout(out)

        # Head mask broadcasting
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, 1, -1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), 1, head_mask.size(1), 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                if head_mask.shape[-1] == head_mask.shape[-2] and head_mask.shape[-1] != out.shape[-1]:
                    head_mask = head_mask.mean(dim=-1, keepdim=True)
            out = out * head_mask.to(dtype)

        out = out.reshape(B, N_pad, self.all_head_size)

        if pad_len > 0:
            out = out[:, :N, :]

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        return (output, None)


@ATTENTION_REGISTRY.register("vig_inject")
@ATTENTION_REGISTRY.register("vig_attention_inject")
@ATTENTION_REGISTRY.register("inject_vig")
def inject_vig(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[ViGAttention]:
    """Safe injection helper for ViG attention."""
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

    try:
        new_attn = ViGAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        layer.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_vig failed at layer %d: %s", layer_idx, e)
        return None
```

## Configuration & Hyperparameters

The `ViGAttention` module (registered under `"vig"` and `"vig_attention"`, with injection helper `inject_vig`) provides configurable parameters for runtime optimization, numerical stability thresholds, and low-rank adapter allocation. Projections are initialized via `LoRALinear`, an internal helper module with frozen base weights and trainable low-rank adapters:

| Class | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `ViGAttention` | `config` | `Any` | `None` | Model configuration instance (provides hidden size, head count, etc.). |
| `ViGAttention` | `rank` | `Optional[int]` | `None` | Low-rank bottleneck projection rank. |
| `ViGAttention` | `pretrained_attn` | `Optional[nn.Module]` | `None` | Source attention module from which base weights and biases are copied. |
| `ViGAttention` | `lora_rank` | `int` | `16` | Rank of the residual LoRA parameter update matrix in projection layers. |
| `ViGAttention` | `lora_alpha` | `float` | `32.0` | Scaling multiplier for LoRA weight updates (`scaling = alpha / rank`). |
| `ViGAttention` | `is_decoder` | `Optional[bool]` | `None` | Explicit flag to enable per-axis causal softmax attention for autoregressive models. |

*Note: `LoRALinear` is an internal helper module implementing low-rank updates with frozen base weights; it is not registered as a standalone attention mechanism.*

## Usage & Integration Examples

### HuggingFace Injection & Verification

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Initialize base model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# 2. Inject ViGAttention dynamically
model = inject_attention(model, attn_type='vig', rank=64)

# 3. Execute forward pass
input_ids = torch.randint(0, 1000, (2, 128))
outputs = model(input_ids)
print('Output logits shape:', outputs.logits.shape)
```


## Attention Maps & Activation Diagnostics

The `ViGAttention` architecture supports diagnostic instrumentation for tracking receptive field distributions and representation health:
- **Activation Diagnostics**: Diagnostic telemetry hooks and receptive pattern logging under `results/diagnostics/vig_attention/` are currently planned for future releases.
- **Representation Monitoring**: Gated fusion channel distributions ($g_{\text{row}}$ vs $g_{\text{col}}$) provide telemetry on horizontal vs vertical feature routing across attention heads.


## Future Work & Algorithmic Extensions

- **Hardware-Fused Triton Kernels**: Co-designing block-level GPU kernel fusions to eliminate intermediate SRAM-to-HBM write cycles for row and column passes.
- **Dynamic Deformable Grid Topologies**: Parameterizing grid dimensions $H(x)$ and $W(x)$ via input-dependent routing to eliminate padding waste.
- **Quantized Low-Rank Representation**: Joint FP4/INT4 quantization of factorized projection matrices with non-uniform codebooks.



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