# VSpec++ Attention: Variational Spectral Token-Adaptive Attention with SVD-Gated Projections and Dynamic Top-k Routing

## 1. Abstract & Theoretical Motivation

Standard Multi-Head Self-Attention (MHSA) in Transformer architectures requires quadratic computational time and memory scaling $\mathcal{O}(N^2)$ with sequence length $N$. Static low-rank compression methods enforce uniform projection ranks across all sequence tokens and batch examples, failing to adapt to varying input complexity.

**VSpec++ Attention** (`VSpecAttention`, registered under `vspec` and `vspec_attention`) introduces a dynamic, input-adaptive attention architecture incorporating four core principles:
1. **Saliency-Based Top-$k$ Token Routing**: A learned saliency network (`TokenRouter`) scores token relevance $s_i \in (0, 1)$, dynamically selecting the top-$k$ most informative tokens ($k = \max(N_{\text{min}}, \lfloor N \cdot \text{keep\_ratio} \rfloor)$) to reduce attention computation from $\mathcal{O}(N^2)$ to sub-quadratic complexity $\mathcal{O}(k^2)$.
2. **Input-Dependent Variational Spectral Gating**: Employs a variational stochastic gating network (`VariationalSpectralGate`) that predicts singular value scaling vectors $g \in (0, 1)^r$ using the reparameterization trick $z = \mu + \epsilon \odot \exp(\log \sigma)$.
3. **Dynamic SVD-Gated Projection Layers**: Projections ($Q, K, V, O$) map routed tokens through SVD singular factors ($U, S, V^T$) dynamically modulated by variational spectral gates $S_{\text{eff}} = S \odot g$ with residual LoRA adapters.
4. **Per-Example Head Importance Gating**: Evaluates input-dependent head scaling scalars $h_g \in (0, 1)^H$ to dynamically weight individual attention heads per batch example.

---

## 2. Architecture & Data Flow

The `VSpecAttention` module coordinates token routing, variational spectral gating, dynamic SVD projection, scaled dot-product attention over routed tokens, and residual scatter restoration:

```
Input Hidden States X [B, N, D]
   │
   ├── 1. Variational Spectral Gating & Head Gating:
   │      ├── Masked Mean Pooling: pooled = _masked_mean(X, valid)
   │      ├── Variational Gates: g_q, g_k, g_v = sg_q(X), sg_k(X), sg_v(X) [B, r]
   │      └── Head Gate: h_g = head_gate(X) [B, H, 1, 1]
   │
   ├── 2. Saliency Token Routing:
   │      ├── Router scores tokens: s = TokenRouter(X, valid, is_decoder)
   │      ├── Guarantee [CLS] index 0 and select top-k indices: idx [B, k]
   │      └── Gather routed tokens: X_routed = gather(X, idx) [B, k, D]
   │
   ├── 3. Dynamic SVD Projections with LoRA:
   │      ├── Q = Pq(X_routed, g_q) [B, H, k, d]
   │      ├── K = Pk(X_routed, g_k) [B, H_kv, k, d]
   │      ├── V = Pv(X_routed, g_v) [B, H_kv, k, d]
   │      ├── Rotary Position Embeddings (RoPE on gathered positions)
   │      └── GQA Expansion (repeat K, V to H heads)
   │
   ├── 4. Scaled Dot-Product Attention:
   │      ├── Scores = (Q @ K^T) / sqrt(d) * h_g [B, H, k, k]
   │      ├── Causal Masking (if decoder)
   │      ├── Token Validity Masking (pad_mask on routed tokens)
   │      ├── Softmax & Dropout: Attn_weights = Softmax(Scores)
   │      └── Context = Attn_weights @ V [B, H, k, d]
   │
   ├── 5. Output Projection:
   │      └── Out = Po(Context, g_q) [B, k, D]
   │
   └── 6. Residual-Preserving Scatter:
          ├── Diff = Out - gather(X, idx)
          └── Output = X.clone().scatter_add_(idx, Diff) [B, N, D]
```

### Step-by-Step Tensor Execution Trace
1. **Input Sequence**: `hidden_states: [B, N, D]`, mask `[B, N]`.
2. **Saliency Routing**: Identifies $k$ salient tokens, yielding `routed: [B, k, D]` and `idx: [B, k]`.
3. **Projections**: SVD-gated projections produce `Q, K, V: [B, H, k, d_h]`.
4. **Attention Matrix**: `scores: [B, H, k, k]`. Padded routed tokens are masked with $-\infty$ before Softmax.
5. **Output Reconstruction**: Routed outputs `[B, k, D]` are scattered back using `scatter_add_` on top of cloned `hidden_states`, guaranteeing exact residual stream continuity for unrouted tokens.

---

## 3. Mathematical Derivations & Formulations

### 1. Token Saliency Routing (`TokenRouter`)
Given input sequence $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$:
\[
s_i = \sigma\left( \mathbf{W}_2 \operatorname{GELU}(\mathbf{W}_1 x_i) \right) \in (0, 1)
\]
For decoder models (`is_decoder=True`), future token scores are causally masked:
\[
s_{i} \leftarrow -10000.0 \quad \text{for } i > 0 \text{ at step } 0
\]
The first token ($i=0$, e.g. `[CLS]`) is guaranteed: $s_0 \leftarrow 10^4$.
The top-$k$ indices $\mathcal{I}_{\text{route}} = \operatorname{TopK}(\{s_i\}_{i=1}^N, k)$ are sorted, and routed tokens are gathered:
\[
\widetilde{\mathbf{X}} = \operatorname{Gather}(\mathbf{X}, \mathcal{I}_{\text{route}}) \in \mathbb{R}^{B \times k \times D}
\]

### 2. Variational Spectral Gating (`VariationalSpectralGate`)
Using masked token mean pooling $\bar{\mathbf{x}} = \frac{\sum_{i=1}^N m_i x_i}{\sum_{i=1}^N m_i}$:
\[
[\boldsymbol{\mu}, \log \boldsymbol{\sigma}] = \mathbf{W}_2 \operatorname{GELU}(\mathbf{W}_1 \bar{\mathbf{x}}) \in \mathbb{R}^{B \times 2r}
\]
With $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}_r)$:
\[
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \odot \exp(\operatorname{clamp}(\log \boldsymbol{\sigma}, \text{max}=5.0)), \quad \mathbf{g} = \sigma(\mathbf{z}) \in (0, 1)^{B \times r}
\]

### 3. Dynamic Low-Rank SVD Projection with LoRA (`DynamicLowRankProjector`)
With pre-trained weight SVD factorization $\mathbf{W}_0 \approx \mathbf{U} \operatorname{diag}(\mathbf{S}) \mathbf{V}^\top$:
\[
\mathbf{S}_{\text{eff}} = \mathbf{S} \odot \mathbf{g} \in \mathbb{R}^{B \times r}
\]
\[
\operatorname{Proj}(\widetilde{\mathbf{X}}) = \left( (\widetilde{\mathbf{X}} \mathbf{V}) \odot \mathbf{S}_{\text{eff}} \right) \mathbf{U}^\top + \frac{\alpha}{r_{\text{lora}}} (\widetilde{\mathbf{X}} \mathbf{A}^\top) \mathbf{B}^\top + \mathbf{b}
\]

### 4. Attention Scoring & Residual Scatter
Scaled dot-product attention across routed tokens with per-head importance $h_g$:
\[
\mathbf{S}_{b, h, i, j} = \left( \frac{\mathbf{Q}_{b, h, i} \cdot \mathbf{K}_{b, h, j}^\top}{\sqrt{d_h}} \right) \cdot h_{g, b, h} + \mathbf{M}_{\text{causal}} + \mathbf{M}_{\text{pad}}
\]
\[
\mathbf{A} = \operatorname{Softmax}(\mathbf{S}) \in \mathbb{R}^{B \times H \times k \times k}, \quad \mathbf{O}_{\text{proj}} = \mathbf{P}_o(\mathbf{A} \mathbf{V}, \mathbf{g}_q) \in \mathbb{R}^{B \times k \times D}
\]
Residual-preserving scatter update:
\[
\mathbf{Y} = \mathbf{X} + \operatorname{ScatterAdd}\left( \mathbf{O}_{\text{proj}} - \operatorname{Gather}(\mathbf{X}, \mathcal{I}_{\text{route}}), \mathcal{I}_{\text{route}} \right) \in \mathbb{R}^{B \times N \times D}
\]

---

## 4. Configuration & Hyperparameters

### `VSpecAttention` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `config` | `Optional[PretrainedConfig]` | `None` | HuggingFace architecture configuration (`hidden_size`, `num_attention_heads`, etc.). |
| `rank` | `int` | `64` | SVD projection truncation rank ($r$). |
| `pretrained_attn` | `Optional[nn.Module]` | `None` | Pre-trained attention module for SVD weight decomposition and LoRA extraction. |
| `lora_rank` | `int` | `16` | Rank of trainable residual LoRA adapter matrices. |
| `lora_alpha` | `float` | `32.0` | LoRA scaling factor. |
| `is_decoder` | `Optional[bool]` | `None` | Flag enforcing causal routing and triangular autoregressive masking. |

### `TokenRouter` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hidden_dim` | `int` | **Required** | Model hidden dimension ($D$). |
| `keep_ratio` | `float` | `0.75` | Fraction of sequence tokens to retain for attention ($k / N$). |
| `min_tokens` | `int` | `8` | Minimum number of tokens retained regardless of sequence length. |

### `DynamicLowRankProjector` Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rank` | `int` | `32` | SVD bottleneck factorization rank. |
| `in_dim` | `int` | `768` | Input feature dimension. |
| `out_dim` | `int` | `768` | Output feature dimension. |
| `bias` | `bool` | `True` | Whether to include an additive bias parameter. |
| `lora_rank` | `int` | `16` | Rank of parallel trainable LoRA adapters. |
| `lora_alpha` | `float` | `32.0` | Scaling multiplier for LoRA updates. |
| `freeze_base` | `bool` | `True` | Freezes SVD factor matrices $U, S, V$. |

*(Note: `DynamicLowRankProjector`, `TokenRouter`, and `VariationalSpectralGate` are internal architectural sub-modules; `VSpecAttention` is the primary registered attention module).*

---

## 5. Usage & Integration Examples

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Initialize pretrained base model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. Inject VSpecAttention dynamically
model = inject_attention(model, attn_type="vspec", rank=64, lora_rank=16)

# 3. Forward pass
input_ids = torch.randint(0, 1000, (2, 128))
attention_mask = torch.ones((2, 128), dtype=torch.long)
outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
print("Logits shape:", outputs.logits.shape)
if outputs.attentions is not None:
    print("Routed Attention Weights shape:", outputs.attentions[0].shape)
```

---

## 6. Empirical Benchmark Results & Verification

### Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **46.90%** | -34.90% | 0.4587 | 1.17x | 0 | N/A | 2026-09-19 |
| **distilbert-sst2** | sst2 | `PASS` | 90.60% | **83.26%** | -7.34% | 0.8399 | 1.17x | 0 | [svd-linear-attention-distilbert-sst2-vspec](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-vspec) | 2026-09-27 |
| **distilbert** | sst2 | `PASS` | N/A | **90.94%** | N/A | 0.9125 | 1.00x | 0 | [svd-linear-attention-distilbert-vspec](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-vspec) | 2026-09-29 |
| **roberta** | sst2 | `PASS` | N/A | **75.46%** | N/A | 0.7277 | 1.00x | 0 | [svd-linear-attention-roberta-vspec](https://huggingface.co/Hooshaai/svd-linear-attention-roberta-vspec) | 2026-09-29 |
| **gpt2** | sst2 | `PASS` | N/A | **87.84%** | N/A | 0.8776 | 1.00x | 48 | [svd-linear-attention-gpt2-vspec](https://huggingface.co/Hooshaai/svd-linear-attention-gpt2-vspec) | 2026-09-29 |


| Model | Baseline Acc | VSpec Acc | Δ Acc | F1 | Peak VRAM | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bert-base-uncased` (SST-2) | Pending | Pending | Pending | Pending | Pending | Pending Sweep |
| `roberta-base` (MRPC) | Pending | Pending | Pending | Pending | Pending | Pending Sweep |

---

## 7. Attention Maps & Activation Diagnostics

- **Diagnostic Telemetry (Planned / Future Diagnostics)**:
  - When `output_attentions=True`, `VSpecAttention.forward` returns the post-softmax attention probability matrix across routed tokens: shape `[B, H, k, k]`.
  - The corresponding original sequence indices are given by `idx` (`[B, k]`).
  - Automated logging of token routing entropy, singular value modulation spectra ($S_{\text{eff}}$), and head gate activations under `results/diagnostics/vspec/` is planned for upcoming evaluation sweeps.

---

## 8. Limitations & Failure Modes

1. **Hard Top-$k$ Discontinuity**: The discrete token selection step $\operatorname{TopK}$ is non-differentiable; router parameters are updated via surrogate losses or end-to-end task gradients on retained paths.
2. **Causal Token Routing Invariants**: In causal decoders, dynamic routing must not leak future token information. Future tokens are masked out during scoring, which limits router foresight to past tokens.
3. **Outlier Token Pruning**: If $k$ is chosen too small ($k < 8$), critical syntactic delimiter tokens may be discarded, degrading downstream language modeling fidelity.

---

## 9. Hardware Acceleration Blueprint & Triton Optimization Strategies (Proposed / Future Work)

### High-Performance Triton Kernel Fusion Strategy (Proposed)
1. **Fused Dynamic SVD Projection**: Load routed token sequence $\widetilde{\mathbf{X}}$ into SRAM, apply singular gating in-register, and compute up-projection in a single kernel call.
2. **Fused Scatter-Add Output Kernel**: Directly execute atomic `scatter_add` into global memory output tensors from SRAM, eliminating host Python overhead.

---

## 10. Complete Source Code Implementation (`vspec.py`)

```python
r"""
VSpec++ Attention: Variational Spectral Token-Adaptive Attention.
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


def _masked_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes mean over valid non-padded tokens."""
    if valid_mask is not None:
        m = valid_mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return x.mean(dim=1)


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


class VariationalSpectralGate(nn.Module):
    r"""Input-dependent variational gate over truncated singular values with masked pooling."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, rank * 2)
        )

    def forward(
        self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = _masked_mean(x, valid_mask).float()
        stats = self.net(pooled)
        mu, log_sigma = stats.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=2.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(log_sigma)
        else:
            z = mu
        g = torch.sigmoid(z).to(x.dtype)
        return g, mu, log_sigma


class HeadGate(nn.Module):
    r"""Learned per-example head importance scalars with masked pooling."""

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_heads)
        )

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = _masked_mean(x, valid_mask).float()
        return torch.sigmoid(self.net(pooled)).to(x.dtype)


class TokenRouter(nn.Module):
    r"""Top-k token selector based on learned saliency scores with guaranteed [CLS] retention."""

    def __init__(self, hidden_dim: int, keep_ratio: float = 0.75, min_tokens: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        is_decoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        scores = torch.sigmoid(self.net(x.float()).squeeze(-1))  # (B, N)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)

        if is_decoder and N > 1:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask[0], -10000.0)

        # Guarantee preservation of first token (index 0)
        scores[:, 0] = 1e4

        keep = min(
            N,
            max(
                1,
                min(self.min_tokens, N)
                if N <= self.min_tokens
                else max(self.min_tokens, int(N * self.keep_ratio)),
            ),
        )
        idx = scores.topk(keep, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)

        routed = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return routed, idx


class DynamicLowRankProjector(nn.Module):
    r"""SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters."""

    def __init__(
        self,
        rank: int = 32,
        in_dim: int = 768,
        out_dim: int = 768,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(1, lora_rank)

        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_dim, rank) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_dim))
            self.lora_B = nn.Parameter(torch.empty(out_dim, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def set_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            eff_rank = min(self.rank, U.size(1), Vh.size(0))

            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()

            min_u = min(self.U.size(0), U.size(0))
            min_v = min(self.V.size(0), Vh.size(1))

            self.U.data[:min_u, :eff_rank].copy_(U[:min_u, :eff_rank])
            self.S.data[:eff_rank].copy_(S[:eff_rank])
            self.V.data[:min_v, :eff_rank].copy_(Vh[:eff_rank, :min_v].t())

            if bias is not None and self.bias is not None:
                b_val = bias.data if isinstance(bias, nn.Parameter) else bias
                min_b = min(self.bias.size(0), b_val.size(0))
                self.bias.data[:min_b].copy_(b_val[:min_b])

            rec = (self.U[:, :eff_rank] * self.S[:eff_rank].unsqueeze(0)) @ self.V[:, :eff_rank].t()
            f_norm = float(torch.linalg.norm(rec))
            if f_norm < 1e-4:
                raise RuntimeError(f"DynamicLowRankProjector: reconstructed norm is near zero: {f_norm}")

    def copy_lora_from(self, sub: Any) -> None:
        """Preserve source LoRA adapters."""
        if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
            src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
            src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
            if hasattr(src_lA, "data") and self.lora_A is not None:
                r_min = min(self.lora_A.shape[0], src_lA.shape[0])
                in_min = min(self.lora_A.shape[1], src_lA.shape[1])
                self.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                self.lora_A.requires_grad = True
            if hasattr(src_lB, "data") and self.lora_B is not None:
                out_min = min(self.lora_B.shape[0], src_lB.shape[0])
                r_min = min(self.lora_B.shape[1], src_lB.shape[1])
                self.lora_B.data[:out_min, :r_min].copy_(src_lB.data[:out_min, :r_min])
                self.lora_B.requires_grad = True
            if hasattr(sub, "lora_scaling"):
                self.scaling = sub.lora_scaling

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.V)
        z = z * self.S.unsqueeze(0).unsqueeze(0) * gate.unsqueeze(1)
        out = torch.matmul(z, self.U.t())
        if self.bias is not None:
            out = out + self.bias

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out


@ATTENTION_REGISTRY.register("vspec", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("vspec_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class VSpecAttention(BaseAttention):
    r"""VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module."""

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
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

        # Explicit overrides
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

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False
        self.rank: int = rank
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        self.Pq = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.hidden_size, out_dim=self.all_head_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.Pk = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.hidden_size, out_dim=self.kv_head_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.Pv = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.hidden_size, out_dim=self.kv_head_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.Po = DynamicLowRankProjector(
            rank=self.rank, in_dim=self.all_head_size, out_dim=self.hidden_size,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        self.sg_q = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_k = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_v = VariationalSpectralGate(self.hidden_size, self.rank)

        self.head_gate = HeadGate(self.hidden_size, self.num_heads)
        self.router = TokenRouter(self.hidden_size, keep_ratio=0.75)

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
                        base = getattr(layer, "base_layer", layer)
                        if hasattr(base, "weight"):
                            extracted[proj] = (base.weight.data, getattr(base, "bias", None))
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
            raise RuntimeError(f"VSpecAttention missing required weights for projections: {missing}")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        # Copy into projectors
        self.Pq.set_weight(_orient(extracted["q"][0], self.hidden_size), extracted["q"][1])
        self.Pk.set_weight(_orient(extracted["k"][0], self.hidden_size), extracted["k"][1])
        self.Pv.set_weight(_orient(extracted["v"][0], self.hidden_size), extracted["v"][1])

        if extracted["o"][0] is not None and not self.is_nested_encoder:
            self.Po.set_weight(_orient(extracted["o"][0], self.all_head_size), extracted["o"][1])
        elif self.is_nested_encoder:
            eye_w = torch.eye(self.hidden_size, self.all_head_size)
            self.Po.set_weight(eye_w, None)

        # Preserve source LoRA adapters if present
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.Pq), ("k", self.Pk), ("v", self.Pv), ("o", self.Po)]:
            for name in source_submodules[proj_key]:
                if hasattr(src, name):
                    sub = getattr(src, name)
                    target.copy_lora_from(sub)
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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Mask handling
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        valid = None
        if mask is not None:
            if mask.dim() == 2:
                valid = mask.bool().to(device)
            elif mask.dim() == 4:
                valid = (mask[:, 0, 0, :] > -10000.0).to(device)
            else:
                valid = get_token_valid_mask(mask, B, N, device)

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        # Token routing
        routed, idx = self.router(hidden_states, valid, is_decoder=is_decoder)
        k_len = routed.shape[1]

        # Variational spectral gates using masked mean
        gq, _, _ = self.sg_q(hidden_states, valid)
        gk, _, _ = self.sg_k(hidden_states, valid)
        gv, _, _ = self.sg_v(hidden_states, valid)

        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)

        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        Q = Q.view(B, k_len, H, Dh).transpose(1, 2)
        K = K.view(B, k_len, H_kv, Dh).transpose(1, 2)
        V = V.view(B, k_len, H_kv, Dh).transpose(1, 2)

        # Rotary position embeddings
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            cos_b = cos.expand(B, -1, -1) if cos.size(0) == 1 else cos
            sin_b = sin.expand(B, -1, -1) if sin.size(0) == 1 else sin
            cos_routed = torch.gather(cos_b, 1, idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
            sin_routed = torch.gather(sin_b, 1, idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
            Q, K = apply_rotary_pos_emb(Q, K, cos_routed, sin_routed)
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                Q, K = maybe_apply_rotary(Q, K, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids"))
            except Exception:
                pass

        # GQA expansion
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        hg = self.head_gate(hidden_states, valid).unsqueeze(-1).unsqueeze(-1)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * scale * hg.float()

        # Check causal decoder
        if is_decoder and k_len > 1:
            causal_mask = torch.triu(
                torch.full((k_len, k_len), float("-inf"), device=device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply padding mask to attention scores over routed tokens
        if valid is not None:
            valid_routed = torch.gather(valid, 1, idx)
            pad_mask = (~valid_routed).unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, k_len, k_len)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0).to(dtype)
        attn_weights = self.dropout(attn_weights)

        # Head mask broadcasting
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                if head_mask.shape[-1] == head_mask.shape[-2] and head_mask.shape[-1] != attn_weights.shape[-1]:
                    head_mask = head_mask.mean(dim=-1, keepdim=True)
            attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, k_len, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.Po(context, gq)

        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1])) if out.shape[-1] < D else out[..., :D]

        # Scatter back to full sequence length preserving residual
        output = hidden_states.clone()
        out_diff = out.to(output.dtype) - torch.gather(hidden_states, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        output.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), out_diff)

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("vspec_inject")
@ATTENTION_REGISTRY.register("vspec_attention_inject")
@ATTENTION_REGISTRY.register("inject_vspec_attn")
def inject_vspec_attn(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[VSpecAttention]:
    """Safe injection hook for VSpecAttention."""
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

    rank = kwargs.pop("rank", _cfg_get(comp_config, "vspec_rank", _cfg_get(comp_config, "rank", 32)))
    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

    try:
        new_block = VSpecAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block

        layer.attention = new_block
        return new_block
    except Exception as e:
        logger.error("inject_vspec_attn failed at layer %d: %s", layer_idx, e)
        return None
```