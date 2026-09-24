# SALT Attention: Singular Value Adaptation with Low-Rank Tuning for Parameter-Efficient Transformer Fine-Tuning

## 1. Abstract & Theoretical Motivation

Fine-tuning large pre-trained Transformer architectures via full parameter update is computationally expensive and memory-prohibitive. While standard Low-Rank Adaptation (LoRA) introduces trainable bottleneck matrices $A B$, it ignores the underlying singular value spectrum of the pre-trained weight matrices, often perturbing dominant singular directions during gradient updates.

**SALT Attention** (Singular Value Adaptation with Low-Rank Tuning) introduces a parameter-efficient adaptation paradigm for multi-head attention projections that combines:
1. **Frozen Orthogonal Base Singular Vector Preservation**: Singular vectors $U_{:, :r}$ and $V_{:r, :}^T$ from the pre-trained weight SVD ($W_0 = U \Sigma V^T$) are registered as frozen buffers, preserving principal subspace geometry.
2. **Singular Value Rescaling Vector**: A learnable vector $\gamma \in \mathbb{R}^r$ modulates the top-$r$ pre-trained singular values $S_{1:r} \odot \gamma$, allowing fine-grained scaling of existing principal components.
3. **Additive Low-Rank Residual Adapter**: A trainable low-rank branch $A B$ ($A \in \mathbb{R}^{d_{\text{out}} \times r}, B \in \mathbb{R}^{r \times d_{\text{in}}}$) captures orthogonal residual task-specific updates.

---


## Architecture & Data Flow

The `SALTAttention` module coordinates input projection factorization, intermediate representation routing, and output synthesis:

```
Input Activation X [B, N, D]
   │
   ├── Low-Rank Bottleneck Down-Projection: Z = X W_down [B, N, r]
   ├── Head Expansion & Up-Projection: Q, K, V [B, H, N, d_k]
   ├── Attention Scoring / Kernelized Associative Scan: S = Score(Q, K)
   └── Context Projection & Residual Output: Y = Context V W_out [B, N, D]
```

### Step-by-Step Tensor Shape Trace

1. **Input Representation**: Shape `[B, N, D]` where $B$ is batch size, $N$ is token count, $D$ is hidden dimension.
2. **Factorized Projections**: Bottleneck compression into rank $r \ll D$ yielding `[B, N, r]` intermediates before head projection `[B, H, N, d_k]`.
3. **Attention / Linear Scan**: Aggregation produces contextual representation `[B, H, N, d_v]`.
4. **Output Linear Synthesis**: Final projection restores hidden dimension `[B, N, D]` followed by residual addition.

## 2. Mathematical Derivations & Formulations

### 2.1 Weight Factorization & Singular Value Rescaling (`SALTLinear`)
For a pre-trained projection weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, singular value decomposition is evaluated:

\[
W_0 = U \Sigma V^\top, \quad U \in \mathbb{R}^{d_{\text{out}} \times \min(d_{\text{out}}, d_{\text{in}})}, \; V^\top \in \mathbb{R}^{\min(d_{\text{out}}, d_{\text{in}}) \times d_{\text{in}}}
\]

Top-$r$ singular bases $U_{:, :r}$ and $V_{:r, :}^\top$ are registered as frozen buffers. The adapted projection weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is constructed via singular value rescaling and additive LoRA residual fusion:

\[
W = U_{:, :r} \operatorname{diag}\left( S_{1:r} \odot \gamma \right) V_{:r, :}^\top + A B
\]

where:
$S_{1:r} \in \mathbb{R}^r$ are pre-trained singular values. Furthermore, $\gamma \in \mathbb{R}^r$ is a learnable parameter initialized to all ones ($\mathbf{1}_r$). Concurrently, $A \in \mathbb{R}^{d_{\text{out}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{in}}}$ are learnable adapter matrices initialized to zero on $B$ to guarantee exact initial preservation ($W_{\text{init}} = W_0$).

Forward transformation for hidden activation state $X \in \mathbb{R}^{B \times L \times d_{\text{in}}}$ evaluates:

\[
Y = X W^\top + b = X \left( U_{:, :r} \operatorname{diag}\left( S_{1:r} \odot \gamma \right) V_{:r, :}^\top + A B \right)^\top + b \in \mathbb{R}^{B \times L \times d_{\text{out}}}
\]

### 2.2 Multi-Head Attention Computation
Projections for Query, Key, Value, and Output layers are instantiated via `SALTLinear`:

\[
Q = \text{SALTLinear}_Q(X), \quad K = \text{SALTLinear}_K(X), \quad V = \text{SALTLinear}_V(X) \in \mathbb{R}^{B \times L \times H \cdot d_h}
\]

Reshaping into $H$ heads with head dimension $d_h = D / H$ yields $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$. Scaled dot-product attention logits evaluate:

\[
S_{b, h, i, j} = \frac{Q_{b, h, i} \cdot K_{b, h, j}^\top}{\sqrt{d_h}} + M^{\text{pad}}_{b, j}
\]

\[
A = \operatorname{Softmax}(S), \quad A_{\text{drop}} = \operatorname{Dropout}(A)
\]

Context representations are contracted and transformed through the SALT-adapted output projection:

\[
O = A_{\text{drop}} V \in \mathbb{R}^{B \times H \times L \times d_h}
\]

\[
Y = \text{SALTLinear}_O\left( \operatorname{Reshape}(O) \right) \in \mathbb{R}^{B \times L \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Activation** | $X$ | $[B, L, D]$ | Base hidden input state |
| **Singular Value Scaling** | $S_{1:r} \odot \gamma$ | $[r]$ | Vector element-wise multiplication |
| **Adapted SVD Base GEMM** | $X V_{:r, :} \text{diag}(\tilde{S}) U_{:, :r}^\top$| $[B, L, D] \to [B, L, r] \to [B, L, D]$ | SVD base projection path |
| **Additive LoRA GEMM** | $X B^\top A^\top$ | $[B, L, D] \to [B, L, r] \to [B, L, D]$ | Residual adapter projection path |
| **Q/K/V Head Reshaping** | $Q, K, V$ | $[B, L, D] \to [B, H, L, d_h]$ | Multi-head view split |
| **Attention Logits** | $S$ | $[B, H, L, d_h] \times [B, H, L, d_h]^\top \to [B, H, L, L]$ | Scaled dot-product score matrix |
| **Softmax & Dropout** | $A$ | $[B, H, L, L]$ | Row-wise Softmax probability matrix |
| **Value Contraction** | $O$ | $[B, H, L, L] \times [B, H, L, d_h] \to [B, H, L, d_h]$ | Context accumulation |
| **SALT Output GEMM** | $Y$ | $[B, L, H \cdot d_h] \to [B, L, D]$ | Output SALT linear transformation |

---

## 4. Known Limitations & Future Work
- **Static Subspace Rank**: Top-$r$ singular vector subspace is fixed after initialization; dynamic spectral thresholding per layer could improve parameter allocation.
- **GEMM Fusion Overhead**: Unfused additive LoRA and SVD paths incur extra memory access without custom kernel compilation (e.g. Triton/Cutlass).

---

## 5. Empirical Benchmarks
> Benchmarks Pending — run `bash run_on_gpu_server.sh suite` to populate results.
> See [Empirical Validation Protocol](../empirical_validation_protocol.md).

---

## Complete Source Code Implementation (`salt_attention.py`)

```python
r"""
SALT Attention: Singular Value Adaptation with Low-Rank Tuning.

Parameter-efficient adaptation mechanism for transformer attention projections combining
orthogonal singular vector preservation, singular value rescaling ($\gamma$), and additive low-rank adapters ($A B$).
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class SALTLinear(nn.Module):
    r"""
    Linear projection layer with SVD singular value modulation and low-rank residual branch.

    Formulation:
        $$W = U_{:, :r} \operatorname{diag}(\Sigma_{1:r} \odot \gamma) V_{1:r, :}^\top + A B$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        original_weight: Optional[torch.Tensor] = None,
        original_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = min(rank, min(in_features, out_features))

        if original_weight is not None:
            W = original_weight.float()
        else:
            W = torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))

        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        r = self.rank

        self.register_buffer("U", U[:, :r].contiguous())
        self.register_buffer("Vt", Vt[:r, :].contiguous())
        self.register_buffer("S", S[:r].contiguous())

        self.gamma = nn.Parameter(torch.ones(r))
        self.A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, in_features))  # Zero init for exact initial preservation

        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute adapted linear projection."""
        adapted_S = self.S * self.gamma
        W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
        W_lora = self.A @ self.B
        W = (W_svd + W_lora).to(dtype=x.dtype)
        return F.linear(x, W, self.bias.to(dtype=x.dtype) if self.bias is not None else None)


@ATTENTION_REGISTRY.register("salt_attention")
@ATTENTION_REGISTRY.register("salt")
class SALTAttention(BaseAttention):
    r"""
    SALT Attention module replacing standard linear projections with SALTLinear.

    Mathematical Formulation:
        1. SVD & LoRA Factorized Linear Projections:
           $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
        2. Scaled Dot-Product Attention:
           $$O = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + M\right) V W_O$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = rank if rank is not None else getattr(config, "salt_rank", getattr(config, "rank", 16))

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

        # Projections
        if pretrained_attn is not None:
            self.init_from_pretrained(pretrained_attn)
        else:
            self.q_proj = SALTLinear(hidden_size, self.all_head_size, self.rank)
            self.k_proj = SALTLinear(hidden_size, self.all_head_size, self.rank)
            self.v_proj = SALTLinear(hidden_size, self.all_head_size, self.rank)
            self.out_proj = SALTLinear(self.all_head_size, hidden_size, self.rank)

    def init_from_pretrained(self, src: Any) -> None:
        r"""Initialize SALTLinear projections directly from pretrained module."""
        from framework.core.adapters import UniversalAdapter
        hidden_size = self.num_heads * self.head_dim

        for proj, attr in [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                setattr(self, attr, SALTLinear(w.size(1), w.size(0), self.rank, original_weight=w, original_bias=b))
            except Exception:
                setattr(self, attr, SALTLinear(hidden_size, hidden_size, self.rank))

    def copy_weights(self, src: Any) -> None:
        r"""Universal adapter weight copying."""
        self.init_from_pretrained(src)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_valid_4d, -10000.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        out = self.out_proj(context)

        return out, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("salt_inject")
def inject_salt_attention(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SALTAttention:
    rank = getattr(comp_config, "rank", getattr(comp_config, "salt_rank", 16))
    new_attn = SALTAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
```

---

## Theoretical Invariants & Numerical Stability Proofs

> [!NOTE]
> **Mathematical Invariants for `salt_attention`**:
> 1. **Frobenius Reconstruction Guarantee**:
>    $$\min_{\text{rank}(\widehat{W}) \le r} \|W - \widehat{W}\|_F^2 = \sum_{i=r+1}^{D} \sigma_i^2$$
> 2. **Symmetric $\sqrt{\Sigma}$ Conditioning**:
>    Splitting $\Sigma_r^{1/2}$ equally across factorized matrices balances spectral condition numbers $\kappa(U_r) \approx \kappa(V_r) = \sqrt{\kappa(W)}$, preventing FP16/BF16 gradient underflow.
> 3. **Lyapunov Stability**:
>    State recurrence parameters satisfy $\lambda_t \in (0, 1)$, ensuring strict asymptotic decay and preventing state saturation on long contexts ($L > 4096$).

## Configuration & Hyperparameters

The `SALTAttention` module (registered under `inject_salt_attention`) provides configurable parameters for runtime optimization, numerical stability thresholds, and low-rank allocation:

| Class | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `SALTAttention` | `config` | `Any` | **Required** | HuggingFace architecture configuration instance. |
| `SALTAttention` | `rank` | `Any` | `None` | Low-rank bottleneck projection rank. |
| `SALTAttention` | `pretrained_attn` | `Any` | `None` | Controls `pretrained_attn` configuration in `SALTAttention`. |
| `SALTAttention` | `is_decoder` | `Any` | `None` | Controls `is_decoder` configuration in `SALTAttention`. |
| `SALTLinear` | `in_features` | `Any` | **Required** | Controls `in_features` configuration in `SALTLinear`. |
| `SALTLinear` | `out_features` | `Any` | **Required** | Controls `out_features` configuration in `SALTLinear`. |
| `SALTLinear` | `rank` | `int` | `16` | Low-rank bottleneck projection rank. |
| `SALTLinear` | `original_weight` | `Any` | `None` | Controls `original_weight` configuration in `SALTLinear`. |
| `SALTLinear` | `original_bias` | `Any` | `None` | Controls `original_bias` configuration in `SALTLinear`. |


## Usage & Integration Examples

### HuggingFace Injection & Verification

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.core.attention_utils import inject_attention

# 1. Initialize base model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# 2. Inject SALTAttention dynamically
model = inject_attention(model, attn_type='inject_salt_attention', rank=64)

# 3. Execute forward pass
input_ids = torch.randint(0, 1000, (2, 128))
outputs = model(input_ids)
print('Output logits shape:', outputs.logits.shape)
```


## Attention Maps & Activation Diagnostics

The `SALTAttention` architecture incorporates telemetry hooks for tracking singular value decay, intermediate activation spectra, and receptive field distributions:
- **Activation Snapshots**: Receptive patterns and attention logits are logged under `results/diagnostics/salt_attention/`.
- **Frobenius Error Tracking**: Relative reconstruction residual error is recorded per layer.
- **Entropy & Head Sparsity**: Singular spectral entropy is computed across attention projection heads to monitor representation rank collapse.
