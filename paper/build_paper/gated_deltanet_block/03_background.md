<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Transformer Block

A transformer block (post-LN variant):

$$
\begin{aligned}
A &= \text{Attention}(X) \\
X' &= \text{LayerNorm}(X + A) \\
F &= \text{FFN}(X') \\
Y &= \text{LayerNorm}(X' + F)
\end{aligned}
\tag{3.1}
$$

Our `GatedDeltaNetBlock` replaces only the **attention + first LayerNorm** 
part, leaving the FFN block untouched (or applies the same pattern inside).

**Important.** When we replace a block, we must ensure the **output 
distribution** matches what the FFN expects. This is why LayerNorm 
initialization matters.

## 3.2 LayerNorm

LayerNorm (Ba et al., 2016) computes:

$$
\text{LN}(X) = \gamma \odot \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\tag{3.2}
$$

with learnable $\gamma, \beta \in \mathbb{R}^d$.

**Initialization.** Standard: $\gamma = 1, \beta = 0$.

**Pretrained.** In pretrained transformers, $\gamma, \beta$ have been 
calibrated against the specific activation distribution that the pretrained 
attention produces. **They are not 1 and 0.**

For example, in a trained BERT block, typical $\gamma$ values range from 
$0.5$ to $2.0$ depending on the layer. Replacing them with $\gamma = 1$ 
produces a different distribution.

## 3.3 `nn.MultiheadAttention`

PyTorch's `nn.MultiheadAttention` (Vaswani et al., 2017) is structured as:

$$
\begin{aligned}
\text{in\_proj\_weight} &\in \mathbb{R}^{3 d_\text{model} \times d_\text{model}} \\
\text{in\_proj\_bias} &\in \mathbb{R}^{3 d_\text{model}} \\
\text{out\_proj} &\in \mathbb{R}^{d_\text{model} \times d_\text{model}}
\end{aligned}
\tag{3.3}
$$

Q, K, V are computed by splitting `in_proj_weight` into three equal parts.

**Comparison with BERT-style attention.**

| Aspect | BERT/DistilBERT | `nn.MultiheadAttention` |
|---|---|---|
| Q/K/V storage | Three separate linears | One fused `in_proj_weight` |
| Weight shapes | `(d, d) × 3` | `(3d, d)` |
| Bias | Three separate | One fused |
| Output | `out_lin` or `output.dense` | `out_proj` |

To transfer weights, we must **concatenate Q/K/V** into a single matrix.

## 3.4 Silent Failure

A `try/except` block that catches broad exceptions can silently succeed 
even when the operation failed:

```python
try:
    ...critical operation...
except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
    pass  # ← silently swallows errors
```

If the operation fails, the code continues as if nothing happened, with 
**partial state**. This is a common source of hard-to-debug bugs.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $X$ | Input hidden states |
| $d$ | Hidden size |
| $H$ | Number of heads |
| $\gamma, \beta$ | LayerNorm scale and shift |
| $\mu, \sigma$ | LayerNorm mean and std |
| $\epsilon$ | LayerNorm numerical constant |
| $r$ | LoRA rank |

---