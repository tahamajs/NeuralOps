<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Block Decomposition

The pretrained block:

$$
Y_\text{pre} = \text{LN}_\text{pre}(X + \text{Attn}_\text{pre}(X))
\tag{A.1}
$$

Our block (non-nested):

$$
Y_\text{new} = \text{LN}_\text{new}(X + \text{Attn}_\text{new}(X))
\tag{A.2}
$$

**Error:**

$$
Y_\text{new} - Y_\text{pre} = (\text{LN}_\text{new} - \text{LN}_\text{pre})(\dots) + \text{LN}_\text{pre}(\text{Attn}_\text{new} - \text{Attn}_\text{pre})(X)
\tag{A.3}
$$

**Two error sources:**
1. LayerNorm difference
2. Attention difference

## A.2 LayerNorm Sensitivity

The derivative of LN w.r.t. $\gamma$:

$$
\frac{\partial \text{LN}}{\partial \gamma_i} = \frac{X_i - \mu_i}{\sigma_i + \epsilon}
\tag{A.4}
$$

If $\sigma$ is small (activations are nearly constant), the derivative is 
**large** — a small $\Delta \gamma$ produces a large $\Delta$ output.

## A.3 Silent Failure Analysis

The `try/except` catches exceptions. If the transfer fails:

$$
W_\text{new} \sim \mathcal{N}(0, \sigma^2) \quad \text{(random init)}
\tag{A.5}
$$

Output is meaningless.

## A.4 Non-Causal MHA

`nn.MultiheadAttention` computes:

$$
\text{Attn}_i = \sum_j \text{softmax}(Q_i K_j^\top / \sqrt{d}) V_j
\tag{A.6}
$$

**Without causal masking**, $j$ ranges over all tokens including future ones.

For GPT-2, this violates causality: $y_i$ depends on $x_j$ for $j > i$.

## A.5 Combined Error Bound

$$
\|Y_\text{new} - Y_\text{pre}\|_F \leq \underbrace{\|\text{LN}_\text{new} - \text{LN}_\text{pre}\|_\text{Lip} \cdot \|X + A\|_F}_{\text{LN error}} + \underbrace{\|\text{LN}_\text{pre}\|_\text{Lip} \cdot \|A_\text{new} - A_\text{pre}\|_F}_{\text{attn error}}
\tag{A.7}
$$

Even small LayerNorm mismatch ($\|\text{LN}_\text{new} - \text{LN}_\text{pre}\| \sim 0.1$) 
produces ~10% output error.

## A.6 Parameter Count

Per block:

| Component | Params |
|---|---|
| Attention | $4 d^2 + \text{LoRA}$ |
| LayerNorm $\gamma$ | $d$ |
| LayerNorm $\beta$ | $d$ |
| **Total new** | $4 d^2 + \text{LoRA} + 2d$ |

---