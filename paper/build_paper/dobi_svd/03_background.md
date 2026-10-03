<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Truncated SVD

For $W \in \mathbb{R}^{m \times n}$:

$$
W = U \Sigma V^\top
\tag{3.1}
$$

Truncated to rank $r$:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:r, :}^\top
\tag{3.2}
$$

Optimal in Frobenius norm by Eckart–Young.

**Symmetric form.** Split $\Sigma^{1/2}$:

$$
W_r = \underbrace{U_{:, :r} \Sigma_{:r, :r}^{1/2}}_{U_r} \cdot \underbrace{\Sigma_{:r, :r}^{1/2} V_{:r, :}^\top}_{V_r}
\tag{3.3}
$$

Dobi-SVD uses this form.

## 3.2 Sigmoid Gating

The sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
\tag{3.4}
$$

- $\sigma(0) = 0.5$
- $\sigma(2.5) \approx 0.924$
- $\sigma(5) \approx 0.993$
- $\sigma(-5) \approx 0.007$

Dobi-SVD uses sigmoid to gate each rank component:

$$
\sigma(\alpha_i) \in (0, 1)
\tag{3.5}
$$

The gate controls the contribution of the $i$-th rank component to the 
output.

## 3.3 Gate-Compensated Initialization

If we initialize $U, V$ from SVD and gates $\alpha = \alpha_\text{init}$, 
the initial output is:

$$
y_\text{init} = (x (V_r \odot \sigma(\alpha_\text{init}))^\top) U_r^\top
\tag{3.6}
$$

This differs from the SVD approximation by a factor of $\sigma(\alpha_\text{init})$ 
in the $V$ factor. To make $y_\text{init}$ equal the SVD approximation, 
we pre-scale $V$:

$$
V_r \leftarrow \frac{V_r}{\sigma(\alpha_\text{init})}
\tag{3.7}
$$

Then at step 0:

$$
y_\text{init} = (x (V_r / \sigma_\text{init} \odot \sigma_\text{init})^\top) U_r^\top = (x V_r^\top) U_r^\top
\tag{3.8}
$$

**This is exactly the SVD approximation.** The gating is a differentiable 
refinement.

## 3.4 Multi-Head Self-Attention

Standard MHSA:

$$
\begin{aligned}
Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
\text{MHSA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\end{aligned}
\tag{3.9}
$$

Dobi-SVD factorizes $W_Q, W_K, W_V, W_O$ (when `target_attention_only = True`).

## 3.5 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads. Repeated $H / H_{kv}$ times during forward.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable residuals:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $B = 0$ at initialization. In Dobi-SVD, LoRA is applied to the 
**wrapped layer**, not inside `DobiSVDLinearLayer`.

## 3.7 Conv1D in GPT-2

GPT-2 uses HuggingFace `Conv1D`, storing `(in, out)`:

```python
self.weight = nn.Parameter(torch.empty(nx, nf))
```

Dobi-SVD detects this and transposes before SVD.

## 3.8 Orthogonal Initialization

Dobi-SVD initializes $U, V$ orthogonally for numerical stability. For 
FP16, initialization is done in FP32 and cast.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $W, U, V$ | Weight and factors |
| $\alpha$ | Gate logits |
| $\sigma(\alpha)$ | Sigmoid gates |
| $r$ | Nominal rank |
| $r_\text{eff}$ | Effective rank |
| $\tau_g$ | Gate pruning threshold |
| $d, H, H_{kv}, d_h$ | Model dims |

---