<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 The Delta Rule

The Delta Rule (Widrow–Hoff, 1960) updates a weight matrix $W$ to 
minimize squared error:

$$
\Delta W_t = \eta (v_t - W_t x_t) x_t^\top
\tag{A.1}
$$

For attention, $x_t = k_t$ (key) and target is $v_t$ (value). The state 
$S$ replaces $W$:

$$
S_t = S_{t-1} + \eta (v_t - S_{t-1} k_t) k_t^\top
\tag{A.2}
$$

## A.2 Adding Gating

Replace fixed step size $\eta$ with input-dependent scalars:

$$
S_t = \beta_t S_{t-1} + \alpha_t (v_t - S_{t-1} k_t) k_t^\top
\tag{A.3}
$$

where $\alpha_t, \beta_t \in (0, 1)$ are computed via sigmoid.

**Note on our implementation.** We use $S_{t-1} k_t$ instead of $k_t S_{t-1}$ 
because keys are row vectors in our convention.

## A.3 Normalization

For numerical stability, we normalize the update:

$$
S_t = \beta_t S_{t-1} + \alpha_t \cdot \frac{(v_t - k_t S_{t-1}) k_t^\top}{1 + \|k_t\|^2}
\tag{A.4}
$$

The denominator $1 + \|k_t\|^2$ prevents overflow when $\|k_t\|$ is large.

## A.4 Key and Query Normalization

We additionally normalize keys and queries:

$$
k_t \leftarrow \frac{k_t}{\|k_t\|_2}, \quad
q_t \leftarrow \frac{q_t}{\|q_t\|_2}
\tag{A.5}
$$

This ensures the inner products lie in $[-1, 1]$, matching softmax attention.

## A.5 Causality of the Forward Scan

**Claim.** The forward scan satisfies:

$$
\frac{\partial y_t}{\partial x_s} = 0 \text{ for } s > t
\tag{A.6}
$$

**Proof.** $y_t = q_t S_t$. By induction, $S_t$ depends only on $k_1, 
\dots, k_t$, $v_1, \dots, v_t$, $\alpha_1, \dots, \alpha_t$, $\beta_1, 
\dots, \beta_t$, which in turn depend only on $x_1, \dots, x_t$. So $y_t$ 
depends only on $x_1, \dots, x_t$. $\square$

## A.6 Bidirectional Average

For encoders, we average forward and backward scans:

$$
y = \frac{1}{2}(y_\text{fwd} + y_\text{bwd})
\tag{A.7}
$$

**Note.** This breaks causality intentionally — encoder models are 
bidirectional.

## A.7 FLOP Counting

For one head, one layer, sequence length $L$, head dim $d_h$:

| Operation | FLOPs |
|---|---|
| Q/K/V/Z projection | $4 L d d_h$ |
| Gating | $2 L d H$ |
| State update | $L (3 d_h^2 + 2 d_h)$ |
| Output $y_t = q_t S_t$ | $L d_h^2$ |
| SiLU | $2 L d_h$ |
| Output projection | $L H d_h d$ |

**Total FLOPs**: $\sim 5 L d d_h + 4 L d_h^2 + \dots \approx \mathcal{O}(L d^2)$.

**Baseline attention FLOPs**: $\sim 4 L^2 d_h + \dots \approx \mathcal{O}(L^2 d)$.

For $L = 128$ and $d = 64$: linear has ~2× fewer FLOPs.

## A.8 Wall-Clock Complexity

Due to Python loop overhead:

$$
T_\text{wall} \approx N_\text{iters} \cdot T_\text{iter}
\tag{A.8}
$$

where $N_\text{iters} = 2L$ and $T_\text{iter} \approx 8$ μs.

For $L = 128$: $T \approx 2 \times 128 \times 8 \mu\text{s} = 2$ ms per 
head. With 12 heads and 6 layers: **144 ms per forward**. Matches observed 
timing.

## A.9 Memory Analysis

| Component | Memory |
|---|---|
| State $S$ | $B H d_h^2$ |
| Query, Key, Value | $3 B L d_h H$ |
| Z | $B L H d_h$ |
| Gating ($\alpha, \beta$) | $2 B L H$ |
| Checkpointed per-chunk | $B H (L/C) d_h^2$ |

For DistilBERT: $S = 8 \times 12 \times 64^2 = 393K$ floats = **1.5 MB per 
layer**. With checkpointing, state is re-created in backward.

---