<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Recurrent Neural Networks

An RNN maintains a hidden state $h_t$ updated as:

$$
h_t = f(h_{t-1}, x_t), \quad y_t = g(h_t)
\tag{3.1}
$$

The **sequential** nature prevents parallelization across time. This is the 
core trade-off: linear attention has better asymptotic complexity but worse 
parallelism.

## 3.2 Associative Scan

For operations of the form $h_t = a_t h_{t-1} + b_t$ (linear recurrences), 
the **associative scan** computes all $h_t$ in parallel:

$$
h_t = \sum_{s=0}^{t} \left(\prod_{u=s+1}^{t} a_u\right) b_s
\tag{3.2}
$$

This reduces wall-clock time from $\mathcal{O}(L)$ to $\mathcal{O}(\log L)$ 
with sufficient parallel hardware.

**DeltaNet does not admit a simple associative scan** because the state 
update involves a nonlinear operation (division by $1 + \|k\|^2$).

## 3.3 The Delta Rule

The Delta Rule (Widrow & Hoff, 1960) is a classical learning rule for linear 
associative memory. Given a key $k_t$ and target $v_t$:

$$
\Delta W = \eta (v_t - W k_t) k_t^\top
\tag{3.3}
$$

The update **corrects** the prediction error $v_t - W k_t$ in the direction 
of $k_t$. This is the basis of the DeltaNet attention mechanism.

**Interpretation.** DeltaNet maintains an associative memory: the state $S$ 
maps keys to values. When a new key-value pair arrives, the state is updated 
to reduce the prediction error.

## 3.4 Exponential Decay Gating

The Delta Rule alone has no forgetting mechanism. Adding a scalar decay:

$$
S_t = \alpha_t S_{t-1} + \beta_t k_t^\top (v_t - k_t S_{t-1})
\tag{3.4}
$$

allows the model to forget old information. When $\alpha_t$ and $\beta_t$ 
are **input-dependent**, the model can selectively remember or forget.

## 3.5 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} BA
\tag{3.5}
$$

with $B = 0$ at initialization. Only $A, B$ are trained.

## 3.6 Complexity of Attention

| Mechanism | Time | Memory |
|---|---|---|
| Standard softmax attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| Linear attention (state-based) | $\mathcal{O}(L d^2)$ | $\mathcal{O}(d^2)$ |
| **Gated DeltaNet** | $\mathcal{O}(L d^2)$ | $\mathcal{O}(d^2)$ |

For $d = 64$ and $L = 128$, linear attention does $\sim 128 \times 64^2 = 
524K$ FLOPs vs standard attention's $\sim 128^2 \times 64 = 1M$ FLOPs. 
So linear attention is **~2× fewer FLOPs** but with worse GPU utilization.

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of heads |
| $S_t$ | State matrix at time $t$ |
| $\alpha_t, \beta_t$ | Input-dependent gates |
| $\eta$ | Delta Rule learning rate |
| $r_L$ | LoRA rank |
| $C$ | Chunk size (implementation) |
</parameter>

---