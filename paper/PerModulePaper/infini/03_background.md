<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.1}
$$

with $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and:

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.2}
$$

**Complexity:** $\mathcal{O}(L^2 d)$.

## 3.2 Sliding-Window Attention

Restrict attention to a window of size $W$ around each position:

$$
\text{SWA}(X)_i = \text{softmax}\!\left(\frac{Q_i K_{[i-W:i+W]}^\top}{\sqrt{d_h}}\right) V_{[i-W:i+W]}
\tag{3.3}
$$

For causal models, the window is one-sided: $K_{[i-W:i]}$.

**Complexity:** $\mathcal{O}(L \cdot W \cdot d)$.

**Trade-off.** Tokens outside the window cannot attend to each other. 
Global information must propagate through the residual stream, which 
requires many layers.

## 3.3 Compressive Memory

A compressive memory $M \in \mathbb{R}^{d_c \times d_h}$ is a small matrix 
that acts as $d_c$ learned "virtual tokens". Queries attend to it via:

$$
\text{MemAttn}_i = \text{softmax}\!\left(\frac{Q_i M^\top}{\sqrt{d_h}}\right) M
\tag{3.4}
$$

**Intuition.** The memory summarizes the global context in $d_c$ slots. 
Since $d_c \ll L$, this is a form of learned summarization.

**Complexity:** $\mathcal{O}(L \cdot d_c \cdot d_h)$.

## 3.4 Gated Combination

Given two branches (local and memory), we combine them via a scalar gate:

$$
\text{output} = \text{local} + \sigma(\beta) \cdot \text{memory}
\tag{3.5}
$$

with $\beta$ a learnable scalar. The addition (rather than concatenation) 
preserves the pretrained output distribution when $\sigma(\beta) \to 0$.

**Initialization.** Setting $\beta = -5$ gives $\sigma(-5) \approx 0.0067$, 
meaning the memory contributes <1% at start.

## 3.5 Grouped-Query Attention (GQA)

For GQA with $H_{kv} < H$, we repeat $K, V$ $H / H_{kv}$ times before 
attention. Infini-Attention supports GQA via `repeat_kv`.

## 3.6 Rotary Position Embeddings (RoPE)

RoPE rotates Q, K by position-dependent angles:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.6}
$$

Infini-Attention preserves RoPE via `rotary_emb`.

## 3.7 LoRA

LoRA (Hu et al., 2021) freezes $W$ and adds $\frac{\alpha}{r} B A$ with 
$B = 0$ at initialization.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $W$ | Window size |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $d_h$ | Per-head dimension |
| $d_c$ | Compression dimension |
| $M$ | Memory matrix |
| $\beta$ | Gate scalar |
| $r$ | LoRA rank |

---