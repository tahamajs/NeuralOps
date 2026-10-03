<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Local Window Attention

$$
O_{\text{loc}, i} = \frac{\sum_{j: |i - j| \leq W, j \leq i} \exp(q_i^\top k_j / \sqrt{d}) v_j}{\sum_{j} \exp(q_i^\top k_j / \sqrt{d})}
\tag{A.1}
$$

Standard softmax over the window.

## A.2 Sparse Block Scores

For blocks $i, j$:

$$
s_{ij} = \frac{\bar{q}_i^\top \bar{k}_j}{\sqrt{d}}, \quad
\bar{q}_i = \frac{1}{B_s} \sum_{t \in \text{block}_i} q_t
\tag{A.2}
$$

Selection:

$$
\mathcal{B}_i = \text{topk}_{k}\{s_{ij}\}_j, \quad
O_{\text{sp}, i} = \text{softpick}(Q_i K_{\mathcal{B}_i}^\top / \sqrt{d}) V_{\mathcal{B}_i}
\tag{A.3}
$$

## A.3 RFF Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)]
\tag{A.4}
$$

$\phi(x) \in \mathbb{R}^{2F}$.

**Approximation.**

$$
\phi(q)^\top \phi(k) = \frac{1}{F} [\cos(\Omega q)^\top \cos(\Omega k) + \sin(\Omega q)^\top \sin(\Omega k)]
\tag{A.5}
$$

For $\Omega$ with orthogonal rows, this approximates $\exp(q^\top k)$.

## A.4 Causal RFF via Cumsum

For causal decoder:

$$
(KV_c)_i = \sum_{j \leq i} \phi(k_j) v_j^\top
\tag{A.6}
$$

$$
(Z_c)_i = \sum_{j \leq i} \phi(k_j)
\tag{A.7}
$$

Computed via `cumsum` along sequence dimension.

$$
O_{\text{glb}, i} = \frac{\phi(q_i)^\top (KV_c)_i}{\phi(q_i)^\top (Z_c)_i + \epsilon}
\tag{A.8}
$$

## A.5 Gate Computation

$$
g = \text{softmax}(W_g \bar{x}), \quad
\bar{x} = \frac{1}{L} \sum_{t=1}^{L} x_t, \quad
W_g \in \mathbb{R}^{3 \times d}
\tag{A.9}
$$

## A.6 Fusion

$$
O = g_0 O_{\text{loc}} + g_1 O_{\text{sp}} + g_2 O_{\text{glb}}
\tag{A.10}
$$

## A.7 Softpick with Fallback

$$
p_i = \text{ReLU}(e^{x_i - m} - e^{-m})
\tag{A.11}
$$

where $m = \max_j x_j$ (clamped to $0$ if $m < -10^3$).

$$
\hat{p}_i = \begin{cases} p_i / \sum_j p_j & \text{if } \sum_j p_j > \epsilon \\ \text{softmax}(x)_i & \text{otherwise} \end{cases}
\tag{A.12}
$$

**Batch-level fallback.** The `torch.where` is applied element-wise, but 
the fallback is broadcast to all elements where `zero_mask` is True.

## A.8 Complexity

| Branch | FLOPs |
|---|---|
| Local window | $B H L W d$ |
| Sparse block | $B H L k B_s d$ |
| RFF | $B H L F d$ |
| Gate | $B L d$ |
| **Total** | $B H L (W + k B_s + F) d$ |

## A.9 Memory of Causal RFF

The `KV_cumsum` tensor size:

$$
\text{size} = B H L (2F) d
\tag{A.13}
$$

For $B=8, H=12, L=128, F=64, d=64$: 50 MB in FP32.

---