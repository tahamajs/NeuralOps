<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Linear Attention

**Katharopoulos et al. (2020)** introduced the reordering trick:

$$
\text{LinAttn}(Q, K, V)_i = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{2.1}
$$

with $\phi(x) = \mathrm{elu}(x) + 1$. Complexity: $\mathcal{O}(L d^2)$ — 
linear in $L$ but quadratic in $d$. Since $d \ll L$ for long sequences, 
this is a win.

**Choromanski et al. (2021)** used positive orthogonal random features 
(PORF) to approximate the softmax kernel more accurately. Performer achieves 
unbiased estimation but requires $m \gg d$ features for low error.

**Schlag et al. (2021)** showed that linear attention with $\phi(x) = 
\mathrm{ReLU}(x)$ or $\mathrm{elu}(x) + 1$ cannot represent certain 
induction-head behaviors.

**Zhang et al. (2024)** generalized linear attention to GLA (Gated Linear 
Attention) with data-dependent decay.

## 2.2 Rank-Augmented Attention

**Low-rank attention.** Linformer (Wang et al., 2020) projects $K, V$ to a 
low rank globally. This reduces the KV-cache but loses head-specific 
structure.

**Nyströmformer** (Xiong et al., 2021). Nyström approximation of the attention 
matrix. Landmark points are selected globally.

**Rank-1 attention.** Some efficient attention variants add a rank-1 
correction to the linear attention output (e.g., Delta Net, Yang et al., 
2024). RALA generalizes this to rank-$r$ per head.

## 2.3 Kernel Feature Maps

| Kernel | $\phi(x)$ | Reference |
|---|---|---|
| elu+1 | $\mathrm{elu}(x) + 1$ | Katharopoulos et al., 2020 |
| ReLU | $\max(0, x)$ | Schlag et al., 2021 |
| Softplus | $\log(1 + e^x)$ | This work |
| Squared ReLU | $\max(0, x)^2$ | Zhang et al., 2024 |
| Exponential | $e^x$ (unstable) | — |
| RFF | $\cos(\omega x + b)$ | Choromanski et al., 2021 |

**Why softplus?** Smooth, strictly positive, unbounded. No kink at $x = 0$ 
(unlike elu+1). Numerically stable (in contrast to $e^x$).

## 2.4 Position of Our Work

| Axis | Prior work | RALA |
|---|---|---|
| Complexity | $\mathcal{O}(L)$ | $\mathcal{O}(L)$ |
| Kernel | elu+1 / ReLU / RFF | Softplus |
| Augmentation | None / rank-1 | Rank-$r$ per head |
| Gate | None | Scalar $\alpha$ (init 0) |
| Parameter reduction | Sometimes | No (increases) |

**Key distinction.** RALA adds a **low-rank per-head correction** to linear 
attention. This is closest to Delta Net (Yang et al., 2024) but uses a 
**learned scalar gate** rather than a data-dependent update.

---

## Comparison Table

| Method | Complexity | Kernel | Augmentation | Gate |
|---|---|---|---|---|
| Softmax | $\mathcal{O}(L^2)$ | — | — | — |
| LinAttn | $\mathcal{O}(L)$ | elu+1 | — | — |
| Performer | $\mathcal{O}(L)$ | RFF | — | — |
| Linformer | $\mathcal{O}(L)$ | — | Low-rank proj | — |
| Delta Net | $\mathcal{O}(L)$ | — | Rank-1 data-dep | — |
| GLA | $\mathcal{O}(L)$ | — | Data-dep decay | — |
| **RALA** | **$\mathcal{O}(L)$** | **softplus** | **Rank-$r$ per head** | **Scalar $\alpha$** |

---