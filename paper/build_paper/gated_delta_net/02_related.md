<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020). Reformulates attention 
with a kernel function $\phi$:

$$
y_t = \frac{\sum_{s \leq t} \phi(q_t)^\top \phi(k_s) v_s}{\sum_{s \leq t} \phi(q_t)^\top \phi(k_s)}
\tag{2.1}
$$

which can be computed with a running state $S_t = \sum_{s \leq t} \phi(k_s) v_s^\top$. 
**No gating, no error correction.**

**Performer** (Choromanski et al., 2021). Uses random Fourier features for 
$\phi$. Same linear state but with better kernel approximation.

## 2.2 Gated Linear Attention

**RetNet** (Sun et al., 2023). Introduces exponential decay:
$$
S_t = \gamma S_{t-1} + k_t v_t^\top
\tag{2.2}
$$
with fixed (not input-dependent) $\gamma$.

**Mamba** (Gu & Dao, 2023). Adds input-dependent gating to state-space models, 
but does not use the Delta Rule.

**GLA** (Yang et al., 2024). Combines Mamba-style gating with linear attention.

## 2.3 Delta Rule Attention

**DeltaNet** (Schlag et al., 2021). Introduces the Delta Rule to linear 
attention:
$$
S_t = S_{t-1} + \beta_t k_t^\top (v_t - k_t S_{t-1})
\tag{2.3}
$$

**Gated DeltaNet** (Yang et al., 2024). Adds input-dependent forget gate. 
This is the direct inspiration for our work.

**DeltaNet with Parallel Scan** (Yang et al., 2024; Wang et al., 2024). Uses 
WY representation for efficient parallel computation.

## 2.4 State-Space Models

**S4** (Gu et al., 2022), **S5** (Smith et al., 2023), **Mamba** (Gu & Dao, 
2023). Model sequences as linear time-invariant or time-varying systems. 
Related but not equivalent to DeltaNet.

## 2.5 Position of Our Work

| Axis | Prior work | Gated DeltaNet (ours) |
|---|---|---|
| **State update** | Fixed or input-gated | Delta Rule + gating |
| **Bidirectional** | Usually causal only | Both supported |
| **Implementation** | CUDA kernels (Mamba) | **Pure PyTorch (slow)** |
| **Weight transfer** | Trained from scratch | **Pretrained Q/K/V/O** |
| **LoRA recovery** | Rare | Yes |

**Key distinction.** Our contribution is **integration** with pretrained 
transformers — a mechanism to reuse $W_Q, W_K, W_V, W_O$ from a pretrained 
model. The algorithm itself follows Yang et al. (2024).

---

## Comparison Table

| Method | Complexity | Gating | Delta Rule | Bidirectional | Pretrained transfer |
|---|---|---|---|---|---|
| Standard MHSA | $O(L^2)$ | — | — | Yes | — |
| Linear Transformer | $O(L)$ | No | No | Yes | Rare |
| RetNet | $O(L)$ | Fixed | No | No | From scratch |
| Mamba | $O(L)$ | Input | No | No | From scratch |
| DeltaNet | $O(L)$ | Input | Yes | No | Rare |
| **Gated DeltaNet** | **$O(L)$** | **Input** | **Yes** | **Yes** | **Yes** |

---