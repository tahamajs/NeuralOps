<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention

### 2.1.1 Sparse Attention

Longformer (Beltagy et al., 2020) restricts attention to a sliding window 
plus global tokens, achieving $\mathcal{O}(L \sqrt{L})$. BigBird (Zaheer et 
al., 2020) adds random connections. Primal Attention is **dense** in content 
but **low-rank** in structure.

### 2.1.2 Kernel Attention

Performer (Choromanski et al., 2021) approximates softmax via random Fourier 
features, achieving $\mathcal{O}(L)$ with a fixed number of random features. 
Primal Attention uses landmarks instead of random features.

### 2.1.3 Low-Rank Attention

Linformer (Wang et al., 2020) projects $K, V$ to a fixed low rank via learned 
projections. Nyströmformer (Xiong et al., 2021) uses Nyström landmark 
approximation with the same landmarks for $Q$ and $K$. Primal Attention 
**breaks this symmetry**.

### 2.1.4 Performer vs Nyström

| Method | Approximation | Symmetric? | Trainable? |
|---|---|---|---|
| Performer | Random Fourier features | Yes | Yes |
| Nyströmformer | Symmetric landmarks | Yes | Yes |
| **Primal Attention** | Asymmetric landmarks | **No** | **Yes** |

## 2.2 Nyström Methods

### 2.2.1 Classical Nyström

Williams & Seeger (2001) approximate a Gram matrix $K_{L \times L}$ as 
$C W^{-1} C^\top$. This requires $C \in \mathbb{R}^{L \times m}$ and 
$W \in \mathbb{R}^{m \times m}$.

### 2.2.2 Nyströmformer

Xiong et al. (2021) apply Nyström to attention by selecting $m$ landmarks 
via **mean pooling** with stride $L/m$, then using the same landmarks for 
both $Q$ and $K$. Primal Attention generalizes this by allowing separate 
landmarks.

### 2.2.3 Randomized Nyström

Drineas & Mahoney (2005) select landmarks randomly. Li et al. (2021) use 
k-means clustering. Primal Attention uses **segment pooling** for simplicity 
and determinism.

## 2.3 Pseudoinverse Computation

### 2.3.1 SVD-based Pseudoinverse

Standard: $A^+ = V \Sigma^+ U^\top$. Cost: $\mathcal{O}(m^3)$ per layer, 
which is the dominant cost.

### 2.3.2 Schulz Iteration

Schulz (1933) proposed the iterative scheme $X_{t+1} = X_t (2I - A X_t)$, 
which converges quadratically to $A^{-1}$. We adapt this for pseudoinverse 
with a slight modification.

### 2.3.3 Newton-Schulz in Deep Learning

Bernstein & Newhouse (2024) use Newton-Schulz iteration for orthogonalization 
in optimizers. We use it for attention pseudoinverse.

## 2.4 Position of Our Work

| Axis | Nyströmformer | Primal Attention |
|---|---|---|
| Landmark selection | Mean pooling (shared) | Segment pooling (asymmetric) |
| Kernel symmetry | Symmetric | **Asymmetric** |
| Pseudoinverse | SVD-based | **Schulz iteration** |
| Differentiable | Yes | Yes |
| GQA support | No | Yes |
| RoPE support | Partial | Full |

**Key distinction.** Primal Attention's asymmetric formulation allows the 
model to use different summaries for $Q$ and $K$, capturing the different 
roles of these projections. This is not possible in standard Nyström 
approximation.

---
