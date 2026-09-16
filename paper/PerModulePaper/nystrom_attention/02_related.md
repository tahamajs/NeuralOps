<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention Mechanisms

### 2.1.1 Sparse Attention

Longformer (Beltagy et al., 2020) restricts attention to a sliding window 
plus global tokens. BigBird (Zaheer et al., 2020) adds random attention. 
These methods **sparsify** the attention pattern; Nyströmformer 
**approximates** it.

### 2.1.2 Low-Rank Attention

Linformer (Wang et al., 2020) projects $K, V$ to a fixed low rank. Nyströmformer 
uses landmark-based reconstruction — a **different** low-rank strategy.

### 2.1.3 Kernel Attention

Performer (Choromanski et al., 2021) uses random features. Linear Transformer 
(Katharopoulos et al., 2020) uses $\phi(x) = \mathrm{elu}(x) + 1$. Nyströmformer 
uses landmarks instead of random features.

## 2.2 The Nyström Method

The Nyström method (Nyström, 1930) approximates a kernel matrix $K \in \mathbb{R}^{N \times N}$ 
using a subset of $m$ columns:

$$
K \approx K_{:, m} K_{m, m}^{-1} K_{m, :}
\tag{2.1}
$$

This is exact if $K$ has rank $m$. In practice, kernel matrices are 
approximately low-rank, making the approximation useful.

**Application to transformers.** Nyströmformer (Xiong et al., 2021) applies 
Nyström to the softmax attention matrix.

## 2.3 Landmark Selection

**Random landmarks.** Simple but noisy.

**K-means landmarks.** Better but expensive.

**Adaptive pooling landmarks.** Nyströmformer uses `adaptive_avg_pool1d` 
to select landmarks. This is efficient and deterministic.

## 2.4 Pseudoinverse Computation

**Direct pinv.** `torch.linalg.pinv` uses SVD. $\mathcal{O}(m^3)$.

**Iterative Newton-Schulz.** For a matrix $A$ with $\|A\|_2 < 1$:

$$
V_{k+1} = V_k (2I - A V_k)
\tag{2.2}
$$

converges to $A^{-1}$. Preconditioning with $\|A\|_1 \|A\|_\infty$ 
extends to general matrices:

$$
\alpha = \frac{1}{\|A\|_1 \|A\|_\infty + \epsilon}
\tag{2.3}
$$

Nyströmformer uses iterative pinv for efficiency.

## 2.5 Position of Our Work

| Aspect | Xiong et al. 2021 | Our work |
|---|---|---|
| **Target architecture** | BERT | Four families |
| **Landmark selection** | Adaptive pooling | Same |
| **Pinv method** | Iterative | Same + fallback |
| **Short sequence handling** | Dense fallback | Same + tuned threshold |
| **Causal handling** | Not addressed | Explicit decoder path |
| **Reproduction** | — | This paper |

**Key distinction.** We do not claim to improve Nyströmformer. We reproduce 
it, port it to non-BERT architectures, and document the engineering effort.

## 2.6 Comparison with Companion Methods

| Method | Reduces | Orthogonal to |
|---|---|---|
| MinimalSVD | Parameters | Nyströmformer |
| SVD-LLM V2 | Parameters | Nyströmformer |
| Dobi-SVD | Parameters | Nyströmformer |
| MLA | KV cache | Nyströmformer |
| **Nyströmformer** | **Attention compute** | All above |

**All methods are complementary.** Nyströmformer reduces compute; the others 
reduce memory. They can be combined.

---