<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Decomposition

### 2.1.1 SVD and Eckart–Young

The truncated SVD is optimal in Frobenius norm (Eckart & Young, 1936). However, 
the factors $U, V$ are dense and orthogonal, which limits interpretability.

### 2.1.2 CUR Decomposition

Mahoney & Drineas (2009) introduced CUR as an interpretable alternative: 
select $r$ columns and $r$ rows of $W$, then reconstruct as $C \cdot U^+ \cdot R$ 
where $U$ is the intersection. The key insight is that **statistical leverage 
scores** (Mahoney et al., 2012) provide the optimal sampling probabilities. 
PHANTOM uses deterministic top-$r$ selection by leverage score.

### 2.1.3 Randomized SVD

Halko et al. (2011) provide randomized SVD with complexity 
$\mathcal{O}(d^2 r)$. We use exact SVD for leverage computation because 
matrix sizes are manageable.

## 2.2 Sparse Attention

### 2.2.1 Fixed-Pattern Sparsity

Longformer (Beltagy et al., 2020) uses a sliding window plus global tokens. 
BigBird (Zaheer et al., 2020) adds random edges. Both use **fixed** patterns 
that do not adapt to input.

### 2.2.2 Content-Based Sparsity

Reformer (Kitaev et al., 2020) uses LSH to group similar queries and keys. 
Routing Transformer (Roy et al., 2021) uses online k-means. PHANTOM uses 
**multi-round angular LSH** with a rectangular hash function that is 
differentiable through sign operations.

### 2.2.3 Comparisons

| Method | Pattern | Adaptive | Complexity |
|---|---|---|---|
| Longformer | Sliding window + global | No | $\mathcal{O}(L \sqrt{L})$ |
| BigBird | Window + random + global | No | $\mathcal{O}(L)$ |
| Reformer | LSH buckets | Yes | $\mathcal{O}(L \log L)$ |
| Routing | k-means clusters | Yes | $\mathcal{O}(L \log L)$ |
| **PHANTOM** | Multi-round angular LSH | Yes | $\mathcal{O}(L \cdot B)$ |

## 2.3 Attention Activations

### 2.3.1 Softmax

Standard softmax has two limitations: (i) it never reaches exactly zero, so 
far-tail positions receive non-negligible weight; (ii) it is sensitive to 
outliers.

### 2.3.2 Sparsemax

Martins & Astudillo (2016) propose sparsemax, which projects onto the simplex 
and produces sparse attention. However, sparsemax has no closed-form for 
batched computation.

### 2.3.3 Softpick

Lu et al. (2024) propose softpick: $\text{ReLU}(e^{(x_i - m)} - e^{-m})$ 
where $m = \max_j x_j$. This is sparse and has closed form. We use a 
**rectified** variant with a softmax fallback when the denominator is zero.

## 2.4 Post-Training Compression

| Method | Type | Retrain? | Compression | Attention? |
|---|---|---|---|---|
| SVD (Eckart-Young) | Low-rank | No | Ratio $r$ | No |
| A3 | Fisher-weighted SVD | LoRA | Ratio $r$ | No |
| AASVD | Covariance-weighted SVD | LoRA | Ratio $r$ | No |
| SparseGPT | Pruning | No | 50% | No |
| **PHANTOM** | CUR + LSH | LoRA | Ratio $r$ | **Yes** |

**Key distinction.** PHANTOM is the only method in this list that compresses 
both **projections** (via CUR) **and attention** (via LSH).

## 2.5 Position of Our Work

PHANTOM is unique in combining:
1. **CUR decomposition** for interpretable low-rank projections.
2. **Multi-round LSH** for content-adaptive sparse attention.
3. **Rectified softpick** for stable sparse attention.

Prior work addresses each of these individually, but no prior method combines 
all three.

---