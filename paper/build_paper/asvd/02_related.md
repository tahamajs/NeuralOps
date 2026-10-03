<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

### 2.1.1 Classical SVD

Eckart & Young (1936) proved that the optimal rank-$r$ Frobenius-norm 
approximation of $W$ is the truncated SVD $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$. 
This is the foundation for all SVD-based compression.

### 2.1.2 Randomized SVD

Halko et al. (2011) provide a randomized algorithm for truncated SVD with 
complexity $\mathcal{O}(mn \log r)$, trading accuracy for speed. For our 
layer sizes (up to 3072×768), exact SVD is fast enough.

### 2.1.3 Weighted Low-Rank Approximation

Golub et al. (1987), Srebro & Jaakkola (2003). Weighted approximation 
minimizes $\|(W - W_r) D\|_F$ where $D$ is a weighting matrix. ASVD uses 
$D = \text{diag}(\sqrt{F})$ where $F$ is the activation energy.

### 2.1.4 Tensor Decomposition

Tensor-train (Novikov et al., 2015) uses higher-order factorization. 
ASVD is a matrix method.

## 2.2 Activation-Aware Compression

### 2.2.1 ASVD Paper

Yuan et al. (2023) — arXiv:2307.07382 — introduced ASVD for LLM 
compression. The core idea: scale the weight matrix by the input activation 
magnitude before SVD. Our implementation extends this with explicit $\Sigma$, 
energy-based dynamic rank, and pre-flight validation.

### 2.2.2 AWQ

Lin et al. (2023) use activation magnitude for weight quantization. 
ASVD uses it for low-rank factorization.

### 2.2.3 SmoothQuant

Xiao et al. (2023) migrate activation scale to weights for INT8 quantization. 
ASVD performs an analogous migration but for the SVD input space.

### 2.2.4 A3

Our companion module A3 uses diagonal Fisher weighting with LayerNorm 
recalibration. ASVD differs by keeping $\Sigma$ explicit and using energy-based 
rank selection instead of adaptive retry.

### 2.2.5 AASVD

Our companion module AASVD uses full covariance weighting with streaming 
EMA. ASVD uses a simpler diagonal weighting.

## 2.3 Parameter-Efficient Fine-Tuning

### 2.3.1 LoRA

Hu et al. (2021) — low-rank adapters with $B=0$ initialization. ASVD uses 
LoRA for recovery after compression.

### 2.3.2 AdaLoRA / DoRA

Zhang et al. (2023), Liu et al. (2024) — adaptive rank allocation and 
weight decomposition. ASVD uses standard LoRA.

## 2.4 Position of Our Work

| Axis | Prior work | ASVD |
|---|---|---|
| **Weighting** | Unweighted or diagonal | Diagonal (activation energy) |
| **Σ handling** | Implicit or folded | **Explicit matrix** |
| **Rank selection** | Fixed or adaptive-retry | **Energy-based dynamic** |
| **Validation** | Rare | **Strict Frobenius with abort** |
| **Pre-flight check** | None | **Extraction validation** |
| **Multi-architecture** | Usually one family | Four families |

**Key distinction.** ASVD's explicit $\Sigma$ matrix distinguishes it from 
A3 and AASVD, which fold the singular values into the factors. This enables 
inspection and rank reduction without re-factorization.

## 2.5 Comparison to Companion Modules

| Module | Weighting | Σ handling | Rank selection |
|---|---|---|---|
| A3 | Fisher diagonal | Folded | Adaptive retry |
| AASVD | Full covariance | Folded | Fixed with cap |
| **ASVD** | Activation energy | **Explicit** | **Energy-based** |
| SDRM | None | Folded | Budget-proportional |

---