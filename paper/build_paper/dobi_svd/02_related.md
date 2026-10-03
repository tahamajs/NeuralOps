<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Classical SVD.** The Eckart–Young theorem (Eckart & Young, 1936) states 
that the optimal rank-$r$ approximation of $W$ in Frobenius norm is the 
truncated SVD. Dobi-SVD uses this for initialization.

**Randomized SVD.** Halko et al. (2011) provide randomized algorithms. Not 
used here; exact SVD is feasible for our sizes.

**Symmetric vs. explicit factorization.** Standard implementations distribute 
$\Sigma^{1/2}$ between $U$ and $V$ (Golub & Van Loan, 2013). Dobi-SVD 
uses symmetric form and pre-scales $V$ by the gate compensation factor.

**Weighted low-rank.** Golub et al. (1987); Srebro & Jaakkola (2003). 
Activation-aware methods (Lin et al., 2023; Xiao et al., 2023) use 
activation statistics. Dobi-SVD uses unweighted SVD with **learnable gating**.

## 2.2 Rank Selection Strategies

### 2.2.1 Fixed Rank

Most methods use a single global $r$ (Yu et al., 2017). Simple but ignores 
layer heterogeneity.

### 2.2.2 Energy-Based Rank

Choose the smallest $r$ with cumulative spectral energy $\geq \eta$. 
Training-free. Used by MinimalSVD, MinimalNOVA, MinimalPRISM.

### 2.2.3 Learned Rank (AdaLoRA)

AdaLoRA (Zhang et al., 2023) learns per-layer rank budgets via 
importance scores. Related but different: AdaLoRA learns a **budget**, 
Dobi-SVD learns **which rank components survive**.

### 2.2.4 Differentiable Pruning

Louizos et al. (2018) use $L_0$ gates for structured pruning. Dobi-SVD 
applies a similar idea to **rank components** in SVD factorization.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) freeze base weights and train rank-$r$ adapters. 
Dobi-SVD trains LoRA + gates jointly.

**DoRA.** Liu et al. (2024) decompose weights into magnitude and direction. 
Dobi-SVD decomposes into rank components with gating.

## 2.4 Attention Variants

**Standard MHSA.** Vaswani et al. (2017).

**GQA.** Ainslie et al. (2023). Supported via `repeat_kv`.

## 2.5 Model Compression

| Method | Rank selection | Differentiable? | Post-hoc pruning? |
|---|---|---|---|
| DistilBERT | — (distillation) | — | — |
| AWQ | — (quantization) | — | — |
| SparseGPT | — (pruning) | — | — |
| MinimalSVD | Energy | ❌ | ❌ |
| SVD-LLM V2 | Energy | ❌ | ❌ |
| MLA | Fixed latent | ❌ | ❌ |
| AdaLoRA | Learned budget | ✅ | Partial |
| **Dobi-SVD** | **Learnable gates** | **✅** | **✅** |

## 2.6 Position of Our Work

Dobi-SVD differs from prior work along three axes:

| Axis | Prior | Dobi-SVD |
|---|---|---|
| **Rank representation** | Integer | Continuous gates |
| **Differentiability** | No | Yes |
| **Post-hoc pruning** | Retrain | Threshold gates |

**Key distinction.** Unlike fixed or energy-selected rank, Dobi-SVD's rank 
is **learned jointly with recovery**. After training, gates provide a 
data-driven signal for which components to keep.

---