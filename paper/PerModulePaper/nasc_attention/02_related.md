<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Hybrid Attention

**Longformer** (Beltagy et al., 2020). Combines sliding-window with global 
tokens.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global.

**Hybrid Transformer** (Zhu et al., 2021). Local attention + low-rank 
global.

**Routing Transformer** (Roy et al., 2021). Content-based clustering.

**NASC** combines three branches and uses a gate.

## 2.2 Sparse Block Attention

**BlockBERT** (Qiu et al., 2020). Block-sparse attention patterns.

**Sparse Transformers** (Child et al., 2019). Fixed sparsity patterns.

**NASC's block branch** selects blocks dynamically via mean Q/K scores.

## 2.3 Random Fourier Features

**Performer** (Choromanski et al., 2021). Positive orthogonal random features 
for softmax approximation.

**RFF in NASC** uses concatenated cos/sin features:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)]
\tag{2.1}
$$

where $\Omega \in \mathbb{R}^{d \times F}$ is orthogonally initialized.

## 2.4 Softpick / ReLU Attention

**ReLU attention** (Katharopoulos et al., 2020). Uses $\text{relu}(\cdot)$ as 
a kernel feature.

**Softpick** (Alman & Song, 2024). A sparse alternative to softmax with 
reduced numerical overhead.

**NASC's `softpick`** is ReLU-based with a batch-level softmax fallback.

## 2.5 Position of Our Work

| Axis | Prior work | NASC |
|---|---|---|
| **Branches** | 2 (local + global) | **3 (local + sparse + RFF)** |
| **Gate** | Learned per-token | **Learned per-batch** |
| **Attention** | Softmax | **Softpick (ReLU + fallback)** |
| **Causal path** | Full | **Prefix-sum for RFF** |

**Key distinction.** NASC's three-branch structure with batch-level gating 
is unusual. The gate simplification reduces per-token adaptivity but 
avoids gate collapse.

---

## Comparison Table

| Method | Branches | Gate | Sparse | RFF | Causal |
|---|---|---|---|---|---|
| Longformer | 2 | Fixed | Fixed | No | Yes |
| BigBird | 3 | Fixed | Random | No | Yes |
| Performer | 1 | N/A | No | Yes | Yes |
| **NASC** | **3** | **Batch-softmax** | **Block** | **Yes** | **Yes** |

---