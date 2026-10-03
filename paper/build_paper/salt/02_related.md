<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Adaptation

### 2.1.1 LoRA

LoRA (Hu et al., 2021) parameterizes updates as $\Delta W = \frac{\alpha}{r} BA$. 
Base weights are frozen.

**Limitation:** Total parameters unchanged.

### 2.1.2 AdaLoRA

AdaLoRA (Zhang et al., 2023) adaptively allocates rank across layers.

**Limitation:** Still freezes full base weights.

### 2.1.3 DoRA

DoRA (Liu et al., 2024) decomposes weights into magnitude and direction.

**Limitation:** Magnitude is scalar per output; still stores full W.

## 2.2 Low-Rank Weights

### 2.2.1 SVD Compression

Classical SVD compression (Denton et al., 2014) replaces W with $U_r \Sigma_r V_r^\top$.

**Limitation:** No adaptation after compression; accuracy may drop.

### 2.2.2 Tensor Decomposition

Tensor train (Novikov et al., 2015) and Tucker decomposition.

**Limitation:** Complex to train; not pretrained-friendly.

### 2.2.3 ALBERT Factorized Embeddings

ALBERT (Lan et al., 2020) factorizes the embedding matrix.

**Limitation:** Only embeddings, not attention weights.

## 2.3 Hybrid Methods

### 2.3.1 PiSSA

PiSSA (Meng et al., 2024) initializes LoRA factors from the SVD of W.

**Difference from SALT:** PiSSA still stores the full W; SALT stores only 
$U_r, \Sigma_r, V_r$.

### 2.3.2 OLoRA

OLoRA (Büyükakyüz, 2024) initializes LoRA with orthogonal matrices.

**Difference:** OLoRA does not reduce total parameters.

### 2.3.3 GaLore

GaLore (Zhao et al., 2024) projects gradients into a low-rank subspace.

**Difference:** GaLore reduces gradient memory, not weight storage.

## 2.4 Position of Our Work

| Axis | LoRA | PiSSA | SVD Comp | SALT |
|---|---|---|---|---|
| Total params reduced? | No | No | Yes | **Yes** |
| Trainable adaptation? | Yes | Yes | No | **Yes** |
| Pretrained init? | Base only | SVD init | Base | **SVD + γ + AB** |
| Inference speedup? | No | No | Yes (if fused) | No (naive) |

**Key distinction.** SALT reduces **total** parameters AND supports 
adaptation AND starts from the pretrained state. Prior work achieves at most 
two of these three.

---

## Comparison Table

| Method | Total params | Trainable | Init from W | Inference memory |
|---|---|---|---|---|
| Full FT | 100% | 100% | W | 100% |
| LoRA | 100% | ~1% | W | 100% |
| PiSSA | 100% | ~1% | SVD(W) | 100% |
| SVD Comp | ~10% | 0% | SVD(W) | ~10% (fused) |
| **SALT** | **~10%** | **~9%** | **SVD(W)** | **~10% (fused)** |

---