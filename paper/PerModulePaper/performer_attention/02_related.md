<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Kernel Attention

**Linear Transformer** (Katharopoulos et al., 2020). Uses $\phi(x) = \text{elu}(x) + 1$. 
Simple, no theoretical bound.

**Performer** (Choromanski et al., 2021). FAVOR+ with positive orthogonal 
random features. Theoretical guarantees.

**Random Feature Attention** (Peng et al., 2021). RFF without orthogonality.

**Nyströmformer** (Xiong et al., 2021). Nyström landmarks.

**Cosformer** (Qin et al., 2022). Cosine reweighting.

## 2.2 Random Features

**Rahimi & Recht (2007).** Random Fourier Features (RFF) approximate 
shift-invariant kernels.

**Choromanski et al. (2018).** The geometry of random features.

**FAVOR+ (2021).** Positive orthogonal random features with variance 
reduction.

## 2.3 Efficient Attention

**Longformer** (Beltagy et al., 2020). Sparse.

**BigBird** (Zaheer et al., 2020). Sparse + random.

**FlashAttention** (Dao et al., 2022). Exact, tiled.

**Linformer** (Wang et al., 2020). Low-rank.

## 2.4 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| **Method** | FAVOR+ (published) | Port to pretrained |
| **Weight transfer** | Rare | Exact, four architectures |
| **Causal scan** | Yes | Yes (chunk=64) |
| **Numerical stability** | Standard | Max-subtraction + norm-sub |
| **Bug disclosure** | Rare | Two discrepancies documented |

**Key distinction.** Unlike the original Performer paper (which trains from 
scratch), we focus on **weight-preserving port** to pretrained models. This 
requires careful handling of the feature map and the prefix scan.

---

## Comparison Table

| Method | Kernel | Theory | Pretrained transfer | Causal |
|---|---|---|---|---|
| Linear Transformer | elu+1 | No | Rare | Yes |
| Performer | FAVOR+ | Yes | **Ours** | Yes |
| RFA | RFF | Partial | Rare | Yes |
| Nyströmformer | Landmark | Yes | Rare | Yes |
| Cosformer | cos/sin | Yes | Yes | Yes |

---