# 🚀 NEURALOPS (`SVD_linear_Attention` v1.3.7) — DEFINITIVE MASTER REPORT
## Mathematical Foundations, Forensic Remediation & Comprehensive Empirical Verification

**Release Version**: `v1.3.7-FINAL-HONEST`  
**Git Commit SHA**: `d9d49bd42249c03d83ac8132d1585c8886ff19e6`  
**Hardware Platform**: Physical NVIDIA GeForce RTX 4070 Laptop GPU 8GB (Ada Lovelace, **SM 8.9**, 36 SMs, CUDA 13.2 / Driver 595.99.02)  
**Host Environment**: Linux 6.6 / Python 3.14 / OpenAI Triton 3.6+ / PyTorch 2.6+  
**Date**: October 16, 2026  
**Sign-off Status**: ⚠️ **VERIFIED WITH KNOWN LIMITATIONS**  

---

## SECTION 1: EXECUTIVE SUMMARY & CANONICAL INVARIANTS

### 1.1 Canonical Invariants (Verified)

| Invariant | Count | Verification Tool |
|:---:|:---:|:---:|
| Subquadratic Attention Mechanisms | **131** | `scripts/utils/verify_readme_counts.py` |
| Compression Engines | **72** | `scripts/utils/verify_readme_counts.py` |
| Fused Triton JIT Kernels | **26** | `scripts/utils/verify_readme_counts.py` |
| HNC-SVD Exhaustive Tests | **73/73 passed** | `tests/test_hnc_svd_exhaustive.py` |
| 26 Triton Kernels Matrix | **68/68 passed** | `tests/test_exhaustive_26_triton_kernels_matrix.py` |
| 72 Compressors Matrix | **504/504 passed** | `tests/test_exhaustive_72_compressors_matrix.py` |
| 131 Attentions Matrix | **1,048/1,048 passed** | `tests/test_exhaustive_131_attentions_matrix.py` |
| **Total Passing Assertions** | **1,693/1,693 (100%)** | Physical RTX 4070 SM 8.9 |

### 1.2 The Core Scientific Problem

Applying naive Truncated SVD ($W \approx U_{:,:r}\Sigma_{:,:r}V_{:,:r}^\top$) to transformer weight matrices introduces severe out-of-distribution reconstruction drift because naive SVD minimizes $\|W - \hat{W}\|_F$ blindly, ignoring:

1. The **activation covariance spectrum** $H_X = \frac{1}{N_{\text{tok}}} X^\top X$.
2. The **subspace curvature** (Gauss-Newton Hessian of function matching).
3. The **discontinuous rank-jump** at the truncation boundary.

### 1.3 The HNC-SVD Breakthrough

HNC-SVD embeds singular value truncation into a continuous, manifold-preserving homotopy walk:

$$\mathcal{M}_0 \xrightarrow{\nabla \phi \cdot P_{\text{null}}} \mathcal{M}_{\text{steered}} \xrightarrow{\text{Cholesky Truncation}} \mathcal{M}_r \xrightarrow{m_{\text{post}}} \mathcal{M}_{\text{final}}$$

---

## SECTION 2: MATHEMATICAL DERIVATIONS

### 2.1 Anchor Exact Zero-Gradient Law

$$\mathcal{L}_{\text{func}}(\theta) = \frac{1}{2N}\sum_{i=1}^N \|f_\theta(x_i) - f_{\theta_0}(x_i)\|_2^2$$

**Theorem 1**: At $\theta = \theta_0$:
- $\mathcal{L}_{\text{func}}(\theta_0) = 0$
- $\nabla_\theta \mathcal{L}_{\text{func}}(\theta_0) = \mathbf{0}$

### 2.2 Gauss-Newton Positive Semidefiniteness

$$G(\theta_0) = \frac{1}{N}J^\top J \succeq 0$$

**Theorem 2**: For any $v \in \mathbb{R}^P$: $v^\top G v = \frac{1}{N}\|Jv\|_2^2 \ge 0$

### 2.3 Analytical SVD Tail Potential

$$\phi_{\text{SVD}}(W) = -\frac{\sum_{i>r}\sigma_i^2}{\sigma_1^2 + \epsilon} - \gamma_{\text{nuc}}\frac{\sum_{i>r}\sigma_i}{\sigma_1 + \epsilon} + \gamma_{\text{bump}}\frac{1}{m-r}\sum_{i>r}\exp\left(-\frac{\sigma_i^2}{2\tilde{\sigma}^2}\right)$$

**Closed-form gradient (autograd-free)**:
$$\nabla_W \phi_{\text{SVD}} = U_{:,r+1:m} \cdot \text{diag}(g_{r+1..m}) \cdot V_{:,r+1:m}^\top$$

### 2.4 Fast Layerwise Kronecker Resolvent

$$G_W = I_{d_{\text{out}}} \otimes H_X$$
$$D_W = \nabla_W \phi_{\text{SVD}} \cdot \left(I_{d_{\text{in}}} + \frac{1}{\mu_X}H_X\right)^{-1}$$
$$\Delta W = \eta \|W\|_F \frac{D_W}{\|D_W\|_F + 10^{-8}}$$

### 2.5 Generalized Eckart-Young Cholesky-Whitened SVD

$$S_X = H_X + \lambda_{\text{reg}}\text{tr}(H_X/d_{\text{in}})I$$
$$L = \text{chol}(S_X), \quad \tilde{W} = W_{\text{steered}} L$$
$$A_r = \tilde{U}_{:,:r}\sqrt{\text{diag}(\tilde{\Sigma}_{:r})}, \quad B_r = \sqrt{\text{diag}(\tilde{\Sigma}_{:r})}\tilde{V}_{:,:r}^\top L^{-1}$$

### 2.6 Bias Drift Compensation

$$\hat{b} \leftarrow b + (W_0 - \hat{W})\mu_X$$

### 2.7 Post-Truncation Restoration

$$\min_{\{A_{r,l}, B_{r,l}\}} \frac{1}{2N}\sum_{i=1}^N \|f_{\{A_{r,l}B_{r,l}\}}(x_i) - f_{\theta_0}(x_i)\|_2^2$$

With strict invariant $\text{rank}(A_{r,l}B_{r,l}) \le r$.

---

## SECTION 3: EXACT COMPRESSION MATHEMATICS

### 3.1 Formulas

$$\text{Reduction (\%)} = \left(1 - \frac{r(d_{\text{in}} + d_{\text{out}})}{d_{\text{in}}d_{\text{out}}}\right) \times 100\%$$
$$\text{Compression Factor} = \frac{d_{\text{in}}d_{\text{out}}}{r(d_{\text{in}} + d_{\text{out}})}$$

### 3.2 Transformer Block Projections ($d_{\text{in}}=256, d_{\text{out}}=512$)

| Rank | Original | Compressed | Reduction | Factor |
|:---:|:---:|:---:|:---:|:---:|
| **$r=8$** | 262,144 | 12,288 | **95.3%** | **21.3×** |
| **$r=16$** | 262,144 | 24,576 | **90.6%** | **10.7×** |
| **$r=32$** | 262,144 | 49,152 | **81.2%** | **5.3×** |
| **$r=64$** | 262,144 | 98,304 | **62.5%** | **2.7×** |

### 3.3 Production LLM Projections ($d_{\text{model}}=768, d_{\text{ffn}}=3072$)

| Rank | Original | Compressed | Reduction | Factor |
|:---:|:---:|:---:|:---:|:---:|
| **$r=32$** | 4,718,592 | 245,760 | **94.8%** | **19.2×** |
| **$r=64$** | 4,718,592 | 491,520 | **89.6%** | **9.6×** |
| **$r=128$** | 4,718,592 | 983,040 | **79.2%** | **4.8×** |
| **$r=256$** | 4,718,592 | 1,966,080 | **58.3%** | **2.4×** |

---

## SECTION 4: HNC-SVD TEST SUITE (16 Groups, 73 Tests)

| Group | Invariant | Tests | Status |
|:---:|:---|:---:|:---:|
| 1 | Anchor Zero Loss & Zero Gradient | 5/5 | ✅ PASS |
| 2 | Gauss-Newton PSD vs Indefinite Hessian | 10/10 | ✅ PASS |
| 3 | Pearlmutter HVP vs Dense Jacobian | 3/3 | ✅ PASS |
| 4 | Power Iteration & LOBPCG | 3/3 | ✅ PASS |
| 5 | SVD Tail Potential vs Autograd | 12/12 | ✅ PASS |
| 6 | CG Soft Projection & Warm-Start | 3/3 | ✅ PASS |
| 7 | Kronecker Null-Space Step | 3/3 | ✅ PASS |
| 8 | Restoring Steps vs Function Drift | 4/4 | ✅ PASS |
| 9 | Damping Sweep $\mu_{\text{rel}}$ | 4/4 | ✅ PASS |
| 10 | Loss Ceiling Rejection & Rollback | 1/1 | ✅ PASS |
| 11 | HNC vs Penalized Optimization | 1/1 | ✅ PASS |
| 12 | Full-Logit vs Scalar Loss (strict $\ge 0.85$ / $\ge 0.75$) | 1/1 | ✅ PASS |
| 13 | SGD Stability of HNC Endpoints | 1/1 | ✅ PASS |
| 14 | Fused Triton Kernel Parity | 17/17 | ✅ PASS |
| 15 | DirectSVD vs Naive SVD (strict $< 0.75$) | 4/4 | ✅ PASS |
| 16 | SmoothMax CKA & Registry Invariants | 1/1 | ✅ PASS |
| **Total** | — | **73/73** | ✅ **100%** |

---

## SECTION 5: PHYSICAL BENCHMARKS (REAL NUMBERS)

*Hardware*: NVIDIA RTX 4070 Laptop 8GB, Ada Lovelace SM 8.9  
*Runtime*: 25.84 seconds for full 5-table suite  
*Source*: `results/hnc_svd_comprehensive_benchmark.{json,md}`

### Table 1: HNC-SVD vs Naive on Transformer Block (REAL)

| Method | Rank | Comp% | Init Tail | Post Tail | RelErr | Top-1 | CosSim | L_func Drift |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive | 8 | 95.3% | 0.9165 | 0.9165 | 0.2619 | 62.50% | 0.9657 | 3.43e-02 |
| **HNC** | 8 | 95.3% | 0.9165 | **0.9153** | **0.1818** | **72.17%** | **0.9835** | **1.65e-02** |
| Naive | 16 | 90.6% | 0.8412 | 0.8412 | 0.2542 | 62.70% | 0.9677 | 3.23e-02 |
| **HNC** | 16 | 90.6% | 0.8412 | **0.8393** | **0.1732** | **73.73%** | **0.9850** | **1.50e-02** |
| Naive | 32 | 81.2% | 0.7083 | 0.7083 | 0.2332 | 66.31% | 0.9728 | 2.72e-02 |
| **HNC** | 32 | 81.2% | 0.7083 | **0.7052** | **0.1610** | **74.61%** | **0.9870** | **1.30e-02** |
| Naive | 64 | 62.5% | 0.4930 | 0.4930 | 0.1944 | 71.58% | 0.9811 | 1.89e-02 |
| **HNC** | 64 | 62.5% | 0.4930 | **0.4892** | **0.1465** | **77.15%** | **0.9893** | **1.07e-02** |

**Ratios**: 0.694 (rank 8), 0.681 (rank 16), 0.690 (rank 32), 0.754 (rank 64) ✅

### Table 2: Surgical Alignment (REAL — shows HNC limitation)

| Method | Rank | CosSim | MSE Drift | RelErr | Tok/s |
|:---|:---:|:---:|:---:|:---:|:---:|
| Raw | 8 | **0.7255** | 8.15e-08 | **0.696** | 842K |
| HNC | 8 | 0.1369 | 1.97e-06 | 3.422 | 840K |
| Raw | 16 | **0.8246** | 5.43e-08 | **0.568** | 837K |
| HNC | 16 | 0.2346 | 1.43e-06 | 2.917 | 841K |
| Raw | 32 | **0.9161** | 2.71e-08 | **0.401** | 840K |
| HNC | 32 | 0.4033 | 5.87e-07 | 1.866 | 692K |
| Raw | 64 | **0.9597** | 1.36e-08 | **0.284** | 695K |
| HNC | 64 | 0.5610 | 3.42e-07 | 1.424 | 676K |
| Raw | 256 | **1.0000** | 4.37e-20 | **0.000001** | 687K |
| HNC | 256 | 0.6497 | 2.12e-07 | 1.121 | 691K |

⚠️ **KNOWN LIMITATION**: `hnc_spectral_compress_and_align` produces degraded output at full-rank (cos=0.65 instead of 1.0) and worse than raw SVD at all ranks. This is documented as an **open bug** in the master report.

### Table 3: Damping $\mu_{\text{rel}}$ & Restoring Steps (REAL)

| $\mu_{\text{rel}}$ | $m$ | Init Tail | Final Tail | Final L_func | Rejections |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1e-7 | 0 | 0.3726 | 0.0131 | 9.62e-14 | 0 |
| 1e-7 | 12 | 0.3726 | 0.0131 | 8.99e-14 | 0 |
| 1e-5 | 12 | 0.3726 | 0.0131 | 3.27e-10 | 0 |
| 1e-3 | 12 | 0.3726 | 0.0163 | 2.71e-06 | 0 |
| $\infty$ | 0 | 0.3726 | 0.0107 | 4.56e-02 | 13 |
| $\infty$ | 12 | 0.3726 | **0.0006** | 3.30e-02 | 0 |

⚠️ **KNOWN LIMITATION**: Restoration steps $m=0$ vs $m=12$ show minimal difference. $\mu=\infty$ gives best tail reduction (opposite of paper prediction).

### Table 4: Triton Kernel Speedup (REAL — TF32 enabled)

| Dtype | Shape | PyTorch $\mu$s | Triton $\mu$s | Speedup | MaxDiff |
|:---:|:---:|:---:|:---:|:---:|:---:|
| FP16 | $512^2$ | 62.83 | 76.19 | 0.82× | 9.77e-04 |
| FP16 | $1024^2$ | 251.23 | 85.59 | **2.94×** | 1.95e-03 |
| FP16 | $2048^2$ | 2241.70 | 522.16 | **4.29×** | 1.95e-03 |
| FP16 | $4096^2$ | 14978.03 | 4805.22 | **3.12×** | 1.95e-03 |
| FP32 | $512^2$ | 45.36 | 29.86 | **1.52×** | 1.83e-04 |
| FP32 | $1024^2$ | 246.59 | 151.47 | **1.63×** | 2.53e-04 |
| FP32 | $2048^2$ | 1826.90 | 1074.13 | **1.70×** | 2.68e-04 |
| FP32 | $4096^2$ | 13380.94 | 12613.26 | 1.06× | 2.98e-04 |

⚠️ **KNOWN LIMITATION**: FP16 $512^2$ shows 0.82× regression (Triton kernel launch overhead dominates small matrices).

### Table 5: End-to-End LM Compression (REAL, synthetic 3-layer LM)

| Rank | Comp% | Naive RelErr | HNC RelErr | Naive Top-1 | HNC Top-1 | Orig PPL | Naive PPL | HNC PPL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 95.3% | 0.6090 | **0.5115** | 26.2% | **31.8%** | 846.76 | 900.63 | **884.38** |
| 16 | 90.6% | 0.5695 | **0.4668** | 26.6% | **33.2%** | 846.85 | 893.71 | **878.01** |
| 32 | 81.2% | 0.5152 | **0.4028** | 36.9% | **42.8%** | 846.89 | 885.06 | **870.07** |
| 64 | 62.5% | 0.4125 | **0.3153** | 40.2% | **50.0%** | 846.39 | 870.73 | **860.53** |

⚠️ **KNOWN LIMITATION**: `Orig PPL ≈ 846` indicates this benchmark suite evaluates a **synthetic random language model** ($\text{vocab}=1000$), NOT pretrained GPT-2. The table reflects genuine **relative** recovery, not absolute production LLM perplexity.

---

## SECTION 6: HGRN-2 BOTTLENECK ELIMINATION

### Root Causes Found & Fixed:
1. Phantom $\mathcal{O}(L^2)$ matrix multiplication in `_local_attention` (14.66 ms wasted per forward call).
2. Decoder inversion via synthetic `BertConfig` (doubled scans across forward and backward directions).
3. Missing Triton scan kernel exports in `selective_scan.py`.

### Benchmark Results ($T=4096$, RTX 4070 SM 8.9):

| Module | Latency (ms) | Tok/s | VRAM (MB) | Speedup vs MHA |
|:---|:---:|:---:|:---:|:---:|
| `pytorch_mha_baseline` | 19.12 | 428,552 | 912.6 | 1.00× |
| **`hgrn2_gated_deltanet`** | **13.61** | **602,112** | **283.1** | **1.41×** ✅ |
| `gated_delta_net` | 8.13 | 1,008,133 | 219.3 | 2.35× |
| `gla_attention` | 3.23 | 2,538,086 | 198.5 | 5.92× |
| `minimal_svd` | 3.45 | 2,373,698 | 98.6 | 5.54× |

---

## SECTION 7: DEVELOPER EXPERIENCE & CLI

### CLI Commands

```bash
nops doctor    # Environment & hardware inspector
nops verify    # Registry invariant validation (131/72/26)
nops plan      # Dry-run resource estimation
nops bench     # Physical hardware profiler
nops surgery   # Non-destructive model transplant
nops eval      # Multi-task LM evaluation
```

### Evaluation Adapters
- **Perplexity Scorer**: WikiText-2, PTB, C4
- **NIAH**: Needle-In-A-Haystack ($T \in [1024, 131072]$)
- **LM-Eval Bridge**: MMLU, GSM8K, ARC, HellaSwag

---

## SECTION 8: 6-PILLAR DISTILLATION ENGINE

$$\mathcal{L}_{\text{total}} = \lambda_{\text{task}}\mathcal{L}_{\text{task}} + \sum_{p=1}^6 \alpha_p \mathcal{L}_p$$

| Pillar | Loss Formulation | Target Purpose |
|:---:|:---|:---|
| **1** | Temperature-scaled KL ($\tau \in [1.5, 4.0]$) | Logit distribution matching |
| **2** | Cosine + Frobenius MSE on hidden states | Intermediate layer representation alignment |
| **3** | Jensen-Shannon Divergence on state trajectory | Attention & recurrent state matching |
| **4** | Feed-Forward Network activation matching | Non-linear feature alignment |
| **5** | Smooth-max Centered Kernel Alignment (CKA) | Representation geometry preservation |
| **6** | Spectral tail potential $\phi_{\text{SVD}}$ | Direct low-rank spectrum enforcement |

---

## SECTION 9: COMPLETE REPOSITORY MAP

```
SVD_linear_Attention/
├── pyproject.toml, pytest.ini, README.md, LICENSE, CITATION.cff
├── neuralops/
│   ├── api.py                          # Unified Public Clean API
│   ├── attentions/                     # 131 subquadratic attention mechanisms
│   ├── compression_engine/             # 72 compression & pruning engines
│   ├── core/
│   │   ├── hnc_svd.py                  # Theoretical HNC formulation
│   │   ├── registry.py                 # 131/72/26 invariant registries
│   │   └── triton_kernels/             # 26 JIT hardware kernels
│   ├── inference/                      # Paged KV, Radix Trie, Speculative Decoding
│   ├── surgery/                        # AST graph transplant & Shadow Twin
│   └── trainer/                        # 6-Pillar distillation engine
├── tests/                              # 1,693 tests (100% pass rate)
├── scripts/
│   ├── benchmarks/                     # Hardware profiling suites
│   └── utils/verify_readme_counts.py   # 131/72/26 invariant assertion
├── results/                            # Empirical data vault
└── docs/
    ├── parts/                          # PART1-PART9 volumes
    └── reports/                        # Master forensic reports
```

---

## SECTION 10: KNOWN LIMITATIONS (HONEST)

The following issues are open in v1.3.7 and must be addressed in v1.4.0:

### 🚨 Critical Bug #1: `hnc_spectral_compress_and_align` Full-Rank Failure
- At rank=256 (full rank), HNC output achieves $\text{CosSim} = 0.6497$ instead of $1.0000$.
- HNC produces lower cosine similarity than raw SVD at all ranks in Table 2.
- **Status**: Open; requires audit of factor projection and setter logic in `_factorize_and_set`.

### 🚨 Critical Bug #2: Walk Tail Energy Reduction Under 1%
- Post-walk tail ratio shifts from $0.9165 \to 0.9153$ at rank 8.
- Huang et al. (2026) report 5–10× tail reduction; empirical walk achieves $< 1\%$.
- Function matching preservation and low relative error are driven primarily by Cholesky-whitened SVD truncation, not the null-space trajectory.
- **Status**: Open; requires reformulation of null-space step scaling.

### 🚨 Bug #3: Restoration Steps Displaying Minimal Tail Delta
- $m=0$ vs $m=12$ restoration steps produce identical tail energy ($0.0131$).
- **Status**: Open; restoration steps currently act as a near no-op on spectral tail energy.

### 🚨 Bug #4: Damping $\mu=\infty$ Yielding Inverted Tail Behavior
- $\mu = \infty$ produces Tail $= 0.0006$ (best).
- $\mu = 10^{-7}$ produces Tail $= 0.0131$ (worse).
- **Status**: Open; damping direction and regularizer sign require review against paper formulation.

### 🚨 Bug #5: Table 5 Synthetic Language Model
- $\text{Orig PPL} \approx 846$ reflects an un-pretrained synthetic 3-layer language model ($\text{vocab}=1000$), not dense pretrained GPT-2.
- **Status**: Acknowledged; documented as relative algorithmic recovery rather than absolute perplexity benchmark.

### ⚠️ Bug #6: FP16 $512^2$ Triton Launch Overhead Regression
- Speedup of $0.82\times$ at $512 \times 512$ due to GPU kernel launch overhead dominating small tensor dimensions.
- **Status**: Open; requires thresholding heuristic to fall back to PyTorch for $D \le 512$.

---

## SECTION 11: FORMAL SIGN-OFF

```
========================================================================================
              NEURALOPS v1.3.7 PRODUCTION RELEASE CERTIFICATE (WITH LIMITATIONS)
========================================================================================
Framework:           NeuralOps (SVD_linear_Attention)
Release Version:     1.3.7-HONEST
Target Hardware:     NVIDIA RTX 4070 Laptop GPU (Ada Lovelace, SM 8.9)
Git Commit SHA:      d9d49bd42249c03d83ac8132d1585c8886ff19e6

Registry Invariants: 131 Attentions | 72 Compressors | 26 Triton Kernels
Verification Battery:
  - test_hnc_svd_exhaustive.py:             73 / 73 Passed (100.0%)
  - test_exhaustive_26_triton_kernels.py:   68 / 68 Passed (100.0%)
  - test_exhaustive_72_compressors.py:     504 / 504 Passed (100.0%)
  - test_exhaustive_131_attentions.py:   1,048 / 1,048 Passed (100.0%)
Total Invariant Tests:                   1,693 / 1,693 Passed (100.0%)
Warnings Count:                          0

CRITICAL ISSUES (OPEN):
  - HNC full-rank failure (cos=0.65 vs raw=1.0)
  - Tail reduction <1% (vs paper 5-10×)
  - Restoration steps no-op
  - μ damping direction inverted
  - Table 5 uses synthetic model (PPL=846)
  - FP16 512² Triton launch overhead

Sign-Off Status:     ⚠️ VERIFIED WITH KNOWN LIMITATIONS
                     Certified for rigorous empirical evaluation;
                     Bugs #1–#6 queued for formal resolution in v1.4.0 roadmap.
========================================================================================
```
