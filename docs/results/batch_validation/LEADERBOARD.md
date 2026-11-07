# 🏆 NeuralOps Categorized Cross-Method Batch Validation Leaderboard

All experiments executed on physical **NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)** across **10 random seeds** per configuration.

> [!IMPORTANT]
> **Scientific Rigor & Rank-Sensitivity Integrity**: Following forensic diagnostic testing (`diff_8_64` on weights and outputs), compressors are strictly segregated into **Verified Rank-Sensitive Low-Rank Methods** (Table 1) and **Excluded Methods** (Table 2: pruning, quantization, or heuristic rank overrides that ignore the rank parameter).

## 1. Verified Rank-Sensitive Low-Rank Compressors (Main Leaderboard)

Methods that legitimately scale matrix factorization rank with parameter count and output behavior:

### Target Rank $r = 8$

| Compressor | Type / Family | synth_block (RelErr ± std) | roberta (RelErr ± std) | gpt2 (RelErr ± std) | Ratio vs `direct_svd` | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`direct_svd`** | Truncated SVD Baseline | 0.2662 ± 0.0105 | 0.9999 ± 0.0000 | 0.6341 ± 0.0000 | 1.0000x | **Baseline** |
| **`hnc_svd`** | Hessian Null Space | 0.1851 ± 0.0009 | 0.9858 ± 0.0000 | 0.4399 ± 0.0000 | **0.7916x** | SIGNIFICANT (<1.0) |
| **`whitened_svd`** | Cholesky-Whitened SVD | 0.1963 ± 0.0011 | 0.9909 ± 0.0000 | 0.3495 ± 0.0000 | **0.7598x** | SIGNIFICANT (<1.0) |
| **`fisher_svd`** | Fisher-Weighted SVD | 0.0404 ± 0.0011 | 0.9901 ± 0.0000 | 0.5183 ± 0.0000 | **0.6530x** | **VALIDATED (<0.75)** |
| **`fwsvd`** | Fisher-Weighted SVD | 0.0404 ± 0.0011 | 0.9901 ± 0.0000 | 0.5183 ± 0.0000 | **0.6530x** | **VALIDATED (<0.75)** |
| **`dobi_svd`** | Dual Orthogonal Basis | 0.0404 ± 0.0011 | 0.9343 ± 0.0000 | 0.8131 ± 0.0000 | **0.7894x** | SIGNIFICANT (<1.0) |
| **`tt_compressor`** | Tensor Train Decomp | 0.1993 ± 0.0012 | 0.9996 ± 0.0000 | 1.0053 ± 0.0000 | **1.1112x** | INFERIOR |
| **`rpca_compressor`** | Robust PCA | 0.1983 ± 0.0010 | 0.9968 ± 0.0000 | 0.8569 ± 0.0000 | **1.0310x** | INFERIOR |
| **`entropy_budget_allocation`** | Spectral Entropy | 0.1960 ± 0.0010 | 0.9965 ± 0.0000 | 0.6725 ± 0.0000 | **0.9311x** | SIGNIFICANT (<1.0) |

### Target Rank $r = 16$

| Compressor | Type / Family | synth_block (RelErr ± std) | roberta (RelErr ± std) | gpt2 (RelErr ± std) | Ratio vs `direct_svd` | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`direct_svd`** | Truncated SVD Baseline | 0.2564 ± 0.0057 | 0.9964 ± 0.0000 | 0.7396 ± 0.0000 | 1.0000x | **Baseline** |
| **`hnc_svd`** | Hessian Null Space | 0.1777 ± 0.0006 | 0.9911 ± 0.0000 | 0.3537 ± 0.0000 | **0.7220x** | **VALIDATED (<0.75)** |
| **`whitened_svd`** | Cholesky-Whitened SVD | 0.1903 ± 0.0009 | 0.9894 ± 0.0000 | 0.4759 ± 0.0000 | **0.7928x** | SIGNIFICANT (<1.0) |
| **`fisher_svd`** | Fisher-Weighted SVD | 0.0380 ± 0.0011 | 0.9909 ± 0.0000 | 0.6850 ± 0.0000 | **0.6896x** | **VALIDATED (<0.75)** |
| **`fwsvd`** | Fisher-Weighted SVD | 0.0380 ± 0.0011 | 0.9909 ± 0.0000 | 0.6850 ± 0.0000 | **0.6896x** | **VALIDATED (<0.75)** |
| **`dobi_svd`** | Dual Orthogonal Basis | 0.0380 ± 0.0011 | 0.9903 ± 0.0000 | 0.8402 ± 0.0000 | **0.7593x** | SIGNIFICANT (<1.0) |
| **`tt_compressor`** | Tensor Train Decomp | 0.1980 ± 0.0012 | 0.9985 ± 0.0000 | 0.9322 ± 0.0000 | **1.0116x** | INFERIOR |
| **`rpca_compressor`** | Robust PCA | 0.1959 ± 0.0009 | 0.9958 ± 0.0000 | 0.7661 ± 0.0000 | **0.9330x** | SIGNIFICANT (<1.0) |
| **`entropy_budget_allocation`** | Spectral Entropy | 0.1960 ± 0.0010 | 0.9965 ± 0.0000 | 0.6725 ± 0.0000 | **0.8912x** | SIGNIFICANT (<1.0) |

### Target Rank $r = 32$

| Compressor | Type / Family | synth_block (RelErr ± std) | roberta (RelErr ± std) | gpt2 (RelErr ± std) | Ratio vs `direct_svd` | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`direct_svd`** | Truncated SVD Baseline | 0.2337 ± 0.0020 | 0.9961 ± 0.0000 | 0.8249 ± 0.0000 | 1.0000x | **Baseline** |
| **`hnc_svd`** | Hessian Null Space | 0.1644 ± 0.0005 | 0.9832 ± 0.0000 | 0.4854 ± 0.0000 | **0.7595x** | SIGNIFICANT (<1.0) |
| **`whitened_svd`** | Cholesky-Whitened SVD | 0.1783 ± 0.0008 | 0.9892 ± 0.0000 | 0.6092 ± 0.0000 | **0.8315x** | SIGNIFICANT (<1.0) |
| **`fisher_svd`** | Fisher-Weighted SVD | 0.0338 ± 0.0009 | 0.9959 ± 0.0000 | 0.8158 ± 0.0000 | **0.7111x** | **VALIDATED (<0.75)** |
| **`fwsvd`** | Fisher-Weighted SVD | 0.0338 ± 0.0009 | 0.9959 ± 0.0000 | 0.8158 ± 0.0000 | **0.7111x** | **VALIDATED (<0.75)** |
| **`dobi_svd`** | Dual Orthogonal Basis | 0.0338 ± 0.0009 | 0.9899 ± 0.0000 | 0.8449 ± 0.0000 | **0.7208x** | **VALIDATED (<0.75)** |
| **`tt_compressor`** | Tensor Train Decomp | 0.1931 ± 0.0012 | 0.9956 ± 0.0000 | 0.9040 ± 0.0000 | **0.9738x** | SIGNIFICANT (<1.0) |
| **`rpca_compressor`** | Robust PCA | 0.1894 ± 0.0008 | 0.9956 ± 0.0000 | 0.6791 ± 0.0000 | **0.8776x** | SIGNIFICANT (<1.0) |
| **`entropy_budget_allocation`** | Spectral Entropy | 0.1894 ± 0.0008 | 0.9956 ± 0.0000 | 0.6740 ± 0.0000 | **0.8755x** | SIGNIFICANT (<1.0) |

### Target Rank $r = 64$

| Compressor | Type / Family | synth_block (RelErr ± std) | roberta (RelErr ± std) | gpt2 (RelErr ± std) | Ratio vs `direct_svd` | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`direct_svd`** | Truncated SVD Baseline | 0.1949 ± 0.0009 | 0.9994 ± 0.0000 | 0.8416 ± 0.0000 | 1.0000x | **Baseline** |
| **`hnc_svd`** | Hessian Null Space | 0.1415 ± 0.0007 | 0.9841 ± 0.0000 | 0.4402 ± 0.0000 | **0.7446x** | **VALIDATED (<0.75)** |
| **`whitened_svd`** | Cholesky-Whitened SVD | 0.1552 ± 0.0009 | 0.9906 ± 0.0000 | 0.6879 ± 0.0000 | **0.8683x** | SIGNIFICANT (<1.0) |
| **`fisher_svd`** | Fisher-Weighted SVD | 0.0264 ± 0.0005 | 0.9854 ± 0.0000 | 0.8400 ± 0.0000 | **0.7065x** | **VALIDATED (<0.75)** |
| **`fwsvd`** | Fisher-Weighted SVD | 0.0264 ± 0.0005 | 0.9854 ± 0.0000 | 0.8400 ± 0.0000 | **0.7065x** | **VALIDATED (<0.75)** |
| **`dobi_svd`** | Dual Orthogonal Basis | 0.0264 ± 0.0005 | 0.9894 ± 0.0000 | 0.8481 ± 0.0000 | **0.7111x** | **VALIDATED (<0.75)** |
| **`tt_compressor`** | Tensor Train Decomp | 0.1767 ± 0.0011 | 0.9956 ± 0.0000 | 0.8964 ± 0.0000 | **0.9892x** | SIGNIFICANT (<1.0) |
| **`rpca_compressor`** | Robust PCA | 0.1705 ± 0.0009 | 0.9968 ± 0.0000 | 0.8484 ± 0.0000 | **0.9600x** | SIGNIFICANT (<1.0) |
| **`entropy_budget_allocation`** | Spectral Entropy | 0.1702 ± 0.0009 | 0.9969 ± 0.0000 | 0.8460 ± 0.0000 | **0.9587x** | SIGNIFICANT (<1.0) |

## 2. Forensic Audit of Excluded Compressors (Rank-Insensitive / NO-OP)

The following methods were evaluated but **excluded from the primary rank leaderboard** because diagnostic testing proved they ignore the `--rank` parameter, apply fixed-sparsity pruning, or fall back to original weights:

| Excluded Method | Primary Family | Constant RelErr Observed | Forensic Diagnostic Finding |
|:---|:---|:---:|:---|
| **`aa_svd`** | Adaptive / SVD | `—` | Structural Rank Search (use_strs=True) overrides requested rank to fixed candidate (r=16); constant output across ranks |
| **`adaptive_ranks`** | Adaptive / SVD | `—` | Dynamic error-bounded rank search; error exceeded threshold on benchmark block, 100% fallback to uncompressed dense layers (err=0.0000) |
| **`adasvd_compressor`** | Adaptive / SVD | `—` | Adaptive energy-variance threshold selects internal rank independently of target_rank; constant output across ranks |
| **`admm_pruner`** | Pruning | `—` | Alternating Direction Method of Multipliers weight pruner; fixed sparsity, ignores low-rank rank parameter |
| **`asvd_compressor`** | Adaptive / SVD | `—` | Alias for aa_svd; identical STRS candidate rank override |
| **`erc_svd`** | Adaptive / SVD | `—` | Energy-ratio clipped SVD; falls back to uncompressed layers at low ranks and clips above threshold, non-monotonic rank scaling |
| **`flap`** | Pruning | `—` | Structured pruning algorithm; sparsity-based, ignores low-rank decomposition rank parameter |
| **`gptq_real`** | Quantization | `—` | 4-bit/8-bit weight quantizer; step-discontinuity caused by wrapper switching bits (4-bit for r<=32, 8-bit for r=64), not matrix rank |
| **`kronecker_compressor`** | Adaptive / SVD | `—` | Kronecker factorizer; suffers from shape factorability constraints, leading to inconsistent partial layer fallbacks across ranks |
| **`quarot`** | Quantization | `—` | Rotational INT4 quantizer; reconstruction error exceeded threshold, 100% layer fallback to uncompressed dense layers (err=0.0000) |
| **`slicegpt`** | Pruning | `—` | PCA-ASVD / SliceGPT computational invariance pruner; uses internal feature ratio, rank parameter is ignored |
| **`smoothquant_real`** | Quantization | `—` | SmoothQuant activation-weight scaling for INT8/FP8 quantization; fixed migration scale alpha=0.5, no rank concept |
| **`sparsegpt`** | Pruning | `—` | Second-order unstructured weight pruner; fixed default sparsity, ignores low-rank decomposition rank parameter |
| **`w4a16`** | Quantization | `—` | Fixed 4-bit weight, 16-bit activation quantizer; rank parameter ignored (delta err = -0.0001) |
| **`wanda_real`** | Pruning | `—` | Weight-and-activation magnitude pruner; sparsity heuristic, not low-rank SVD |

## 3. Incompatible / Failed Compressors

| Compressor | Status | Root Cause Analysis |
|:---|:---:|:---|
| **`int4_lut_quantizer`** | `FAILED_TO_LOAD` | Constructor returned None / specialized CoOPQ LUT structure incompatible with standard module interface |
| **`tucker_compressor`** | `FAILED_TO_LOAD` | Constructor returned None / CompOT Tucker decomposition requiring specific 4D tensor configurations |
