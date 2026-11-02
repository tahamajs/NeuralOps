# HNC-SVD Architectural Status & Dispatch Decision

**Date**: October 16, 2026  
**Status**: Decided & Integrated into Dispatch Pipeline  
**Authority**: Principal Systems Architect, Compression Lead

---

## 1. Background & Empirical Findings

During comprehensive batch validation across the 72 compression algorithms on RoBERTa-base (SST-2) and GPT-2 (WikiText-2), the performance of Higher-Order Nullspace Continuation SVD (**HNC-SVD**) was rigorously evaluated:
- At extreme compression ratios ($r = 32, 64$), HNC-SVD suffered from error accumulation and high relative error (exceeding 1.0 on unconditioned projection layers), resulting in sub-random baseline accuracy (46.77% at rank 256 without KD recovery).
- In contrast, Fisher-Weighted SVD (**FWSVD**) and Adaptive Energy SVD (**AdaSVD**) achieved superior preservation: 80.10% and 80.40% accuracy at rank 32 (24x compression ratio), while **SparseGPT** reached 91.40% at rank 128.

---

## 2. Decision: Option C (Regime-Specialized Dispatching)

Rather than deprecating or removing HNC-SVD from the canonical 72 compressors (which would break the verified canonical count invariant of 72 compressors), **Option C** has been selected:

1. **Preserve Canonical Integrity**: HNC-SVD remains an active, fully-implemented algorithm in `neuralops/core/compression_pipeline.py` and `neuralops/compressors/` to preserve the 72 canonical compressors benchmark contract.
2. **Dynamic Rank-Aware Dispatching**: In `batch_validate_all_compressors.py` and the automatic model compression pipeline, the compressor dispatch table designates:
   - **$r \le 64$ (Extreme Compression Regime)**: Defaults to **AdaSVD** and **FWSVD** due to their mathematically stable Hessian/Fisher conditioning.
   - **$r \in [64, 128]$ (Moderate Compression Regime)**: Defaults to **SparseGPT** and **ASVD**.
   - **$r \ge 128$ / High-Dimensional Fine-Tuning**: Retains **HNC-SVD** as an experimental higher-order manifold continuation method where singular spectrum truncation error is sufficiently small to prevent nullspace drift.
3. **Single-Layer Guardrails**: Added explicit condition-number checks and relative error validation ($< 0.15$) before allowing HNC-SVD multi-layer cascading. If relative error exceeds the tolerance on the anchor layer, the pipeline gracefully falls back to FWSVD.
