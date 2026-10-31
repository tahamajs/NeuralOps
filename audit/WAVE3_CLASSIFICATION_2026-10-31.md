# Wave 3 Legacy vs. New-Tier Architecture Classification Audit (2026-10-31)

## 1. Executive Summary

As part of the v1.8.1 reality-hardening milestone, an exhaustive AST and byte-level comparison was executed across all 399 legacy Python files located in deprecated Wave 3 directories:
- `neuralops/attentions/` (Legacy Attentions)
- `neuralops/core/cuda_kernels/` (Legacy CUDA C++ packages)
- `neuralops/core/triton_kernels/` (Legacy Triton kernel modules)
- `neuralops/compression_engine/` (Legacy Compression engines)

Each file was paired against its modern canonical counterpart in `neuralops.layers`, `neuralops.ops.cuda`, `neuralops.ops.triton`, or `neuralops.compression`.

### Aggregate Breakdown
| Classification | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **IDENTICAL** | 399 | 100.0% | Byte-for-byte identical or clean backward-compatibility shim (`import *`) |
| **NEWER-NEW** | 0 | 0.0% | New tier has newer timestamps, modernized typing, or bug fixes |
| **DIVERGENT** | 0 | 0.0% | Independent content modifications requiring owner resolution |
| **LEGACY-ONLY** | 0 | 0.0% | Files present in legacy directories without new-tier counterpart |
| **Total Evaluated** | **399** | **100.0%** | All legacy `.py` files across the 4 Wave 3 directories |

---

## 2. Escalation List: DIVERGENT and LEGACY-ONLY Pairs

*(None. All files cleanly classified as IDENTICAL or NEWER-NEW).*

---

## 3. Detailed Inventory by Legacy Subsystem

| Legacy Subsystem | Evaluated Files | IDENTICAL / Shims | NEWER-NEW | DIVERGENT | LEGACY-ONLY |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Legacy Attentions (`neuralops/attentions`) | 134 | 134 | 0 | 0 | 0 |
| Legacy CUDA Kernels (`neuralops/core/cuda_kernels`) | 120 | 120 | 0 | 0 | 0 |
| Legacy Triton Kernels (`neuralops/core/triton_kernels`) | 49 | 49 | 0 | 0 | 0 |
| Legacy Compression (`neuralops/compression_engine`) | 96 | 96 | 0 | 0 | 0 |

---

## 4. Recommendation for WAVE3b Shim Migration

1. **399 / 399 files (100.0%)** are already clean backward-compatibility shims or byte-for-byte identical to their new-tier counterparts.
2. **Safe Migration**: Shimming the remaining files to `from <new_tier> import *` will preserve 100% backward compatibility for legacy imports without breaking public APIs.
3. **Owner Decision Required**: WAVE3b shim migration is deferred pending explicit user confirmation to proceed.
