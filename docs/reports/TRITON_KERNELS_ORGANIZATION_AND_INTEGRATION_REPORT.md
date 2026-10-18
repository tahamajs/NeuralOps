# Triton Kernels Organization, Integration & Architecture Report

**Subagent ID**: Subagent-60-Triton-Kernels-Unification-Lead  
**Date**: October 18, 2026  
**Target Environments**:  
- Remote Host: `taha@192.168.1.100` (`/mnt/d/SVD_linear_Attention` and symlink `/home/taha/SVD_Project`)  
- Local Repository: `/Users/tahamajs/Documents/uni/Project`  
- Hardware Target: NVIDIA GeForce RTX 4070 (Ada Lovelace, SM 8.9, 12 GB VRAM, CUDA 12.8, Triton 3.6.0)

---

## 1. Executive Summary & Objective

This report documents the architectural consolidation, bit-for-bit file verification, transparent aliasing, and test verification of the NeuralOps Triton JIT kernel ecosystem in response to the user request:

> *"/home/taha/SVD_Project/neuralops/kernels/triton make this fully ok to put them in ok place in anothr agent"*

Prior to this work, NeuralOps hosted Triton kernels across two directories:
1. `neuralops/core/triton_kernels/`: The canonical production hub housing the 26 categorized Triton kernel modules, dynamic block autotuning (`autotuner.py`), hardware profiling (`hardware_profile.py`), and `torch.library` zero-graph-break custom op registration (`op_registry.py`).
2. `neuralops/kernels/triton/`: The standalone suite containing the 13 flagship Ada Lovelace / Hopper kernels (`fused_lowrank_gemm.py`, `chunked_delta_gate_scan.py`, `cosformer_linear_attention.py`, etc.).

To ensure that any downstream agent or external client can consume kernels from **either namespace seamlessly without import breakages**, we established a **Unified Bi-directional Interoperability Layer** with PEP 562 dynamic resolution, cross-module import hooks in `_ModuleAliasFinder`, and full re-exports of all 13 flagship kernels and their PyTorch CPU/MPS fallbacks.

All 660 Triton unit tests pass on the remote NVIDIA RTX 4070 GPU in 8.34 seconds. The canonical system invariants (131 attention mechanisms, 72 compression engines, 26 Triton kernel modules) remain strictly intact and verified by `scripts/utils/verify_readme_counts.py`.

---

## 2. Remote File Audit & Bit-for-Bit Verification

All files in `/home/taha/SVD_Project/neuralops/kernels/triton` on remote host `taha@192.168.1.100` (which is a symlink to `/mnt/d/SVD_linear_Attention/neuralops/kernels/triton`) were inspected and compared against the local workspace:

| Remote File | Lines | Bytes | MD5 Hash (Remote & Local Workspace) | Parity Status |
| :--- | :---: | :---: | :--- | :---: |
| `block_sparse_streaming_attn.py` | 229 | 7,437 | `8f3f142c578ce6643e61940f652e79eb` | **Bit-for-Bit Match** |
| `chunked_delta_gate_scan.py` | 196 | 5,639 | `13fce8b80610766a8788ddb1a7d5be0b` | **Bit-for-Bit Match** |
| `cosformer_linear_attention.py` | 202 | 6,754 | `92aeecb64694f0e5f71fed99c690cff8` | **Bit-for-Bit Match** |
| `fused_act_scale_quant_gemm.py` | 152 | 4,573 | `402ee26a3d1387f21668e6188d89815a` | **Bit-for-Bit Match** |
| `fused_hadamard_w4a16_gemm.py` | 234 | 8,181 | `08fbbd24f10363aca140021f6033aabb` | **Bit-for-Bit Match** |
| `fused_kronecker_matmul.py` | 183 | 5,778 | `2ce1c6bb0424af229d60a450659a5c86` | **Bit-for-Bit Match** |
| `fused_latent_kv_cache_attention.py` | 188 | 6,613 | `1ed71e8b39d79dc72f1923171ed80bf3` | **Bit-for-Bit Match** |
| `fused_lowrank_gemm.py` | 326 | 10,769 | `73667854049061e7627b739c311a2ae2` | **Bit-for-Bit Match** |
| `hgrn2_fused_bwd_scan.py` | 147 | 4,668 | `00e4d31e39ae2c19a5b3fa7588b6b67b` | **Bit-for-Bit Match** |
| `lut_codebook_dequant_gemm.py` | 208 | 6,654 | `edc95172912c96222e5fcc6e28f8b371` | **Bit-for-Bit Match** |
| `rpca_sparse_gemm.py` | 265 | 9,520 | `8f790defda0f6930d763d878fea707a3` | **Bit-for-Bit Match** |
| `spectral_entropy_svd_gate.py` | 183 | 5,901 | `2438c5d5a491598aa4132e161f6c15b0` | **Bit-for-Bit Match** |
| `strided_tt.py` | 189 | 6,493 | `d1f9efda224f9a83b521c9e966b63a45` | **Bit-for-Bit Match** |
| `__init__.py` | 217 | 7,575 | Consolidated Bi-directional Re-export Engine | **Unified & Synced** |

---

## 3. Architecture & Standardization Layout

### 3.1 The Two Canonical Homes

1. **`neuralops/core/triton_kernels/` (Production & Dispatch Core)**
   - Houses the **26 canonical Triton kernel modules** verified by README Section 15.
   - Modules: `gdn_scan.py`, `gla_scan.py`, `hgrn2_scan.py`, `selective_scan.py`, `fms_scan.py`, `paged_attention.py`, `flash_decoding.py`, `sliding_window.py`, `fused_ce.py`, `fused_kd.py`, `fused_norm_linear.py`, `fused_rope.py`, `low_rank.py`, `elementwise.py`, `linear_attn.py`, `cosformer_triton_kernel.py`, `block_sparse.py`, `sparse_attn.py`, `gqa.py`, `varlen_attention.py`, `state_update.py`, `softplus.py`, `autotuner.py`, `hardware_profile.py`, `op_registry.py`, `_common.py` / `_types.py`.
   - Includes custom shape registrations, FakeTensor rules, dynamic cache memory budgeting, and autotuning configs (`autotuned_configs.json`).

2. **`neuralops/kernels/triton/` (Standalone Flagship JIT Suite)**
   - Houses the 13 modular, zero-copy, SRAM-pinned implementations designed for high-throughput execution on NVIDIA Ada Lovelace (RTX 4070 SM 8.9) and Hopper architectures.
   - Clean self-contained files that can be examined, unit tested, or dispatched in isolation.

### 3.2 Architectural Mapping Matrix

| Flagship Module (`kernels/triton/`) | Primary Triton Entry Point | Corresponding Production Hub (`core/triton_kernels/`) | PyTorch CPU Reference Fallback |
| :--- | :--- | :--- | :--- |
| `fused_lowrank_gemm.py` | `triton_fused_lowrank_gated_gemm` | `low_rank.py` | `triton_fused_lowrank_gated_gemm_pytorch_ref` |
| `rpca_sparse_gemm.py` | `triton_fused_lowrank_sparse_gemm` | `low_rank.py` | `triton_fused_lowrank_sparse_gemm_pytorch_ref` |
| `strided_tt.py` | `triton_strided_tt_contraction` & `_linear` | `low_rank.py` | `triton_strided_tt_contraction_pytorch_ref` |
| `fused_hadamard_w4a16_gemm.py` | `triton_fused_hadamard_w4a16_gemm` | `fused_norm_linear.py` | `triton_fused_hadamard_w4a16_gemm_pytorch_ref` |
| `lut_codebook_dequant_gemm.py` | `triton_lut_codebook_dequant_gemm` | `fused_norm_linear.py` | `triton_lut_codebook_dequant_gemm_pytorch_ref` |
| `fused_kronecker_matmul.py` | `triton_fused_kronecker_matmul` | `low_rank.py` | `triton_fused_kronecker_matmul_pytorch_ref` |
| `chunked_delta_gate_scan.py` | `triton_chunked_delta_gate_scan` | `gdn_scan.py` | `triton_chunked_delta_gate_scan_pytorch_ref` |
| `cosformer_linear_attention.py`| `triton_cosformer_linear_attention` | `cosformer_triton_kernel.py` | `triton_cosformer_linear_attention_pytorch_ref` |
| `block_sparse_streaming_attn.py`| `triton_block_sparse_streaming_attn` | `block_sparse.py` | `triton_block_sparse_streaming_attn_pytorch_ref` |
| `hgrn2_fused_bwd_scan.py` | `triton_hgrn2_fused_bwd_scan` | `hgrn2_scan.py` | `triton_hgrn2_fused_bwd_scan_pytorch_ref` |
| `spectral_entropy_svd_gate.py` | `triton_spectral_entropy_svd_gate` | `low_rank.py` | `triton_spectral_entropy_svd_gate_pytorch_ref` |
| `fused_act_scale_quant_gemm.py`| `triton_fused_act_scale_quant_gemm` | `fused_norm_linear.py` | `triton_fused_act_scale_quant_gemm_pytorch_ref` |
| `fused_latent_kv_cache_attention.py`| `triton_fused_latent_kv_cache_attention`| `linear_attn.py` | `triton_fused_latent_kv_cache_attention_pytorch_ref` |

---

## 4. Transparent Bi-directional Interoperability Layer

To guarantee that another agent or user script can import kernels without needing to know the exact internal organization:

### 1. `neuralops/kernels/triton/__init__.py`
Exports:
- All 13 standalone modules (`block_sparse_streaming_attn`, `chunked_delta_gate_scan`, etc.)
- All 14 kernel callables (`triton_fused_lowrank_gated_gemm`, etc.)
- All 13 PyTorch reference implementations (`triton_fused_hadamard_w4a16_gemm_pytorch_ref`, etc.)
- Dispatch & profiling utilities (`HAS_TRITON`, `triton_dispatch`, `Autotuner`, `check_triton_suitability`)
- Direct access to all 26 core production kernel modules
- PEP 562 `__getattr__` dynamic loader: Any attribute or submodule requested that exists in `neuralops.core.triton_kernels` is lazily resolved.

### 2. `neuralops/core/triton_kernels/__init__.py`
Exports:
- All 26 core production kernel modules and custom ops.
- All 13 flagship kernels and PyTorch references.
- PEP 562 `__getattr__` dynamic loader: Any standalone module requested (e.g. `chunked_delta_gate_scan`, `fused_hadamard_w4a16_gemm`) is dynamically retrieved from `neuralops.kernels.triton`.

### 3. `neuralops/__init__.py`
- Added `"kernels"` to `_SUBPACKAGES` to enable `import neuralops; neuralops.kernels.triton...`.
- Configured bi-directional mappings in `_ModuleAliasFinder._ALIASES` so that:
  ```python
  from neuralops.core.triton_kernels import chunked_delta_gate_scan
  # Automatically resolves to neuralops.kernels.triton.chunked_delta_gate_scan
  from neuralops.kernels.triton import gla_scan
  # Automatically resolves to neuralops.core.triton_kernels.gla_scan
  ```

### 4. `neuralops/kernels/__init__.py`
- Re-exports `triton` subpackage and adds `"triton"` to `__all__`.

---

## 5. Verification & Test Evidence

The unification was rigorously verified directly on the remote GPU host `taha@192.168.1.100`:

### 5.1 Remote Import and Attribute Inspection
```bash
/mnt/d/venv/bin/python3 -c "
import neuralops.kernels.triton as nkt
print('nkt items:', len(dir(nkt)))
import neuralops.core.triton_kernels as ctk
print('ctk items:', len(dir(ctk)))
import neuralops
print('direct access:', neuralops.kernels.triton.triton_fused_hadamard_w4a16_gemm)
from neuralops.kernels.triton import chunked_delta_gate_scan, gdn_scan
print('imports:', chunked_delta_gate_scan, gdn_scan)
from neuralops.core.triton_kernels import chunked_delta_gate_scan as cdgs
print('aliased core access:', cdgs)
"
```
**Output**:
```text
nkt items: 90
ctk items: 146
direct access: <function triton_fused_hadamard_w4a16_gemm at 0x76ca375e2350>
imports: <module 'neuralops.kernels.triton.chunked_delta_gate_scan' from '/mnt/d/SVD_linear_Attention/neuralops/kernels/triton/chunked_delta_gate_scan.py'> <module 'neuralops.core.triton_kernels.gdn_scan' from '/mnt/d/SVD_linear_Attention/neuralops/core/triton_kernels/gdn_scan.py'>
aliased core access: <module 'neuralops.kernels.triton.chunked_delta_gate_scan' from '/mnt/d/SVD_linear_Attention/neuralops/kernels/triton/chunked_delta_gate_scan.py'>
```

### 5.2 Canonical README Invariants Check
```bash
/mnt/d/venv/bin/python3 scripts/utils/verify_readme_counts.py
```
**Output**:
```text
README counts verified successfully (v1.4.0): 131 attention mechanisms, 72 compression engines, 26 Triton kernels.
```

### 5.3 Triton Kernel Unit Tests Suite
```bash
/mnt/d/venv/bin/pytest tests/test_triton_*.py -q
```
**Output**:
```text
........................................................................ [ 10%]
........................................................................ [ 21%]
........................................................................ [ 32%]
........................................................................ [ 43%]
........................................................................ [ 54%]
........................................................................ [ 65%]
........................................................................ [ 76%]
........................................................................ [ 87%]
........................................................................ [ 98%]
............                                                             [100%]
660 passed in 8.34s
```
All 660 tests passed, including the new `test_triton_kernels_unification_and_interoperability` test suite verifying cross-namespace aliasing, PyTorch fallback parity, and module re-exports.

---

## 6. Usage Recipes for Downstream Agents

Downstream agents and client developers can now import Triton kernels using any of the three standard paradigms:

### Pattern A: Direct Import from `neuralops.kernels.triton`
```python
import neuralops.kernels.triton as nkt

# Flagship 13 kernels
y = nkt.triton_fused_hadamard_w4a16_gemm(x, w_packed, scales, zeros)
o = nkt.triton_chunked_delta_gate_scan(q, k, v, alpha, beta)

# PyTorch references for CPU or validation
y_ref = nkt.triton_fused_hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros)

# Core 26 production modules
from neuralops.kernels.triton import gdn_scan, gla_scan, linear_attn
```

### Pattern B: Core Hub Import from `neuralops.core.triton_kernels`
```python
from neuralops.core.triton_kernels import (
    linear_attn,
    low_rank,
    triton_fused_lowrank_gated_gemm,
    triton_chunked_delta_gate_scan,
    triton_dispatch,
)
```

### Pattern C: Top-Level Framework Access
```python
import neuralops as no

# Check hardware suitability
if no.has_triton():
    y = no.kernels.triton.triton_fused_lowrank_gated_gemm(x, U, V, gate)
```

---

## 7. Conclusion

The Triton kernels in `/home/taha/SVD_Project/neuralops/kernels/triton` are fully verified, bit-for-bit synchronized with the repository, and seamlessly interconnected with the core production infrastructure. Any subagent can now import and consume any of the 26 production Triton kernel modules or 13 flagship kernels from either namespace without errors.
