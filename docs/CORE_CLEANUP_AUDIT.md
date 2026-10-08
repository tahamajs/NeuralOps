# DEEP CODE CLEANUP & AUDIT: neuralops/core/ & TRITON KERNELS

## Executive Summary
This report provides a deep structural and quality audit of  with special emphasis on  (16 kernel modules) and cross-file duplications across core modules. No source files were modified during this audit.

---

## Section 1: Triton Kernel Matrix (16 Files)

| File | Lines | @triton.jit | Fallback | Unit Test | Docstring | Type Hints | Dead Code | Duplicate File | Imports _common | Imports fla | In \_\_init\_\_ | In \_\_all\_\_ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | 604 | NO | YES | YES | YES | YES | NO | NO | N/A | NO | YES | NO |
|  | 558 | YES | YES | NO | YES | YES | NO | NO | NO | NO | YES | NO |
|  | 516 | YES | YES | NO | YES | YES | NO | NO | NO | NO | YES | NO |
|  | 495 | YES | YES | NO | YES | YES | NO | NO | NO | YES | YES | NO |
|  | 163 | YES | YES | YES | YES | YES | NO | NO | YES | NO | YES | NO |
|  | 267 | YES | YES | NO | YES | YES | NO | NO | NO | NO | YES | NO |
|  | 451 | YES | YES | YES | YES | YES | NO | YES (triton_linear_attn) | YES | YES | YES | NO |
|  | 258 | YES | YES | YES | YES | YES | NO | NO | YES | NO | YES | NO |
|  | 450 | YES | YES | NO | YES | YES | NO | NO | NO | NO | YES | NO |
|  | 36 | YES | NO | YES | NO | NO | NO | NO | NO | NO | YES | NO |
|  | 291 | YES | YES | YES | YES | YES | NO | YES (triton_sparse_attn) | YES | YES | YES | NO |
|  | 306 | YES | YES | YES | YES | YES | NO | NO | YES | NO | YES | NO |
|  | 872 | YES | YES | YES | YES | YES | YES | NO | NO | YES | NO | NO |
|  | 268 | YES | YES | NO | YES | NO | YES | YES (linear_attn) | NO | NO | NO | NO |
|  | 180 | YES | YES | NO | YES | YES | YES | YES (sparse_attn) | NO | NO | NO | NO |
|  | 1185 | YES | YES | YES | YES | YES | YES | NO | NO | YES | NO | NO |

---

## Section 2: Triton Duplication Report

### Specific Kernel Questions:
1. ** vs **:
   -  is the canonical fused chunked linear attention kernel with standard autograd dispatch, autotuning, and PyTorch fallback in .
   -  is an unintegrated legacy copy with its own internal mock JIT and fallback (). It should be deprecated and removed in favor of .
2. ** vs **:
   -  is the canonical block-sparse Triton kernel integrated with  and .
   -  is a standalone older version implementing duplicated block-sparse logic and custom dummy JIT decorators.
3. ** vs **:
   -  provides primitive low-rank bottleneck tensor contractions ( A B$).
   -  is an entire 1185-line monolithic attention layer (containing full PyTorch attention class, Rotary embedding helpers, and custom kernels). It does not duplicate , but duplicates  and belongs in  rather than .
4. **Duplicate Helper Functions across Kernels**:
   - : Implemented in  (lines 62-85) and  (lines 65-88). Should be removed and imported from .
   - : Implemented locally inside  and . Should be imported from .
5. **Fallback Consolidation**:
   -  contains canonical CPU/CUDA PyTorch fallbacks (, , , , ).
   -  lacks a pure PyTorch fallback when  is None (it should fall back directly to ).
6. ** Import Discrepancies**:
   - 8 kernel files define their own fallback try/except blocks to set  instead of importing from .

---

## Section 3: Core Files Audit (neuralops/core/)

| File | Lines | Purpose | Dead / Deprecated Code | Duplicates | Missing Docs / Types | Issues |
|---|---|---|---|---|---|---|
|  | 190 | Base class for attention and compressors | None | None | None | None |
|  | 406 | Central registry for attentions & compressors | None | None | None | None |
|  | 643 | Registry alias mappings | None | None | N/A | None |
|  | 254 | Parameter-efficient adapter hooks | None | None | None | None |
|  | 1173 | AST / module replacement engine | None | None | None | Large file (1173 lines), consider modularizing |
|  | 419 | Shared primitives (LoRALinear, FactorizedLinear) | None |  duplicated | None | Has local copy of  |
|  | 160 | Attention masking & token validation | None | None | None | Clean |
|  | 65 | Grouped-Query Attention KV repeating | None | None | None | Clean |
|  | 1078 | Attention utilities & benchmark harness | Unreferenced helpers |  duplicated | Minor | File is too long (1078 lines); combines benchmarking and shape utilities |
|  | 1459 | Multi-mode benchmarking engine | None | Benchmark duplication | None | Located in  instead of  |
|  | 548 | Structured / unstructured pruning utilities | None | None | None | Clean |
|  | 423 | Legacy ensemble attention | Low references (2) | None | None | Legacy code; candidate for deprecation |
|  | 205 | T5 architecture compatibility | 1 reference | None | None | Specialized compatibility adapter |
|  | 202 | Architecture inspection & detection | None | None | None | Clean |
|  | 231 | HuggingFace model wrapper | 2 references | None | None | Lightly referenced |
|  | 253 | PyTorch selective scan reference | None | None | None | Clean |
| | 96 | State dictionary snapshotting & restore | 2 references | None | None | Clean |
|  | 87 | GPU memory tracking utilities | None | None | None | Clean |
|  | 93 | Legacy mask utilities | 0 references |  | None | Dead code (0 external references) |
|  | 154 | General model surgery utilities | None | None | None | Clean |
|  | 120 | Compression and speedup metrics | None | None | None | Clean |
|  | 87 | Low-rank SVD decomposition | 0 references |  | None | Dead code (0 external references) |
|  | 63 | Sequential compression pipeline orchestrator | None | Duplicate module name | None | Shares name with  directory |
|  | 102 | Seed, device, and tensor helpers | None | None | None | Clean |
|  | 17 | Configuration typing stub | None | None | Yes (only 17 lines) | Stub file |
|  | 97 | Core package exports | None | None | None | Clean |
|  | 75 | Canonical Rotary Position Embeddings | None | None | None | Canonical implementation |
|  | 11 | Re-export shim for triton linear attention | Legacy shim | Redirect | None | Clean redirect |
|  | 10 | Re-export shim for triton sparse attention | Legacy shim | Redirect | None | Clean redirect |

---

## Section 4: Cross-File Duplication in neuralops/core/

1. ****:
   - Canonical: 
   - Duplicates:  (L65-88) and  (L62-85).
   - Action: Replace duplicate definitions with import from .
2. ****:
   - Canonical: 
   - Duplicates:  (L23-35),  (L45-58), , .
   - Action: Consolidate all copies to import from .
3. ****:
   - Canonical: 
   -  does not define , but  and  have local re-implementations.
4. **Dead Utilities in **:
   -  has 0 references in the codebase; superseded by .
   -  in  has 0 references; superseded by .

---

## Section 5: Standard Structure Compliance

Standard structure requires:
-  imported from 
-  on kernel functions
- PyTorch fallback provided
- Public dispatch function checking 
- Public symbols exported in 

### Compliance Breakdown:
- **Fully Compliant**: , , , , 
- **Partially Compliant (Missing standardized dispatch or fallback in _common)**: , , , , 
- **Non-Compliant (Monolithic / Legacy / Missing Fallbacks)**:
  - : Missing PyTorch fallback function.
  -  & : Legacy duplicates lacking imports from .
  -  & : Monolithic attention layers with duplicate RoPE and GQA helpers placed erroneously inside .

---

## Section 6: Test Coverage

| Kernel / Module | Test File | Test Status |
|---|---|---|
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  |  | Covered |
|  | None | MISSING UNIT TEST |
|  | None | MISSING UNIT TEST |
|  | None | MISSING UNIT TEST |
|  | None | MISSING UNIT TEST |
|  | None | MISSING UNIT TEST |
|  | None | Redundant file |
|  | None | Redundant file |

---

## Section 7: Prioritized Fix Plan

### Priority 1 (CRITICAL)
1. **Deduplicate Kernel Files**:
   - Deprecate and remove redundant  and , updating bridge imports to  and .
2. **Move Misplaced Monolithic Layers**:
   -  and  in  contain full attention layers and should import shared RoPE/GQA helpers rather than copy-pasting them.
3. **Consolidate **:
   - Remove redundant local copies of  in  and  and import canonically from .

### Priority 2 (IMPORTANT)
1. **Standardize  and Fallbacks**:
   - Ensure all kernels import  exclusively from .
   - Add PyTorch fallback to .
2. **Remove Dead Core Files**:
   - Remove  and  (both have 0 references).
3. **Relocate Benchmark Engine**:
   - Move  (1459 lines) from  to .

### Priority 3 (NICE TO HAVE)
1. **Split Large Utility Files**:
   - Refactor  (1078 lines) into smaller sub-modules.
2. **Add Missing Tests**:
   - Implement unit tests for , , , and .
