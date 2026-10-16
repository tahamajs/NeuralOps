# Dead Code Master Report (Round 2)

## Executive Summary

| Category | Count | Actionable | Status |
| :--- | :---: | :---: | :---: |
| Unused imports (F401) | 81 | 81 | ✓ Cleaned / Resolved |
| Unused variables (F841) | 12 | 12 | ✓ Prefixed / Removed |
| Dead functions | 30 | 0 | ✓ Preserved (Triton / Context handlers) |
| Dead classes | 0 | 0 | ✓ All classes active |
| Dead files candidates | 25 | 0 | ✓ All verified active & tested |
| Duplicate functions | 129 groups | Review | Reported in audit |
| Duplicate classes | 12 groups | Review | Reported in audit |
| Commented-out code | 827 | Review | Kept for algorithmic documentation |
| TODO/FIXME/HACK/XXX | 0 | 0 | Zero technical debt markers |
| Empty / Trivial tests | 0 | 0 | Zero empty tests |
| Unused dependencies | 0 | 0 | `packaging` required for version checks |

---

## Detailed Reports

- [Unused Imports & Variables Report](DEAD_CODE_AUDIT_UNUSED_IMPORTS.md)
- [Dead Functions & Classes Report](DEAD_CODE_AUDIT_FUNCTIONS.md)
- [Dead Files Archive Record](../neuralops/_archive/dead_code_2026_10/README.md)
- [Duplicate Code Report](DEAD_CODE_AUDIT_DUPLICATES.md)
- [Comments & Markers Report](DEAD_CODE_AUDIT_COMMENTS.md)
- [Dead & Empty Tests Report](DEAD_CODE_AUDIT_TESTS.md)
- [Dependencies Usage Report](DEAD_CODE_AUDIT_DEPS.md)

---

## Cleanup & Retention Summary

### Cleaned
- **81 unused imports** across core modules, attentions, compressors, benchmarks, and training utilities.
- **12 unused local variables** safely prefixed or pruned.
- **4 re-exports** registered properly in `__all__` lists (`get_best_configurations`, `get_deepspeed_config`, `apply_fsdp_wrapping`, `OPTIMIZER_REGISTRY`, `results_saver`).

### Preserved
- **Core Registries**: Exactly 159 attention mechanisms + 73 compressor modules = 232 modules.
- **Public API**: Complete backward-compatibility preserved across all 657 module aliases.
- **Triton Kernels**: Flash-decoding, Paged Attention, and Fused RoPE kernels retain complete Triton C-ABI parameter signatures.

---

## Verification Matrix

| Check | Expected | Actual | Result |
| :--- | :---: | :---: | :---: |
| `ATTENTION_REGISTRY` | 159 | 159 | ✓ PASS |
| `COMPRESSOR_REGISTRY` | 73 | 73 | ✓ PASS |
| Pyflakes / Ruff (F401, F841) | 0 issues | 0 issues | ✓ PASS |
| Top-level package imports | Clean | Clean | ✓ PASS |
