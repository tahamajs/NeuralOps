# Triton Kernel Subsystem (`neuralops.ops.triton`)

`neuralops.ops.triton` is the canonical namespace for NeuralOps Triton operators (26 modules housing 62 unified operators across 4 tiers).

## Layout & Architecture
- Root files in this directory contain shared dispatch, autotuning, op registry, and general-purpose attention/state operators.
- `standalone/` contains 13 self-contained flagship fused operators with CPU fallback imports.
- `neuralops.core.triton_kernels` is fully maintained as a backward-compatibility shim.

## 4-Tier Structure
1. **Tier 1 (Standalone Flagship Kernels - 13 modules)**: Self-contained fused operators (e.g. `cosformer_linear_attention`, `fused_hadamard_w4a16_gemm`).
2. **Tier 2 (Core Categorized Modules - 27 files)**: Canonical scan and state updates (`gdn_scan`, `gla_scan`, `hgrn2_scan`, `selective_scan`, etc.).
3. **Tier 3 (Root Bridges - 2 modules)**: Compatibility adapters.
4. **Tier 4 (Extended Fused Kernels - 20 operators)**: Fused SwiGLU, RMSNorm, RoPE, ALiBi, Attention Sink, etc.
