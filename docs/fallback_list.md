# NeuralOps 3-Level Dispatch Fallback Ledger

This document tracks operations currently forced to fallback implementations due to known kernel bottlenecks or hardware incompatibilities.

## fused_norm_linear

- **Status:** FALLBACK_ONLY
- **Reason:** Native CUDA kernel is 240× slower than PyTorch reference (measured 2026-10-18 on RTX 4070 SM 8.9).
- **Root cause:** Serial `for(k)` loop with uncoalesced memory access (`weight[n*K+k]`) and redundant `x_norm` recomputation per `(m, n)` pair. No shared memory tiling.
- **Fix path:** Rewrite as tiled GEMM (use `act_scale_gemm` as template with `BLOCK_M=16, BLOCK_N=32, BLOCK_K=32` shared memory staging).
- **Timeline:** Post-defense priority. Not blocking release.
