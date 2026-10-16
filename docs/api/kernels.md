# Kernels API Reference

NeuralOps registers **26 Triton JIT kernel modules** and complementary native CUDA extensions for subquadratic acceleration.

## Flagship Triton Kernels
1. `fused_linear_attention`: High-throughput recurrent attention forward/backward scan.
2. `paged_attention`: Virtual memory KV-cache and recurrent state routing.
3. `fused_layernorm_linear`: Zero-overhead fused normalization and low-rank GEMM.
4. `fast_gelu`: Vectorized activation function with register caching.
5. `swiglu`: High-density gated linear unit operator.
6. `rotary_embedding`: In-place rotary position encoding (RoPE).

## Hardware Support
- NVIDIA Hopper (SM 9.0)
- NVIDIA Ada Lovelace (SM 8.9)
- NVIDIA Ampere (SM 8.0 / 8.6)
- Native PyTorch CPU & Apple Metal (MPS) fallback operators.
