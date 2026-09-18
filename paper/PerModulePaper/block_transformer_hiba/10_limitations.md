# 10. Limitations and Failure Modes

## 10.1 Calibration and Data Sensitivity
The empirical effectiveness of `block_transformer_hiba` is intrinsically conditioned on the quality and distribution of calibration sequences. When calibration data fails to represent the target task distribution, the computed spectral energy or activation covariances may over-index on spurious correlations, resulting in heightened reconstruction error on out-of-distribution inputs.

## 10.2 Architectural Constraints and Rank Granularity
Several concrete implementation constraints govern the deployment of `block_transformer_hiba`:
1. **Rank Divisibility Requirements**: In multi-head attention modules, factorized ranks must align with head counts ($r \pmod H == 0$) to ensure valid tensor reshaping across attention heads without irregular padding.
2. **Short Sequence Overhead**: For very short sequence lengths ($N < 16$), the overhead of executing chained factorized projections ($X V^\top U^\top$) can exceed monolithic GEMM operations on modern GPUs due to kernel launch overhead.
3. **Hardware-Specific Acceleration Boundaries**: Semi-structured 2:4 sparsity requires NVIDIA Ampere or newer tensor cores; running on older architectures incurs software masking overhead without wall-clock acceleration.

## 10.3 Dynamic Range and Quantization Underflow
In extreme low-rank regimes ($r \ll d_{\text{in}}$), singular value truncation discards subtle high-frequency feature directions. When combined with FP16 or BF16 mixed-precision training, low-magnitude singular values can underflow to zero, necessitating manual numerical clamping ($10^{-4}$) as enforced by the safety guardrails in `block_transformer_hiba`.
