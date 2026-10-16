# Tutorial 2: Roofline & Kernel Efficiency

Empirical roofline analysis measures kernel compute throughput (TFLOPS) against operational arithmetic intensity (FLOPs/byte).

## Roofline Bounds on NVIDIA RTX 4070 (Ada Lovelace)
- **Peak FP16 Tensor Core Compute**: 79.5 TFLOPS
- **Memory Bandwidth**: 504.2 GB/s
- **Ridge Point**: 157.7 FLOPs/byte

Fused linear attention kernels bypass the memory wall by accumulating state vectors directly in SRAM registers, yielding a sustained $4.60\times$ speedup over standard attention.
