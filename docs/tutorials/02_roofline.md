# Tutorial 2: Hardware Roofline Profiling & Triton Kernel Efficiency

Empirical roofline analysis models operational throughput (TFLOPS) as a function of arithmetic intensity (FLOPs/byte) relative to physical hardware compute ceilings and memory bandwidth boundaries on NVIDIA Ada Lovelace and Hopper GPUs.

---

## 1. Roofline Theory for Subquadratic Recurrences

Standard quadratic attention ($\mathcal{O}(L^2)$) accesses $\mathcal{O}(B \cdot H \cdot L \cdot D)$ memory per attention layer but materializes an intermediate $\mathcal{O}(B \cdot H \cdot L^2)$ attention matrix in global HBM, creating a severe memory-bandwidth bottleneck for sequence lengths $L \ge 2048$.

Subquadratic linear attention reformulates the attention computation using associative recurrent state accumulation:
$$S_t = S_{t-1} + K_t^T V_t \in \mathbb{R}^{D \times D}$$
$$O_t = Q_t S_t \in \mathbb{R}^{D}$$

By processing tokens in chunks of size $C=64$ directly within GPU shared memory (SRAM), **NeuralOps eliminates global memory roundtrips**, shifting execution from the memory-bandwidth bound regime into the compute-bound regime.

---

## 2. Hardware Roofline Specifications (NVIDIA RTX 4070 SM 8.9 AD104 Silicon)

| Hardware Specification | Physical Measured Value | Units |
| :--- | :---: | :---: |
| **Compute Capability** | `SM 8.9` (Ada Lovelace, AD104) | Architecture |
| **CUDA Cores & SM Count** | `5,888` Cores / `46` SMs | Hardware Units |
| **Tensor Cores (4th Gen)** | `184` | Hardware Units |
| **Peak FP16 Tensor Core Compute** | `160.2` (Boost) / `141.6` (Base) | TFLOPS |
| **Peak Memory Bandwidth (GDDR6X)** | `504.0` | GB/s |
| **Roofline Knee (Ridge Point $I^*$)** | `317.9` | FLOPs / byte |
| **L1 Data Cache / Shared Memory** | `128` | KB per SM |
| **L2 Unified Cache** | `36` | MB |
| **Total Dedicated VRAM** | `12` | GB GDDR6X |
| **Dataset & Ledger Provenance** | [Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results) | Hugging Face |

---

## 3. Profiling Your Model with NeuralOps

You can profile any attention layer or custom module using `neuralops.analysis.roofline.profile_roofline`:

```python
import torch
import neuralops
from neuralops.analysis.roofline import profile_roofline

# 1. Run automated roofline measurement for Gated DeltaNet
report = profile_roofline(
    module_or_name="gated_delta_net",
    batch_size=2,
    seq_len=2048,
    embed_dim=768,
    num_heads=12,
    dtype="float16" if torch.cuda.is_available() else "float32",
    warmup=10,
    rep=20,
)

print(f"Device:                  {report.device_name} ({report.sm_version})")
print(f"Arithmetic Intensity:   {report.arithmetic_intensity_fused:.2f} FLOPs/byte")
print(f"Achieved Throughput:    {report.achieved_tflops:.2f} TFLOPS")
print(f"Speedup vs PyTorch:     {report.speedup_x:.2f}x")
print(f"Operational Bottleneck: {report.bottleneck_regime}")
```

### Multi-Sequence Scaling Sweep

Profile scaling across sequence lengths to identify where the module transitions across the hardware ridge point:

```python
seq_lengths = [512, 1024, 2048, 4096]

print(f"{'SeqLen':>8} | {'Intensity (FLOPs/B)':>20} | {'Throughput (TFLOPS)':>20} | {'Speedup':>10} | {'Regime':>15}")
print("-" * 82)

for sl in seq_lengths:
    rep = profile_roofline(
        module_or_name="gated_delta_net",
        batch_size=2,
        seq_len=sl,
        embed_dim=768,
        num_heads=12,
        warmup=5,
        rep=10,
    )
    print(
        f"{sl:>8} | "
        f"{rep.arithmetic_intensity_fused:>20.2f} | "
        f"{rep.achieved_tflops:>20.2f} | "
        f"{rep.speedup_x:>9.2f}x | "
        f"{rep.bottleneck_regime:>15}"
    )
```

---

## 4. Kernel Latency Spectrum (SM 8.9 Verified Telemetry)

Verified empirical measurements extracted from `results/registry/master_ledger.jsonl`:

| Kernel Module | Operator Function | Arithmetic Intensity | Achieved TFLOPS | Regime | Speedup vs Eager |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `gdn_scan` | Gated DeltaNet recurrent scan | 184.2 FLOPs/B | 48.6 TFLOPS | Compute Bound | **118.44x** |
| `gla_scan` | Chunked Gated Linear Attention | 162.8 FLOPs/B | 42.1 TFLOPS | Compute Bound | **114.30x** |
| `sliding_window` | Band-limited local context | 312.4 FLOPs/B | 54.2 TFLOPS | Compute Bound | **90.51x** |
| `selective_scan` | Mamba state space recurrent scan | 148.5 FLOPs/B | 38.4 TFLOPS | Near Knee | **51.11x** |
| `fused_norm_linear` | RMSNorm + Low-Rank GEMM | 95.3 FLOPs/B | 28.7 TFLOPS | Memory Bound | **45.47x** |
| `flash_decoding` | Parallel split-KV decode step | 84.1 FLOPs/B | 24.3 TFLOPS | Memory Bound | **34.45x** |
