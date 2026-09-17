<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Implementation Bottleneck (Primary)

The pure-PyTorch implementation is **~84.5× slower** than baseline 
attention. This is the **dominant limitation** and prevents practical 
deployment.

**Root cause.** Nested Python loops (outer over chunks, inner over tokens) 
generate ~600 kernel launches per forward pass.

**Fix.** Requires a fused Triton or CUDA kernel.

## 10.2 Memory Exceeds Budget on 12-Layer Models

RoBERTa and GPT-2 (12 layers) **exceed 8 GB VRAM** during training, 
resulting in OOM. This prevents evaluation on these models.

**Cause.** State matrix + gradient checkpointed activations.

## 10.3 No Fusion Available

We did not implement a fused kernel in this work. The implementation uses 
`torch.utils.checkpoint` for memory but at 2× compute cost.

## 10.4 Parameter Overhead

Gated DeltaNet **adds ~10.7% parameters** compared to baseline:
- Z projection: +589K per layer
- α, β gates: +18K per layer
- LoRA: +123K per layer

This is not a compression method.

## 10.5 Single Seed

No multi-seed runs. Accuracy numbers are from one seed.

## 10.6 Task Coverage

Only SST-2. Long-context tasks (where linear attention has its main 
advantage) are not tested.

## 10.7 Model Coverage

Only DistilBERT completes. RoBERTa and GPT-2 OOM.

## 10.8 No Qwen Testing

Qwen3.5 with GQA and RoPE was not tested due to memory constraints.

## 10.9 Numerical Precision

State matrix is accumulated in FP32. For FP16 training, numerical drift 
may occur for long sequences. We test only $L = 128$.

## 10.10 Bidirectional Scan Overhead

For encoders, we run **two** full scans (forward + backward), **doubling** 
compute time. This is not accounted for in the "linear complexity" claim.

## 10.11 Gradient Checkpointing is On by Default

We enable checkpointing by default, which doubles forward time. Disabling 
it causes OOM on larger models.

## 10.12 No Theoretical Guarantee

We do not prove that Gated DeltaNet matches softmax attention's 
expressiveness. The Delta Rule is a heuristic.

## 10.13 Failure to Reproduce Literature

Published Gated DeltaNet results use Triton kernels. Our results do not 
match those speeds.

## 10.14 Code Quality

- Python loops in hot path
- Some redundant tensor allocations
- No profiling instrumentation
- Gradient checkpointing adds complexity

## 10.15 What This Paper Is Not

- Not a speedup method
- Not a practical attention replacement
- Not a comparison with Mamba or GLA
- Not a scaling study

---