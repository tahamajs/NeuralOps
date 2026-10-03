<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 – 1 page (300 – 600 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 AdaSplash is Not a Compression Method

**AdaSplash increases parameter count by ~0.88%** due to LoRA 
adapters. It does **not** reduce model size or VRAM. Any publication using 
AdaSplash should avoid describing it as "compression".

## 10.2 Inference Latency

AdaSplash is **slower** than softmax attention by ~1.9×:

- Entmax bisection: 25 iterations of the same computation
- Softmax fallback computation
- LoRA branch matmuls

For latency-critical applications, use softmax.

## 10.3 Task Coverage

Only **SST-2** is evaluated. Generalization to:
- Generation tasks (summarization, translation)
- Multi-class classification
- Token-level tasks
- Long-context tasks

is unknown.

## 10.4 Model Coverage

Three models tested (66.9M–124.6M). Not tested:
- Large models (>1B)
- Deep models (>24 layers)
- MoE
- Vision

## 10.5 Hardware

All on RTX 4070 Laptop. Behavior on:
- A100/H100: Unknown
- CPU inference: Unknown
- Edge devices: Unknown

## 10.6 Splash Masking on Encoders

Splash masking is only applied to **causal decoders**. Encoders use full 
attention. This is a design choice but not a fundamental limitation.

## 10.7 Attention Sink Count

We use $S = 4$ fixed. The optimal number may depend on the task and model. 
We do not ablate this.

## 10.8 No Sparse Hardware Speedup Demonstration

Despite sparsity, our PyTorch implementation uses **dense matmuls**. The 
sparsity is not exploited for actual speedup. Demonstrating real speedup 
requires specialized kernels.

## 10.9 Single-Seed Results

All results from seed 42. Standard error ~1.69%. Differences below 
3.4% may be noise.

## 10.10 T5 and Qwen Verification

T5 verification is not tested, Qwen is partial.

## 10.11 Bisection Numerical Stability

The bisection algorithm uses a fixed 25 iterations. For extreme score 
distributions, this may not converge to sufficient precision.

## 10.12 Code Quality

Research-grade:
- No CUDA kernel for fused entmax
- No sparse matmul optimization
- Broad exception handling

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.
