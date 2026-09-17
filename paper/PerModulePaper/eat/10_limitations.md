<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

EAT **increases** parameter count by ~66.4M% due to LoRA. It 
does not reduce storage, memory, or model size on disk.

## 10.2 Inference Overhead

EAT is ~2.1\times× slower than baseline when LoRA is enabled. 
The overhead comes from:

- Additional LoRA matmuls per projection
- FP16 → FP32 cast in LoRA branch
- Additional kernel launches

## 10.3 Silent Weight Transfer Failure

EAT's `copy_weights` logs warnings on missing projections but continues 
with random initialization. This can result in a completely broken forward 
pass without any error raised.

## 10.4 Task Coverage

Only SST-2 is evaluated. EAT may behave differently on:

- Generation tasks
- Long-context tasks
- Multi-class classification

## 10.5 Model Coverage

Tested on models with ≤124M parameters. Not tested on:

- >1B parameter models
- Very deep models (>24 layers)
- Mixture-of-Experts

## 10.6 Hardware

All experiments on RTX 4070 Laptop. Not tested on:

- A100/H100
- CPU
- Edge devices

## 10.7 RoBERTa-Specific Issues

RoBERTa's nested encoder requires correct `is_nested_encoder` setting. If 
misconfigured, double projection causes 15-25% accuracy drop.

## 10.8 No Theoretical Guarantees

We provide no bound on:

- Approximation error of EAT vs baseline
- Accuracy loss under pruning
- Optimal pruning ratio

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.10 Pruning Only at Inference

Training with pruning is not supported. Gradient flow through `argsort` 
is undefined.

## 10.11 No Comparison with Some Baselines

We do not directly compare with:

- LTP, PuMer (token pruning)
- DynamicViT, EViT (vision token pruning)
- ToMe (token merging)

## 10.12 Unresolved Failures

0.85

## 10.13 Code Quality

Research-grade implementation. Known issues:

- Silent failure paths
- Broad exception handling
- No unit tests for edge cases

---