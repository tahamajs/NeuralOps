<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 FP8 Path Not Exercised

**The most important limitation.** On our test hardware (RTX 4070 Laptop), 
`flash_attn` is not available and/or FP8 tensor cores are not exposed. 
Therefore:

- The FP8 path was never taken.
- No FP8 speedup was measured.
- No FP8 accuracy impact was measured.

We state this **explicitly in the abstract** and throughout the paper.

**Implication.** This paper validates the **portability** of our 
implementation, not its FP8 performance.

## 10.2 No H100 Testing

Without access to Hopper hardware, we cannot validate the FP8 speedups 
claimed by the original FlashAttention-3 paper.

## 10.3 Parameter Increase

This module **increases parameter count** by **48.5\%%** because 
LoRA adapters are added. It is not a compression method.

## 10.4 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks 
is untested.

## 10.5 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 was not completed. No testing on 
LLaMA, Mistral, or >1B models.

## 10.6 No Cross-Attention

We raise `NotImplementedError` if `encoder_hidden_states` is provided.

## 10.7 Single Seed

No multi-seed runs. Differences below 3.4% may be noise.

## 10.8 RoBERTa Loss Oscillation

RoBERTa's recovery loss oscillates, suggesting optimization instability. 
We did not fully diagnose this.

## 10.9 No Numerical Error Analysis

We cannot quantify FP8 numerical error without a working FP8 path.

## 10.10 No Comparison with `flash-attn` Original

We do not run head-to-head with the original `flash-attn` library because 
it is not installed.

## 10.11 Training Instability on Larger Models

Not tested. LoRA rank, LR, and batch size are tuned for small models.

## 10.12 Hardware-Detection is Coarse

Our detection uses `hasattr(torch, "float8_e4m3fn")` and 
`FLASH_ATTN_AVAILABLE`. It does not check:

- CUDA compute capability
- Driver version
- Tensor core availability

This can cause false positives (attempt FP8 on unsupported hardware) or 
false negatives (skip FP8 on supported hardware).

## 10.13 Code Quality

Research-grade:

- No unit tests
- Broad exception handling in FP8 path hides errors
- No profiling instrumentation

## 10.14 What This Paper Is Not

This is **not**:

- A new attention mechanism
- A compression method
- An FP8 benchmark
- A scaling study

---