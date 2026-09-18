<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

OHA **increases** parameter count by ~1.40%. It is designed for 
**inference efficiency at long sequences**, not for model size reduction.

## 10.2 Cross-Attention Not Supported

OHA does not implement cross-attention. Encoder-decoder models (T5, BART) 
are excluded.

## 10.3 Inference Overhead at Short Sequences

At $L = 128$, OHA is slower than baseline due to extra matmuls and 
compression projections. The benefit materializes only at $L \gg 256$.

## 10.4 Task Coverage

Only SST-2 is evaluated. OHA may behave differently on generation, 
long-context reasoning, and multi-class tasks.

## 10.5 Model Coverage

Tested on ≤124M parameter models. Not tested on >1B models.

## 10.6 Hardware

All experiments on RTX 4070 Laptop. Not tested on A100, CPU, or edge.

## 10.7 `raise` on Copy Failure

OHA's `copy_weights` **raises** `RuntimeError` on missing projections. If 
the source attention has an unusual structure, injection fails hard. This 
is safer than silent fallback but requires more careful source handling.

## 10.8 Nested Encoder Detection

RoBERTa requires `is_nested_encoder = True`. If mis-set, double projection 
causes scale collapse.

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.10 No Theoretical Guarantee

We provide no bound on approximation error vs the baseline attention.

## 10.11 No Comparison with Compressive Transformer

We do not run head-to-head comparison with Compressive Transformer, 
Recurrent Memory Transformer, or Set Transformer.

## 10.12 Unresolved Failures

None observed

## 10.13 Code Quality

Research-grade implementation. Known issues:

- `raise` in `copy_weights` may be too aggressive
- No unit tests
- `output_scale` parameter added but not used

---
