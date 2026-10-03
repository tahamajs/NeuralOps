<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 This is a Reproduction

Nyströmformer is from Xiong et al. (2021). We do not claim to improve it.

## 10.2 Short Sequence Evaluation

At $N=128$ (SST-2), the Nyström branch does not activate. Our main results 
show **fallback behavior**, not Nyström approximation.

## 10.3 Causal Decoder Not Supported

Nyström approximation is not causal. GPT-2 and Qwen use dense causal 
attention, gaining no benefit.

## 10.4 No Parameter Reduction

Nyströmformer does **not** reduce parameter count or KV-cache size.

## 10.5 Approximation Error

Nyströmformer introduces approximation error. For highly non-low-rank 
attention matrices, error can be significant.

## 10.6 Landmark Selection Fixed

We use adaptive pooling. Other methods (k-means, random) might work better.

## 10.7 Task Coverage

Only SST-2 is evaluated. Long-context tasks (document classification, 
QA) may show larger benefits.

## 10.8 Model Coverage

Only three models tested. Larger models (BERT-large, LLaMA) not tested.

## 10.9 Hardware

Only tested on RTX 4070 Laptop.

## 10.10 Single Seed

All results from a single seed. SE ~1.69%.

## 10.11 Heuristic Hyperparameters

`num_landmarks = 32`, short sequence threshold = 128 are empirical.

## 10.12 No Comparison with Performer/Linformer

We do not directly compare with other efficient attention methods.

## 10.13 Code Quality

Research-grade. Known issues:
- Broad exception handling
- `raise` in some paths should log
- No unit tests for edge cases

## 10.14 Ethical Considerations

See Appendix E.

---