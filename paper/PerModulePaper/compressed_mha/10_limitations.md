# 10. Limitations

## 10.1 Task Coverage

Only SST-2 evaluated. Behavior on:
- Generation: unknown
- Multi-class: unknown
- Token-level: unknown
- Long-context: unknown

## 10.2 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.3 Rank Selection

Our default rank (`min(384, all_head + 2*kv_head, hidden)`) is often 
**= hidden_size**, providing no compression. Users must explicitly set a 
lower rank.

## 10.4 Nested Detection Assumptions

We assume:
- Models with `out_lin`/`c_proj`/`o_proj` are flat.
- Models with `self.query` + `output.dense` are nested.

Custom architectures may violate these assumptions.

## 10.5 LoRA Dependency

Without LoRA recovery, accuracy drops significantly. The method is not 
"post-training" in the strict sense.

## 10.6 Inference Latency

Two matmuls per projection vs one. On high-bandwidth GPUs, this may 
slightly increase latency.

## 10.7 Single-Seed Results

SE ~1.7%. Differences below 3.4% may be noise.

## 10.8 No Theoretical Bound on Downstream Accuracy

We bound **reconstruction** error (Frobenius), but not task-level accuracy.

## 10.9 Fallback Behavior

On weight extraction failure, we log a warning and proceed with random init. 
This may silently degrade accuracy. Users should check logs.

## 10.10 Joint SVD Assumption

We assume Q, K, V share an input subspace. For models with **very** different 
Q/K/V distributions, joint SVD may be worse than independent.

## 10.11 No Comparison with Distillation

We do not compare with DistilBERT/TinyBERT-style distillation.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No fused kernel
- Some numerical concerns at FP16

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---