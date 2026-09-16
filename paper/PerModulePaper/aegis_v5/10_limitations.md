# 10. Limitations

## 10.1 Parameter Effect is Conditional

Whether AEGIS-V5 reduces parameters depends on `max_rank`:

- `max_rank=384`: no reduction
- `max_rank=128`: ~40% reduction
- `max_rank=64`: ~70% reduction

We do not make a blanket "compression" claim.

## 10.2 Decoder Fallback

For decoders with $L > W$, we fall back to standard causal attention. This 
preserves correctness but sacrifices linear complexity.

## 10.3 Task Coverage

Only SST-2 evaluated. Behavior on:
- Generation: unknown
- Multi-class: unknown
- Token-level (NER): unknown
- Long-context ($L > 512$): unknown

## 10.4 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.5 Single-Seed Results

SE ~1.7%. Differences below 3.4% may be noise.

## 10.6 Energy-Based Rank Selection

The energy fraction $\eta = 0.95$ is a heuristic. Different values may 
work better for different models.

## 10.7 Softpick Temperature

The `softpick_temperature` is set to 1.0 (i.e., standard softmax). 
Temperature < 1 (sparsification) was not systematically tested.

## 10.8 Head Split Ratio

The 2:1 split is empirical. We did not test 3:1 or 1:1 systematically.

## 10.9 RFF Randomness

The RFF projection is initialized randomly and not tuned. Seed-dependent 
variation is unmeasured.

## 10.10 No Comparison with Published Baselines

We do not compare with Performer, Linformer, Longformer, etc.

## 10.11 No Theoretical Bound on Downstream Accuracy

We provide no bound on task accuracy from SVD error.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No fused kernel
- Some numerical concerns at FP16

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---