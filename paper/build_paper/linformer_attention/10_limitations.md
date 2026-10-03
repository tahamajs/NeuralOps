# 10. Limitations

## 10.1 Decoder Fallback

For decoders with $L > k$, we fall back to dense attention. This **negates 
the linear complexity benefit** for the case where it's most needed.

## 10.2 Parameter Increase

Linformer adds ~7% parameters per layer. This is a real cost.

## 10.3 Sequence Length Extrapolation

For $L > L_\text{max}$, we interpolate $E, F$. This is a heuristic not 
validated on very long sequences.

## 10.4 Task Coverage

Only SST-2 evaluated. Generation, multi-class, and NER tasks untested.

## 10.5 Model Coverage

Tested up to 124M params. Behavior on larger models unknown.

## 10.6 Single-Seed Results

SE ~1.7%. Differences below 3.4% may be noise.

## 10.7 No Comparison with Original Linformer

We do not reproduce Wang et al. (2020) exactly. Our diagonal init, adaptive 
slicing, and causal fallback differ.

## 10.8 E, F Memory

$E, F \in \mathbb{R}^{k \times L_\text{max}}$ cost $2 k L_\text{max}$ 
parameters per layer. For large $L_\text{max}$, this grows.

## 10.9 Causal Analysis

Our causal fallback is a **correctness** measure, not a **performance** 
measure. It sacrifices efficiency.

## 10.10 The output_scale Parameter

`output_scale` is a scalar initialized to 1. It's trainable but rarely 
moves significantly.

## 10.11 Code Quality

Research-grade:
- Broad exception handling
- No fused kernel
- Some numerical stability concerns

## 10.12 Ethical Considerations

Dual-use. See Appendix E.

---