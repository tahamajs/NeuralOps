# 10. Limitations

## 10.1 No Kernelization

The implementation materializes the full $[N \times N]$ attention matrix. 
No speedup.

## 10.2 Polynomial Breaks for Large Scores

For $|s| > 5$, the Taylor approximation is poor. This limits applicability 
to peaked attention distributions.

## 10.3 Task Coverage

Only SST-2 evaluated.

## 10.4 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.5 Single Seed

SE ~1.7%.

## 10.6 Raised on Missing Weights

`copy_weights` raises exceptions on missing weights.

## 10.7 RoPE Bug

`apply_rotary_pos_emb` has a fragile `'q_rot' in locals()` check. Fix needed.

## 10.8 No Comparison with Performer

We do not compare with RFF-based methods.

## 10.9 No Theoretical Bound

We provide no bound on downstream accuracy from the polynomial approximation.

## 10.10 Order Selection

Order 2 is a heuristic. Learned orders not tested.

## 10.11 Centering Overhead

Max-centering adds one reduction over the key dimension.

## 10.12 Code Quality

Research-grade:
- `raise` in copy_weights
- Fragile RoPE check
- No fused kernel

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---