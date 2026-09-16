# 10. Limitations

## 10.1 Delta-KV is Simulation Only

The Delta-KV step does not reduce KV cache memory at inference. The 
reconstructed tensor has the same shape as the original.

## 10.2 4-bit Quantization is Simulation Only

The STE-quantized deltas are stored in FP16/FP32. No bit-level savings.

## 10.3 SVD in Forward Pass

The forward pass includes a full SVD of the deltas. For long sequences, 
this is expensive:
- Time: $\mathcal{O}(\min(N d_h^2, N^2 d_h))$
- This may dominate the block-sparse savings

## 10.4 6D Gather Memory

The `[B, H, N_p, k_exp, P, d_h]` intermediate tensor is memory-heavy. At 
$N = 512$, $k = 4$, $P = 64$, $d_h = 64$: ~256 MB per layer.

## 10.5 Hard Failure on Missing Weights

`copy_weights` and `inject_moe_deltakv_paged` raise exceptions on failure. 
This is safer than silent failure but less robust.

## 10.6 Task Coverage

Only SST-2 evaluated.

## 10.7 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.8 Single Seed

SE ~1.7%.

## 10.9 Block-Sparse Speedup Saturation

Speedup is $\min(N_p/k_\text{exp}, \ldots)$. For small $N$, no benefit.

## 10.10 No Comparison with Actual PagedAttention

We do not compare with vLLM's PagedAttention.

## 10.11 Causal Fallback

For $N_p \leq 2$, we fall back to dense attention. This defeats the purpose 
for short sequences.

## 10.12 Code Quality

Research-grade:
- `raise` in inject
- No fused kernel
- SVD in forward

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---