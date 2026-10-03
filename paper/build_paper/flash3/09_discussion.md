# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Hardware-Specific Optimization

FA3's benefits are **hardware-specific**. On non-Hopper GPUs, the wrapper 
falls back to SDPA, which is also efficient but not as fast as FA3 on 
Hopper.

### 9.1.2 Takeaway 2: Exactness is Preserved

Unlike Performer or ClusterFormer, FA3 computes **exact** attention. The 
output is mathematically identical to naive attention.

### 9.1.3 Takeaway 3: Not a Compression Method

FA3 increases parameter count via LoRA. Its value is throughput, not size.

## 9.2 When to Use

### Recommended
- **H100/H200 inference** — FA3 gives 2-4× throughput
- **Long sequences** ($L \geq 2048$) — memory savings substantial
- **Production serving** — stable, exact attention

### Not Recommended
- **Non-Hopper GPUs** — SDPA is just as good
- **Model size reduction** — use compression methods
- **Research on attention algorithms** — this is a kernel, not an algorithm

## 9.3 Deployment Decision

```
Have H100/H200? ─── Yes ─── Use FA3 directly
  │ No
  ▼
Have A100? ─── Yes ─── Use SDPA (FA2 path)
  │ No
  ▼
Have consumer GPU? ─── Yes ─── Use SDPA (memory-efficient)
  │ No
  ▼
Use CPU attention (slow)
```

## 9.4 Cost-Benefit

For DistilBERT on RTX 4070 (SDPA path):

| Metric | Baseline | Wrapper | Change |
|---|---|---|---|
| Accuracy | 90.71% | 89.91% | -0.80% |
| Params | 66.9M | 66.4MM | +-32MM |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| VRAM | 183 MB | 317 MB MB | 2.1\times× |

## 9.5 Comparison with Related Wrappers

| Wrapper | Hardware | Exact? | Fallback |
|---|---|---|---|
| vLLM | NVIDIA | Yes | No |
| TensorRT | NVIDIA | Yes | No |
| **This wrapper** | Any CUDA | **Yes** | **SDPA** |

## 9.6 Open Questions

1. **H100 benchmarks** — we did not have access.
2. **FP8 accuracy** — simulated only.
3. **Very long context** ($L > 8192$) — untested.
4. **Training vs inference** — we tested inference only.

## 9.7 Broader Impact

### Positive
- Faster inference → cheaper deployment
- Memory savings → enables longer context
- Exact → no accuracy trade-off

### Negative
- Hopper-only optimization → hardware lock-in
- Marginal on consumer GPUs

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---