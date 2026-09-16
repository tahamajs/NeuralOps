# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Joint SVD is Efficient

Sharing the input basis across Q, K, V yields **{{JOINT_GAIN}}%** fewer 
parameters than independent SVD at the same accuracy.

### 9.1.2 Takeaway 2: Nested Detection Matters

The detection order fix restored DistilBERT from 50.92% collapse to 
{{D_ACC}}%.

### 9.1.3 Takeaway 3: LoRA Recovery is Essential

Without LoRA, accuracy drops by {{NOLORA_DROP}}%. With LoRA (rank 16), it 
recovers most of the gap.

## 9.2 When to Use Compressed MHA

### Recommended
- **Memory-constrained deployment**
- **Fine-tuning pipelines** with LoRA
- **Multi-architecture** codebases
- **Base for quantization** (compress then quantize)

### Not Recommended
- **Latency-critical**: two matmuls vs one
- **Extremely small models** ($d < 256$)
- **Extremely large rank needed**: $r > d/2$

## 9.3 Decision Flowchart

```
Want to compress attention projections? ─── Yes ─── Use Compressed MHA
  │ No
  ▼
Need sequence-level compression? ─── Yes ─── Use Linformer/LEKAv3
  │ No
  ▼
Keep standard attention
```

## 9.4 Cost-Benefit

For DistilBERT with rank 64:

| Metric | Baseline | Compressed | Change |
|---|---|---|---|
| Params | 66.95M | {{D_PARAMS}}M | **{{D_PARAM_PCT}}%** |
| Attention params | 14.2M | {{D_ATTN_PARAMS}}M | **{{D_ATTN_RATIO}}×** |
| Peak VRAM | 183 MB | {{D_VRAM}} MB | {{D_VRAM_PCT}}% |
| Accuracy | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |

## 9.5 Comparison with Prior Compression Methods

| Method | Axis | Joint | Params red. | Accuracy |
|---|---|---|---|---|
| Linformer | Sequence | No | No | {{LIN_ACC}} |
| A3 (SVD) | Feature | No | Yes | {{A3_ACC}} |
| AASVD | Feature | No | Yes | {{AASVD_ACC}} |
| **Compressed MHA** | **Feature (joint)** | **Yes** | **Yes** | **{{D_ACC}}** |

## 9.6 Deployment Considerations

### Memory

Compression ratio {{D_RATIO}}× reduces both disk and VRAM footprint.

### Latency

Two matmuls per projection vs one for baseline. On modern GPUs, the smaller 
matrices are often faster due to reduced memory bandwidth.

### Batching

Compression scales well with batch size — the smaller weight matrices fit 
better in cache.

## 9.7 Open Questions

1. **Optimal rank allocation:** Per-layer rank schedules?
2. **Joint with output:** Can we joint-factorize Q/K/V/O?
3. **Cross-layer SVD:** Share factors across layers?
4. **Interaction with quantization:** Combined INT8 + SVD?

## 9.8 Broader Impact

### Positive
- Reduces model size for edge deployment
- Complements other compression methods
- Multi-architecture support aids adoption

### Negative
- Requires fine-tuning for recovery
- Adds complexity vs simpler pruning
```

---