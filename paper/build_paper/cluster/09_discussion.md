# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Linear Attention Without Random Features

ClusterFormer achieves linear complexity **without approximating softmax**. 
It's exact on the cluster level, differentiating it from Performer and 
Linformer.

### 9.1.2 Takeaway 2: Global + Local is Optimal

Neither global clustering alone nor local window alone matches the blend. 
The learnable gate discovers the optimal mix per layer.

### 9.1.3 Takeaway 3: Not a Compression Method

Parameter count increases by LoRA overhead. ClusterFormer targets 
**compute efficiency**, not size reduction.

## 9.2 When to Use ClusterFormer

### Recommended

- **Long sequences** ($L \geq 256$)
- **Attention-bound workloads**
- **Research on clustering attention**
- **Multi-architecture pipelines**

### Not Recommended

- **Short sequences** ($L < 100$) — overhead exceeds savings
- **Model size reduction** — use compression methods
- **Strict causal generation** — decoder uses exact attention anyway
- **Interpretability-first** — attention maps are cluster-level

## 9.3 Decision Flowchart

```
Need size reduction? ─── Yes ─── Use A3/AASVD/Adaptive SVD
  │ No
  ▼
Sequence length > 128? ─── No ─── Use baseline softmax
  │ Yes
  ▼
Latency critical? ─── Yes ─── Consider FlashAttention
  │ No
  ▼
Use ClusterFormer
```

## 9.4 Cost-Benefit

For DistilBERT at $L=256$:

| Metric | Baseline | ClusterFormer | Change |
|---|---|---|---|
| Attention FLOPs | 6.7M | 2.4M | **2.8× reduction** |
| Accuracy | 90.71% | 90.02% | -0.69% |
| Params | 66.9M | 67.55M | +589826M |
| Wall time | 4.2 ms | 3.8 ms | 1.10× |

## 9.5 Comparison with Published Methods

| Method | Type | Complexity | Global | Local |
|---|---|---|---|---|
| Longformer | Sparse | $O(LW)$ | Partial | Yes |
| Performer | Kernel | $O(Lm)$ | Yes | No |
| Reformer | LSH | $O(L\log L)$ | Yes | No |
| **ClusterFormer** | **Cluster** | **$O(LK)$** | **Yes** | **Yes** |

## 9.6 Deployment Considerations

### Latency

Beneficial for $L \geq 128$. Below that, overhead dominates.

### Memory

Attention matrix is $O(KL)$ instead of $O(L^2)$: **substantial savings** 
for long sequences.

### Interpretability

Cluster assignments give coarse interpretability; per-token attention is 
not directly accessible.

## 9.7 Open Questions

1. **Optimal $K$ per layer** — could be learned dynamically?
2. **Interaction with quantization** — INT8 + ClusterFormer?
3. **Causal decoding** — can clustering work for generation?
4. **Very long context** ($L > 4096$) — does the crossover hold?

## 9.8 Broader Impact

### Positive
- Compute reduction for long-context inference
- Democratizes long-context models
- Complements other efficiency methods

### Negative
- Modest parameter increase
- Clustering opacity
- Dual-use

### Environmental
Total compute: ~15 GPU-hours, ~12.4 kg CO₂eq.

---