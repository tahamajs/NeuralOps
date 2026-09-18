<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Compute-Only Compression

Nyströmformer reduces **attention compute**, not parameters or KV cache. 
It is orthogonal to weight/KV methods.

### 9.1.2 Takeaway 2: Encoder Only

Nyström approximation is not causal. Only encoder models benefit.

### 9.1.3 Takeaway 3: Portability Confirmed

The algorithm ports cleanly to non-BERT architectures with careful 
engineering (Conv1D, nested encoder, GQA, RoPE).

## 9.2 When to Use Nyströmformer

### Recommended

- **Long-sequence encoder** ($N > 512$)
- **Combined with weight compression** (MinimalSVD + Nyström)
- **Combined with KV compression** (MLA + Nyström)
- **Inference on compute-constrained hardware**

### Not Recommended

- **Short sequences** ($N \leq 128$)
- **Causal decoders** (GPT, Qwen)
- **When accuracy is critical** (approximation error)

## 9.3 Decision Flowchart

```
Attention compute bottleneck? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Encoder or decoder? ─── Decoder ─── Nyström not applicable
  │ Encoder
  ▼
N > 512? ─── No ─── Dense attention is fine
  │ Yes
  ▼
Use Nyströmformer
```

## 9.4 Cost-Benefit Analysis

For DistilBERT at $N=512$:

| Metric | Baseline | Nyström | Change |
|---|---|---|---|
| Accuracy | 12.4 ms% | 8.2 ms% | 0.85% |
| Time | 0.85 s | 0.85 s | **86.2×** |
| VRAM | 0.85 MB | 0.85 MB | **350 MB×** |
| Params | 66.95M | 66.95M | 1.00× |
| KV cache | Same | Same | 1.00× |

## 9.5 Comparison with Companion Methods

| Method | Reduces | Architecture | Best for |
|---|---|---|---|
| MinimalSVD | Params | Any | Storage-bound |
| SVD-LLM V2 | Params | Any | Storage-bound |
| Dobi-SVD | Params | Any | Unknown rank |
| MLA | KV cache | Any | Long context |
| **Nyströmformer** | **Compute** | Encoder | Long encoder |

## 9.6 Deployment Considerations

### Storage
No change.

### Latency
Speedup scales with $N / m$.

### Batch Size
Benefit increases with batch size.

### Combining Methods
Nyströmformer + MLA → both compute and memory reduced.
Nyströmformer + MinimalSVD → both compute and parameters reduced.

## 9.7 Open Questions

1. **Causal Nyström:** can we make Nyström causal?
2. **Optimal landmark selection:** adaptive pooling vs k-means?
3. **Combined compression:** how do methods interact?
4. **Cross-task:** does Nyström work for generation?

## 9.8 Broader Impact

### Positive
- Enables long-context encoders
- Reduces compute cost

### Negative
- Dual-use concerns

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---