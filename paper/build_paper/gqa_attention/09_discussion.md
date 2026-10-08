<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: GQA is a Memory Optimization

GQA does not significantly reduce parameters (only 1.05×). Its 
primary benefit is **KV cache** reduction (2.0×), which matters at 
inference time, especially for long sequences.

### 9.1.2 Takeaway 2: Quality Loss is Graceful

At $G = H/2$, quality drops only -0.80% on DistilBERT. Even at 
$G = H/4$, the drop is manageable. MQA ($G = 1$) is more aggressive and 
should be used with care.

### 9.1.3 Takeaway 3: Architecture Matters

RoBERTa shows the largest drop at $G = H/2$, DistilBERT the smallest. 
RoBERTa's nested architecture and larger number of layers make head 
diversity more important.

## 9.2 When to Use GQA

### Recommended

- **Long-context inference** ($L > 1024$)
- **Memory-constrained GPUs** where KV cache dominates
- **Batch inference** where KV scales with $B$
- **Combined with quantization** for extreme savings

### Not Recommended

- **Very short sequences** ($L < 128$)
- **Training from scratch** (MHA is better)
- **Tasks requiring high head diversity**

## 9.3 Decision Flowchart

```
Generating long sequences? ─── Yes ─── Use GQA
  │
  No
  ▼
KV cache dominating memory? ─── Yes ─── Use GQA
  │
  No
  ▼
Want to reduce parameters? ─── Use A3 / AASVD instead
  │
  No
  ▼
Keep MHA
```

## 9.4 Choosing $G$

| $G$ | KV ratio | Quality | Use case |
|---|---|---|---|
| $H/2$ | 2× | High | Safe default |
| $H/4$ | 4× | Good | Balanced |
| $H/6$ | 6× | OK | Aggressive |
| 1 | $H$× | Lower | Extreme memory constraint |

## 9.5 Cost-Benefit

For GPT-2 at $L = 4096$, $B = 4$:

| Config | KV cache | Inference time |
|---|---|---|
| MHA | 604 MB | 45 s |
| GQA (H/2) | 302 MB | 3.5 s |
| GQA (H/4) | 151 MB | 3.3 s |

## 9.6 Deployment

### vLLM

GQA is natively supported in vLLM's Paged Attention. Enable via config.

### TensorRT-LLM

Supported with `--num_kv_heads` flag.

### ONNX Runtime

Requires graph rewrite; see our export scripts.

## 9.7 Open Questions

1. **Optimal grouping strategy:** Random vs sequential vs learned?
2. **Interaction with MoE:** How does GQA interact with sparse experts?
3. **Cross-lingual transfer:** Does GQA degrade differently across languages?
4. **Theoretical bound:** What is the exact expressiveness loss?

## 9.8 Broader Impact

### Positive
- **Longer context windows:** Enables more capable applications.
- **Cheaper inference:** Reduces cost per token.
- **Edge inference:** Makes on-device LLMs more feasible.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Centralization:** Favors large labs with long-context models.

### Environmental
Total compute: ~10 GPU-hours, ~12.4 kg CO₂eq.
