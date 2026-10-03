<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: One Module, Four Architectures

UnifiedRoPE works across DistilBERT, RoBERTa, GPT-2, and Qwen without 
per-architecture modifications.

### 9.1.2 Takeaway 2: Replacement, Not Compression

UnifiedRoPE preserves parameter count, KV cache, and compute. It is a 
**substrate** for other techniques.

### 9.1.3 Takeaway 3: LoRA Recovery Needed

Adding RoPE to models trained with absolute positions requires LoRA 
recovery to adapt.

## 9.2 When to Use UnifiedRoPE

### Recommended

- **As a substrate** for compression (SVD, MLA, Nyström)
- **For adding RoPE** to encoder models
- **For unified multi-architecture deployment**

### Not Recommended

- **When compression is the goal** — use MinimalSVD, Dobi, etc.
- **When accuracy is critical** — RoPE addition may reduce accuracy

## 9.3 Decision Flowchart

```
Goal? ─── Compression ─── Use MinimalSVD / Dobi / MLA
  │
  Portability ─── Use UnifiedRoPE
  │
  Add RoPE to encoder ─── Use UnifiedRoPE
```

## 9.4 Cost-Benefit Analysis

For DistilBERT:

| Metric | Baseline | UnifiedRoPE | Change |
|---|---|---|---|
| Accuracy | 90.71% | 87.96% | -2.75% |
| Params | 66.95M | 66.95M | 1.00× |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| VRAM | 183 MB | 341 MB MB | 2.1\times× |

## 9.5 Comparison with Companion Methods

| Method | Purpose | Params | KV | Compute |
|---|---|---|---|---|
| MinimalSVD | Params | ↓ | — | — |
| Dobi-SVD | Params | ↓ | — | — |
| MLA | KV | — | ↓ | — |
| Nyström | Compute | — | — | ↓ |
| **UnifiedRoPE** | **Portability** | — | — | — |

## 9.6 Deployment Considerations

### Storage
No change.

### Latency
Slight overhead from RoPE application.

### Batching
No change.

## 9.7 Open Questions

1. **RoPE for encoders:** does adding RoPE always help?
2. **RoPE base:** what's the optimal base for encoder models?
3. **Combined with compression:** does UnifiedRoPE + MinimalSVD work?
4. **Long context:** how does UnifiedRoPE scale?

## 9.8 Broader Impact

### Positive
- Unified API for multi-architecture deployment
- Enables RoPE for encoder models

### Negative
- Dual-use concerns

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---