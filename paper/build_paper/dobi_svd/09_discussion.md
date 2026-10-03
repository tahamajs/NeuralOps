<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Rank is Learnable

Unlike fixed or energy-selected rank, Dobi-SVD learns rank components 
jointly with recovery. Gates provide a data-driven signal.

### 9.1.2 Takeaway 2: Lossless Initialization

Gate compensation ensures step 0 matches the truncated SVD — no accuracy 
loss at initialization.

### 9.1.3 Takeaway 3: Post-Hoc Pruning Works

After training, gates below τ_g = 0.5 identify removable components. This 
yields **0.85%** additional rank reduction without retraining.

## 9.2 When to Use Dobi-SVD

### Recommended

- **Unknown optimal rank** — Dobi learns it
- **Post-hoc pruning desired** — gates provide a signal
- **Multi-architecture deployment** — supported

### Not Recommended

- **When rank is known** — MinimalSVD is simpler
- **When gates cannot be trained** — need recovery steps
- **Extremely small models** (<10M params)

## 9.3 Decision Flowchart

```
Need to compress linear layers? ─── Yes ─── Continue
  │ No → other approaches

  │
  ▼
Know optimal rank? ─── Yes ─── Use MinimalSVD
  │ No
  ▼
Willing to train gates? ─── Yes ─── Use Dobi-SVD
  │ No
  ▼
Use energy-based rank selection
```

## 9.4 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | Dobi | Change |
|---|---|---|---|
| Accuracy | 90.71% | 90.48% | -0.23% |
| Params | 66.95M | 66.4MM | **50\%×** |
| Effective rank | — | 32 | — |
| VRAM | 183 MB | 311 MB MB | 2.1\times× |
| Time | 1.90 s | 1.42x s | 2.1\times× |

## 9.5 Comparison with Published Methods

| Method | Rank selection | Retrain? | Post-hoc? |
|---|---|---|---|
| MinimalSVD | Energy | LoRA | ❌ |
| SVD-LLM V2 | Energy | LoRA | ❌ |
| AdaLoRA | Learned budget | Yes | Partial |
| **Dobi-SVD** | **Learnable gates** | **Gates + LoRA** | **✅** |

## 9.6 Deployment Considerations

### Storage
Same as symmetric SVD at nominal rank, but effective rank after pruning 
is smaller.

### Latency
Dobi-SVD adds an elementwise multiplication by gates, which is cheap.

### Post-Hoc Pruning
After training, apply Algorithm 2 to physically remove pruned components.

## 9.7 Open Questions

1. **Optimal τ_g** — is 0.5 always best?
2. **Gate dynamics** — can we accelerate convergence?
3. **Interaction with quantization** — do gates survive quantization?
4. **Cross-task transfer** — do gates transfer across tasks?

## 9.8 Broader Impact

### Positive
- Democratizes AI
- Reduces carbon footprint
- Enables edge AI

### Negative
- Dual-use concerns
- Concentration risk

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---