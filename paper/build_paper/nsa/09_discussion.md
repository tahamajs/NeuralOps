<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Practical guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Union of Local + Global

NSA's union of a fixed window and dynamic top-$k$ captures both regimes. 
Neither alone matches the union's accuracy.

### 9.1.2 Takeaway 2: Attention Density is Low

Even at 63% density (short sequences), NSA preserves 95% of the attention 
mass. At long sequences, density drops below 10%.

### 9.1.3 Takeaway 3: Zero Parameters Added

NSA adds no parameters. Its benefit is **attention sparsity**, realized as 
lower memory and (with the right kernels) lower time.

## 9.2 When to Use NSA

### Recommended

- **Long-context inference** where sparse kernels are available
- **Streaming / KV cache** deployment (natural fit)
- **Interpretability studies** of head specialization
- **Base for kernel development** (sparse matmul)

### Not Recommended

- **Short sequences** where density is high
- **Parameter-constrained** applications
- **Environments without sparse kernels**

## 9.3 Choosing Window Size

| Sequence length | Recommended $w$ |
|---|---|
| 128 | 16 |
| 512 | 32 |
| 2048 | 64 |
| 8192 | 128 |

## 9.4 Choosing Top-K

| Use case | Recommended $k$ |
|---|---|
| Memory-efficient | 4 |
| Balanced | 8 |
| High quality | 16 |
| Aggressive capture | 32 |

## 9.5 Cost-Benefit

For DistilBERT at $L = 128$:

| Config | Accuracy | Density |
|---|---|---|
| Baseline | 90.71% | 100% |
| NSA ($w=16, k=8$) | 0.85 | 0.85% |
| NSA ($w=32, k=16$) | 90.83 | 50\%% |
| NSA ($w=64, k=32$) | 0.85 | 0.85% |

## 9.6 Interaction with Other Methods

| Method | NSA-compatible? |
|---|---|
| GQA | ✅ (multiplicative KV reduction) |
| RoPE | ✅ (RoPE before NSA) |
| Quantization | ✅ (INT8 NSA) |
| KV cache | ✅ (NSA reduces cache size) |
| MoE | ✅ (orthogonal) |

## 9.7 Open Questions

1. **Learned window and top-k:** Can we learn $w$ and $k$ per head?
2. **Sparse kernels:** When will they be widely available?
3. **Cross-layer selection transfer:** Do top-$k$ sets overlap across layers?
4. **Long-context evaluation:** How does NSA scale at $L > 8192$?

## 9.8 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Cheaper inference:** Reduces compute per token (with sparse kernels).
- **Interpretability:** Selection analysis reveals structure.

### Negative
- **Dual-use:** Efficiency gains benefit malicious applications.
- **Concentration:** Favors large labs.

### Environmental
Total compute: ~12.5 GPU-hours, ~12.4 kg CO₂eq.

---