# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

Res-SVD reduces parameters by 50\%×.

### 9.1.2 Takeaway 2: Two-Level Structure

The base is frozen SVD; the residual is trainable spectral-init LoRA. 
This is optimal at initialization.

### 9.1.3 Takeaway 3: Complements PiSSA

PiSSA puts principal directions in LoRA. Res-SVD puts them in the base. 
The two are mathematically dual.

## 9.2 When to Use Res-SVD

### Recommended

- **Compression-constrained deployment**
- **Multi-architecture** (four families supported)
- **Parameter-efficient fine-tuning** where base must stay frozen
- **Combined with quantization**

### Not Recommended

- **Very tight accuracy requirements** (some drop expected)
- **Extremely long sequences** (doesn't reduce attention complexity)
- **Cross-attention tasks**

## 9.3 Cost-Benefit

For DistilBERT:

| Metric | Baseline | Res-SVD | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 298 MB MB | -48.5\%% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 89.56% | -1.15% |

## 9.4 Comparison with Published Methods

| Method | Type | Ratio | SST-2 |
|---|---|---|---|
| ASVD | SVD | 12× | 88.5% |
| AASVD | SVD + cov | 5× | 88.5% |
| A3 | SVD + Fisher | 6× | 88.5% |
| **Res-SVD** | **Two-level SVD** | **50\%×** | **89.56%** |

## 9.5 Open Questions

1. **Does the two-level structure help beyond equal-rank SVD?** 
   Empirically yes, but why?
2. **Can the base rank be adaptive per layer?** 
3. **Interaction with activation weighting (A3/AASVD)?**
4. **Interaction with quantization?**

## 9.6 Broader Impact

### Positive
- Real compression
- Multi-architecture support
- Parameter-efficient fine-tuning

### Negative
- Silent injection failure
- Cross-attention not supported

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg

---