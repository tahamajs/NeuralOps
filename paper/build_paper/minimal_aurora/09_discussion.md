# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Real Compression

MinimalAURORA reduces parameters by 50\%×.

### 9.1.2 Takeaway 2: Simplicity

No activation weighting, no calibration, no two-level SVD. The minimal 
machinery for compression + PEFT.

### 9.1.3 Takeaway 3: Code Cleanliness

`pres_gate` is unused. Recommend removal.

## 9.2 When to Use MinimalAURORA

### Recommended

- **Compression-constrained deployment**
- **No access to calibration data**
- **Multi-architecture** (four families supported)
- **Baseline comparison** for more complex methods

### Not Recommended

- **Tight accuracy requirements**
- **Cross-attention tasks** (limited support)
- **Long-context efficiency** (no complexity reduction)

## 9.3 Decision Flowchart

```
Need real compression? ─── Yes ─── Use MinimalAURORA
  │ No
  ▼
Have calibration data? ─── Yes ─── Consider AASVD/A3
  │ No
  ▼
Need attention complexity reduction? ─── Yes ─── Use SVD-Linear
  │ No
  ▼
Use standard attention
```

## 9.4 Cost-Benefit

For DistilBERT:

| Metric | Baseline | MinimalAURORA | Change |
|---|---|---|---|
| Params | 66.95M | 66.4MM | **50\%×** |
| VRAM | 183 MB | 293 MB MB | -48.5\%% |
| Time | 1.90 s | 1.42x s | 2.1\times× |
| Accuracy | 90.71% | 85.44% | -5.27% |

## 9.5 Comparison with Companion Methods

| Method | DistilBERT | Ratio | Calibration? | Complexity |
|---|---|---|---|---|
| ASVD | 88.5% | 12× | No | Low |
| AASVD | 88.5% | 5× | Yes | High |
| A3 | 88.5% | 5× | Yes | Medium |
| Res-SVD | 88.2% | 5.9× | No | Medium |
| **MinimalAURORA** | **85.44** | **50\%×** | **No** | **Low** |

## 9.6 Open Questions

1. **Does the unused `pres_gate` matter?** Would wiring it up help?
2. **Interaction with activation weighting?** Would A3-style Fisher 
   weighting improve MinimalAURORA?
3. **LayerNorm recalibration?** Would A3-style LN rescaling help?
4. **Long-context?** No complexity reduction; how does it scale?

## 9.7 Broader Impact

### Positive
- Real compression
- Simple to implement
- Multi-architecture

### Negative
- Silent injection failure
- `pres_gate` dead code

### Environmental
- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg

---