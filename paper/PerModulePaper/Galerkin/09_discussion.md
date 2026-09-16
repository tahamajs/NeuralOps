<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Fixed Bases Are Insufficient

Learned attention is highly adaptive. A fixed basis — DCT, Fourier, or 
wavelet — cannot capture its structure. **Learnable** projections 
(Linformer, Nyströmformer) are more promising.

### 9.1.2 Takeaway 2: Encoders and Decoders Behave Differently

RoBERTa (12 layers, nested) collapses to chance. DistilBERT (6 layers, 
flat) survives with -25%. The depth amplifies error compounding.

### 9.1.3 Takeaway 3: Fallback ≠ Success

GPT-2's 0% delta is not a success; it is a **no-op**. The module does not 
change anything for decoders.

## 9.2 When to Use Galerkin Attention

### 9.2.1 Recommended

**None.** We do not recommend this module for any of the tested use cases.

### 9.2.2 Possibly Worth Exploring

- **As a diagnostic tool** for understanding attention structure.
- **As a baseline** for spectral attention research.
- **In hybrid learned-fixed settings** where DCT is combined with learned 
  components.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you want to reduce attention memory? ── Yes ── Use FlashAttention-2
  │                                                (not Galerkin)
  No
  ▼
Do you want to compress parameters? ── Yes ── Use A3 or SVD
  │
  No
  ▼
Do you want to explore fixed-basis attention? ── Yes ── Use this module
  │                                                     (as research)
  No
  ▼
Consider alternatives
```

## 9.4 Comparison with Alternatives

| Alternative | Compresses? | Encoder OK? | Decoder OK? | Recommended? |
|---|---|---|---|---|
| Standard MHSA | No | Yes | Yes | Yes |
| FlashAttention-2 | No (memory) | Yes | Yes | Yes |
| Linformer | Yes | Yes | Yes | Yes |
| FNet | No | Marginal | Yes | Marginal |
| **Galerkin** | **No** | **No** | **No** | **No** |

## 9.5 Cost-Benefit Analysis

For DistilBERT on RTX 4070:

| Metric | Baseline | Galerkin | Trade-off |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | **-{{D_DROP}}%** |
| Params | 66.95M | {{D_PARAMS}} | +0.88% |
| Inference VRAM | 183 MB | {{D_VRAM}} MB | +{{D_VRAM_DELTA}}% |
| Inference time | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× |

**Verdict.** Strictly worse than baseline.

## 9.6 Open Questions

1. **Would learned bases work?** (Linformer says yes)
2. **Would data-dependent DCT (per-batch) work?** Unknown.
3. **Would a hybrid DCT + learned projection work?** Untested.
4. **Would smaller rank work on tiny models?** Untested.

## 9.7 Broader Impact

### 9.7.1 Positive

- Documents a failure mode for the community.
- Provides a reproducible baseline for spectral attention.
- Warns against fixed-basis projection.

### 9.7.2 Negative

- May have consumed compute without producing a usable model.

### 9.7.3 Environmental

Total compute: ~{{COMPUTE_HOURS}} GPU-hours, ~{{CO2}} kg CO₂eq.

## 9.8 Deployment Recommendations

**None.** Do not deploy Galerkin Attention in production.

## 9.9 What We Would Do Differently

1. **Start with a learned basis** (Linformer-style).
2. **Test on a small model first** before scaling.
3. **Measure attention sharpness** before designing projection.
4. **Consider that attention is fundamentally data-dependent.**
```

---