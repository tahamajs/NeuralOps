<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **OHA** (Optimized Hybrid Attention), a drop-in replacement 
for MHSA that combines sliding-window local attention with a 
**sequence-compressed global branch**, fused via per-head gating. The 
compressed summary is computed on-the-fly from the input, making it 
data-adaptive.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ % | Time ratio (L=128) |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_TIME_RATIO}}× |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_TIME_RATIO}}× |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_TIME_RATIO}}× |

At longer sequences, the efficiency benefit dominates.

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase.
2. **Long-context benefit.** $\mathcal{O}(L(W + d_c))$ vs $\mathcal{O}(L^2)$.
3. **Data-adaptive compression.** Input-dependent summary.
4. **Multi-architecture.** Same code, four families.

## 11.4 Future Work

### Short-Term
- Cross-attention support
- Better compression initialization
- Fused Triton kernels

### Long-Term
- Theoretical analysis
- Needle-in-haystack evaluation
- Comparison with Set Transformer

## 11.5 Closing Remarks

OHA is a practical tool for long-context inference with data-adaptive 
compression. We release all code and checkpoints.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed OHA, implemented the code, ran experiments, 
  wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---