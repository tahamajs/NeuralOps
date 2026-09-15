<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **MinimalSVD**, a compact, auditable method for compressing 
transformer attention projections via SVD factorization with energy-adaptive 
rank selection. MinimalSVD replaces each dense projection 
$W \in \mathbb{R}^{d \times d}$ with a product $U V$, where the rank is 
selected per layer based on spectral energy.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_RATIO}}×** | -{{VRAM_REDUCTION}}% |

## 11.3 Key Takeaways

1. **Real parameter reduction works.**
2. **Energy-adaptive rank is effective.**
3. **No calibration needed.**
4. **Minimal code enables auditability.**
5. **Companion variants achieve the same accuracy** — engineering 
   differences only.

## 11.4 Future Work

### Short-Term
- Fused Triton kernels
- Frobenius validation per layer
- INT8 combination

### Long-Term
- Theoretical error propagation bounds
- Learned rank schedules
- Extension to FFN projections

## 11.5 Closing Remarks

MinimalSVD demonstrates that training-free, multi-architecture attention 
compression is feasible with minimal code. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed MinimalSVD, implemented the code, ran experiments, 
  wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---
