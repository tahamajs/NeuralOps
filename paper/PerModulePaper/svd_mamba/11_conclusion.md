<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **SVD-Mamba**, a hybrid attention-SSM block that combines 
SVD-factorized attention projections with a Mamba-style selective SSM and 
a global mixing term. The branches are fused via a gate initialized with 
bias $-7$ to keep the SSM silent at start.

## 11.2 Results Recap

| Model | Acc Δ | Param Δ % | Time ratio (L=128) |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_TIME_RATIO}}× |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_TIME_RATIO}}× |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_TIME_RATIO}}× |

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase by ~{{D_PARAM_PCT}}%.
2. **New capability.** SSM path adds linear-time recurrence.
3. **Gate init is critical.**
4. **Numerical stability requires multiple clamps.**

## 11.4 Future Work

### Short-Term
- Stabilize the gate dynamics
- Test on long-context tasks
- Combine with AASVD compression

### Long-Term
- Pretrain the SSM branch
- Theoretical analysis
- Scaling to >1B models

## 11.5 Closing Remarks

SVD-Mamba is an exploration of hybrid attention-SSM. It is not a 
compressor. We release all code and checkpoints.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed SVD-Mamba, implemented the code, ran 
  experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---