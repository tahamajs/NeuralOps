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
| DistilBERT | -39.79% | +48.5\%% | 2.1\times× |
| RoBERTa | -27.64% | +48.5\%% | 2.1\times× |
| GPT-2 | -29.13% | +48.5\%% | 2.1\times× |

## 11.3 Key Takeaways

1. **Not a compressor.** Parameters increase by ~48.5\%%.
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

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed SVD-Mamba, implemented the code, ran 
  experiments, wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---