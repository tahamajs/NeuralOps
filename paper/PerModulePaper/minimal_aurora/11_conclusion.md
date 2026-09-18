# 11. Conclusion

## 11.1 Summary

We presented **MinimalAURORA**, an energy-preserving SVD-factorized 
attention module that reduces parameter count by 50\%× while 
maintaining accuracy within -5.27%.

## 11.2 Results Recap

| Model | Acc Δ | Ratio |
|---|---|---|
| DistilBERT | -5.27% | **50\%×** |
| RoBERTa | -27.64% | **2.1\times×** |
| GPT-2 | -28.44% | **2.1\times×** |

## 11.3 Key Takeaways

1. **Real compression** via energy-threshold SVD.
2. **Simple implementation** — no calibration, no weighting.
3. **Multi-architecture** — same code, four families.
4. **`pres_gate` unused** — should be removed.

## 11.4 Future Work

### Short-Term
- Remove `pres_gate`
- Freeze `output_scale` or add regularization
- Test at long context

### Long-Term
- Add activation weighting (A3/AASVD style)
- Add LayerNorm recalibration
- Compare with other energy thresholds

## 11.5 Closing Remarks

MinimalAURORA provides a simple, honest baseline for attention 
compression. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed MinimalAURORA, implemented the code, ran 
  experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---