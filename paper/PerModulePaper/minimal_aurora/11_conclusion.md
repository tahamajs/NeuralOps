# 11. Conclusion

## 11.1 Summary

We presented **MinimalAURORA**, an energy-preserving SVD-factorized 
attention module that reduces parameter count by {{D_RATIO}}× while 
maintaining accuracy within {{D_DELTA}}%.

## 11.2 Results Recap

| Model | Acc Δ | Ratio |
|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_RATIO}}×** |
| RoBERTa | {{R_DELTA}}% | **{{R_RATIO}}×** |
| GPT-2 | {{G_DELTA}}% | **{{G_RATIO}}×** |

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

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed MinimalAURORA, implemented the code, ran 
  experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.
```

---