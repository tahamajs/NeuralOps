<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **ORION**, a hybrid attention mechanism combining SVD 
factorization, head-wise bifurcation, and random Fourier features. ORION 
achieves **{{D_RED}}×** parameter reduction with **{{D_DELTA}}%** accuracy 
drop on DistilBERT, and similar results on RoBERTa and GPT-2.

## 11.2 Results Recap

| Model | Accuracy | Params | Reduction |
|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_PARAMS}}M | **{{D_RED}}×** |
| RoBERTa | {{R_ACC}}% | {{R_PARAMS}}M | **{{R_RED}}×** |
| GPT-2 | {{G_ACC}}% | {{G_PARAMS}}M | **{{G_RED}}×** |

## 11.3 Key Takeaways

1. **ORION is one of the few modules in our framework that genuinely 
   reduces parameters.**
2. **The `copy_weights` raise** and **`rff_proj` persistence** are two 
   bugs that should be fixed.
3. **Head bifurcation** is a novel idea with empirical support.

## 11.4 Future Work

### 11.4.1 Immediate Fixes

- **Replace `raise` with `logger.debug`** in `copy_weights`.
- **Set `persistent=True`** for `rff_proj`.
- **Add softpick fallback** for masked rows.

### 11.4.2 Medium-Term

- **Per-layer rank selection.**
- **Learn the head split** (2/3 vs 1/3).
- **Test on other tasks.**

### 11.4.3 Long-Term

- **Combine with quantization.**
- **Scale to 7B+ models.**
- **Fused Triton kernel** for factorized projections.

## 11.5 Closing Remarks

ORION demonstrates that combining SVD factorization with hybrid attention 
is viable. With the two bugs fixed, it would be a strong choice for 
parameter-constrained deployments.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.

---