<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **RALA** (Rank-Augmented Linear Attention), a linear-attention 
variant that combines a softplus kernel with a per-head low-rank augmentation 
and a scalar gate. RALA preserves accuracy within **{{D_DELTA}}%** on 
DistilBERT, **{{R_DELTA}}%** on RoBERTa, and **{{G_DELTA}}%** on GPT-2, but 
increases parameter count by **{{D_PARAM_PCT}}%** and slows inference by 
**{{D_TIME_R}}×**.

## 11.2 Results Recap

| Model | Acc | Δ | Params Δ | Time |
|---|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_TIME_R}}× |
| RoBERTa | {{R_ACC}}% | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_TIME_R}}× |
| GPT-2 | {{G_ACC}}% | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_TIME_R}}× |

## 11.3 Key Takeaways

1. **Softplus is a valid kernel choice** but not clearly superior to elu+1.
2. **Rank augmentation is a modest contribution** given frozen $U, V$.
3. **The prefix scan is the bottleneck.**
4. **RALA is not a compression method.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Fused parallel scan (Triton).
- Unfrozen $U, V$.
- Per-head gate.

### 11.4.2 Long-Term

- Compare with Delta Net, GLA, Mamba.
- Test on generation tasks.
- Scale to 7B+.

## 11.5 Closing Remarks

RALA is an experimental contribution to linear attention literature. It is 
**not** production-ready and **not** a compression method. We release the 
code to enable replication and improvement.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented RALA, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.

---