<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Routing Attention**, a content-based sparse attention method 
that clusters tokens via online k-means on the joint query-key 
representation, then restricts attention to intra-cluster tokens plus a 
token-0 anchor. On SST-2, accuracy is preserved within **{{D_DELTA}}%** 
(DistilBERT), **{{R_DELTA}}%** (RoBERTa), and **{{G_DELTA}}%** (GPT-2).

## 11.2 Results Recap

| Model | Acc | Δ | Params Δ | Time |
|---|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_TIME_R}}× |
| RoBERTa | {{R_ACC}}% | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_TIME_R}}× |
| GPT-2 | {{G_ACC}}% | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_TIME_R}}× |

## 11.3 Key Takeaways

1. **Formulation preserves accuracy** within 1%.
2. **Implementation is slow** due to naive cluster loop.
3. **Not a compression method.**
4. **Real speedups require sparse kernels.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Implement sparse matmul via Triton.
- Exclude padding from k-means.
- Replace `RuntimeError` with warning.
- Add double-injection guard.

### 11.4.2 Long-Term

- Gumbel-Softmax for differentiable clustering.
- Query-dependent routing.
- Comparison with Reformer, Routing Transformer.

## 11.5 Closing Remarks

Routing Attention is a research prototype. We release the code honestly, 
document its current limitations, and outline a path to a production-ready 
sparse attention.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.

---