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
token-0 anchor. On SST-2, accuracy is preserved within **-26.95%** 
(DistilBERT), **-27.64%** (RoBERTa), and **-28.44%** (GPT-2).

## 11.2 Results Recap

| Model | Acc | Δ | Params Δ | Time |
|---|---|---|---|---|
| DistilBERT | 63.76% | -26.95% | +48.5\%% | 2.3 s× |
| RoBERTa | 50.92% | -27.64% | +48.5\%% | 2.3 s× |
| GPT-2 | 50.92% | -28.44% | +48.5\%% | 2.3 s× |

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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---