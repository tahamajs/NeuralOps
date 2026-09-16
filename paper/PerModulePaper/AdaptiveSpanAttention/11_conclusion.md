<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Adaptive Span Attention** (ASA-Span), a learnable per-head 
window mechanism for transformer attention. Each head maintains a scalar 
logit that controls its span, masked via a continuous soft ramp. An L1 penalty 
encourages span shrinkage, achieving automatic pruning.

## 11.2 Results Recap

| Model | Acc Δ | Learned span | Reduction |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_SPAN}}** | {{D_SPAN_REDUCTION}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_SPAN}}** | {{R_SPAN_REDUCTION}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_SPAN}}** | {{G_SPAN_REDUCTION}}% |

## 11.3 Key Takeaways

1. **Learned spans are bimodal**: local vs global heads.
2. **Layer depth correlates with span**: early = larger, late = smaller.
3. **L1 regularization shrinks spans** to the minimum needed.
4. **No speedup without hard truncation.**
5. **Interpretability is the main benefit.**

## 11.4 Future Work

### Short-Term
- Learn ramp size per-head
- Hard truncation + sparse kernels
- Combine with content-based sparsity
- Long-context evaluation

### Long-Term
- Theoretical bounds on span vs accuracy
- Cross-task span transfer
- Integration with compression

## 11.5 Closing Remarks

ASA-Span provides an interpretable mechanism for learning per-head attention 
windows. We release code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the implementation, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.
```

---