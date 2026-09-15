<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Progressive Token Pruning** (PTP), a learnable token-selection 
module for efficient transformer attention. PTP scores each token with a 
small MLP, retains the top-$k$ fraction, and preserves [CLS] and causality.

## 11.2 Results Recap

| Model | Acc Δ | Speedup | Kept |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{D_SPEEDUP}}× | {{D_KEPT}}% |
| RoBERTa | {{R_DELTA}}% | {{R_SPEEDUP}}× | {{R_KEPT}}% |
| GPT-2 | {{G_DELTA}}% | {{G_SPEEDUP}}× | {{G_KEPT}}% |

## 11.3 Key Takeaways

1. **Token pruning is effective** at $\rho = {{KEEP_RATIO}}$.
2. **[CLS] preservation is essential.**
3. **Learned scores reveal structure.**
4. **Causality preserved.**
5. **Drop-in wrapper.**

## 11.4 Future Work

### Short-Term
- Causal score network
- Per-head scoring
- Learned keep ratio
- Long-context evaluation

### Long-Term
- Cross-layer score sharing
- Theoretical bounds
- Integration with MoE

## 11.5 Closing Remarks

PTP provides a practical, learnable, drop-in token-pruning module. We 
release code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed PTP, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.
