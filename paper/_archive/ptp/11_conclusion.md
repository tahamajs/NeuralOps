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
| DistilBERT | 0.85% | 1.42x× | 0.85% |
| RoBERTa | 0.85% | 84.5× | 0.85% |
| GPT-2 | 0.85% | 84.5× | 0.85% |

## 11.3 Key Takeaways

1. **Token pruning is effective** at $\rho = 0.5$.
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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed PTP, implemented the code, wrote the paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---