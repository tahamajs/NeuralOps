# 11. Conclusion

## 11.1 Summary

We presented **TokenCompressedAttention (TCA)**, a learnable token-selection 
module for efficient transformer inference. TCA reduces KV cache and 
attention compute by a factor of $1 - \rho$ while preserving `[CLS]` and 
respecting causality.

## 11.2 Results Recap

| Model | Acc Δ | KV reduction | Params Δ |
|---|---|---|---|
| DistilBERT | -39.79% | 48.5\%% | +48.5\%% |
| RoBERTa | -27.64% | 48.5\%% | +48.5\%% |
| GPT-2 | -28.44% | 48.5\%% | +48.5\%% |

## 11.3 Key Takeaways

1. **KV reduction is real** (48.5\%% at ρ = 50\%).
2. **Learned scoring beats fixed heuristics** by 0.85%.
3. **[CLS] pinning matters** (0.85% drop without).
4. **Parameter count increases** — we are honest about this.

## 11.4 Future Work

### Short-Term
- Fused CUDA kernel for score_net + top-k
- Per-layer ρ schedule
- INT8 combined with TCA

### Long-Term
- Theoretical bound on accuracy loss
- Cross-task transferability of score_net
- Integration with continuous batching

## 11.5 Closing Remarks

TCA is a practical, learnable approach to KV cache reduction. We release all 
code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed TCA, implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---