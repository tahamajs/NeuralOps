# 11. Conclusion

## 11.1 Summary

We presented **TokenCompressedAttention (TCA)**, a learnable token-selection 
module for efficient transformer inference. TCA reduces KV cache and 
attention compute by a factor of $1 - \rho$ while preserving `[CLS]` and 
respecting causality.

## 11.2 Results Recap

| Model | Acc Δ | KV reduction | Params Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{KV_PCT}}% | +{{D_PARAM_PCT}}% |
| RoBERTa | {{R_DELTA}}% | {{KV_PCT}}% | +{{R_PARAM_PCT}}% |
| GPT-2 | {{G_DELTA}}% | {{KV_PCT}}% | +{{G_PARAM_PCT}}% |

## 11.3 Key Takeaways

1. **KV reduction is real** ({{KV_PCT}}% at ρ = {{RATIO}}).
2. **Learned scoring beats fixed heuristics** by {{SCORE_NET_GAIN}}%.
3. **[CLS] pinning matters** ({{CLS_DROP}}% drop without).
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

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed TCA, implemented, ran experiments, wrote paper.
- **{{AUTHOR_2}}** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---