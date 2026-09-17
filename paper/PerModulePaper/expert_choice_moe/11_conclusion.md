# 11. Conclusion

## 11.1 Summary

We presented **ExpertChoiceMoEAttention**, an attention block that adds an 
expert-choice MoE layer on top of standard attention. Expert-choice routing 
guarantees perfect load balancing across experts, requires no auxiliary 
loss, and enables dynamic per-token capacity. Combined zero-init experts 
and a negative gate bias ensure the augmented attention is numerically 
equivalent to the base at step 0.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Expert balance |
|---|---|---|---|
| DistilBERT | -0.57% | +48.5\%% | Perfect |
| RoBERTa | +5.73% | +48.5\%% | Perfect |
| GPT-2 | +1.14% | +48.5\%% | Perfect |

## 11.3 Key Takeaways

1. **Expert-choice routing guarantees load balance.**
2. **Dynamic per-token capacity is enabled.**
3. **Zero-init stability is essential.**
4. **Parameter increase (+807%) is the main limitation.**
5. **No auxiliary loss needed.**

## 11.4 Future Work

### Short-Term
- Batched expert forward (no Python loop)
- Learned capacity factor
- INT8 experts

### Long-Term
- Parameter-efficient variants (shared experts)
- Larger-scale evaluation
- Hybrid with token-choice for variable capacity

## 11.5 Closing Remarks

ExpertChoiceMoEAttention shows that expert-choice routing can be applied 
inside attention with guaranteed load balance. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the module, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---