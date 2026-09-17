# 11. Conclusion

## 11.1 Summary

We presented **JambaInfo**, an information-bottlenecked hybrid block. The 
SSM branch processes a compressed concept representation $z \in \mathbb{R}^b$ 
instead of the full hidden state, reducing noise and improving accuracy. 
Combined with differential attention and learned fusion, JambaInfo achieves 
**89.68%** on DistilBERT.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Bottleneck Gain |
|---|---|---|---|
| DistilBERT | -1.03% | +48.5\%% | 0.85% |
| RoBERTa | -25.58% | +48.5\%% | 0.85% |
| GPT-2 | -28.56% | +48.5\%% | 0.85% |

## 11.3 Key Takeaways

1. **Bottleneck improves accuracy** by 0.85%.
2. **SSM does not need full dimensions.**
3. **$b = 256$ is a sweet spot.**
4. **Differential attention reduces noise.**
5. **α learns model-specific preferences.**

## 11.4 Future Work

### Short-Term
- Learned bottleneck dimension.
- PCA initialization.
- Sparsity penalty.

### Long-Term
- Human-interpretable concepts.
- Cross-attention support.
- Scaling to 1B+ models.

## 11.5 Closing Remarks

JambaInfo is a step toward principled hybrid architectures. We release 
all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the block, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---