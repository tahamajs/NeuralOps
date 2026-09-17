# 11. Conclusion

## 11.1 Summary

We presented **ClusterFormer**, a differentiable K-means clustering 
attention mechanism that achieves $\mathcal{O}(KLd)$ complexity while 
preserving global context. The method combines centroid-to-key 
cross-attention with local sliding-window blending, controlled by learnable 
temperature and blend scalars.

## 11.2 Results Recap

| Model | Acc Δ | FLOPs reduction ($L=256$) |
|---|---|---|
| DistilBERT | -0.69% | ~2.7× |
| RoBERTa | -6.54% | ~2.7× |
| GPT-2 | +8.60% | ~2.7× |

## 11.3 Key Takeaways

1. **Linear attention without random features** is possible via clustering.
2. **Global + local blending** is essential.
3. **Learnable hyperparameters** (τ, α) improve results marginally.
4. **Not a compression method** — parameter count increases.

## 11.4 Future Work

### Short-Term
- Fused Triton kernel for K-means
- Learned K per layer
- Long-context evaluation ($L \geq 4096$)

### Long-Term
- Causal generation support
- Theory: bound on approximation error
- Combination with quantization

## 11.5 Closing Remarks

ClusterFormer is a practical linear attention method for long sequences. 
We release code and checkpoints under MIT.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments.
- **Collaborative Research Team** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---