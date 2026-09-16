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
| DistilBERT | {{D_DELTA}}% | ~{{D_FLOP_RED}}× |
| RoBERTa | {{R_DELTA}}% | ~{{R_FLOP_RED}}× |
| GPT-2 | {{G_DELTA}}% | ~{{G_FLOP_RED}}× |

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
We release code and checkpoints under {{LICENSE}}.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented the code, ran experiments.
- **{{AUTHOR_2}}** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---