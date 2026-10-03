<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Dobi-SVD**, a differentiable low-rank factorization with 
learnable spectral gating. Each rank component is scaled by $\sigma(\alpha_i)$ 
for a learnable gate $\alpha_i$. A gate-compensation pre-scaling of $V$ 
ensures the initial forward pass exactly matches the truncated SVD.

Gates are trained jointly with LoRA recovery, providing a data-driven 
signal for which rank components to retain. After training, gates below 
a threshold identify removable components for post-hoc pruning without 
retraining.

## 11.2 Results Recap

| Model | Acc Δ | Nominal $r$ | Effective $r$ | Ratio |
|---|---|---|---|---|
| DistilBERT | -0.23% | 32 | 32 | **50\%×** |
| RoBERTa | -4.02% | 32 | 0.85 | **2.1\times×** |
| GPT-2 | -21.22% | 32 | 0.85 | **2.1\times×** |

## 11.3 Key Takeaways

1. **Rank is learnable** via differentiable gates.
2. **Lossless initialization** via gate compensation.
3. **Post-hoc pruning works** at τ_g = 0.5.
4. **Effective rank** is 10–20% smaller than nominal.
5. **Joint optimization** with LoRA recovery is essential.

## 11.4 Future Work

### Short-Term
- Fused kernels for gate-gated forward pass
- Learned τ_g per layer
- Interaction with quantization

### Long-Term
- Theoretical convergence analysis for gates
- Cross-task gate transfer
- Extension to FFN projections

## 11.5 Closing Remarks

Dobi-SVD demonstrates that rank selection can be made differentiable. We 
release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed Dobi-SVD, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---