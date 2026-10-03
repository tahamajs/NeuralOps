<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **RALA** (Rank-Augmented Linear Attention), a linear-attention 
variant that combines a softplus kernel with a per-head low-rank augmentation 
and a scalar gate. RALA preserves accuracy within **-30.62%** on 
DistilBERT, **-27.76%** on RoBERTa, and **-28.44%** on GPT-2, but 
increases parameter count by **48.5\%%** and slows inference by 
**2.3 s×**.

## 11.2 Results Recap

| Model | Acc | Δ | Params Δ | Time |
|---|---|---|---|---|
| DistilBERT | 60.09% | -30.62% | +48.5\%% | 2.3 s× |
| RoBERTa | 50.80% | -27.76% | +48.5\%% | 2.3 s× |
| GPT-2 | 50.92% | -28.44% | +48.5\%% | 2.3 s× |

## 11.3 Key Takeaways

1. **Softplus is a valid kernel choice** but not clearly superior to elu+1.
2. **Rank augmentation is a modest contribution** given frozen $U, V$.
3. **The prefix scan is the bottleneck.**
4. **RALA is not a compression method.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Fused parallel scan (Triton).
- Unfrozen $U, V$.
- Per-head gate.

### 11.4.2 Long-Term

- Compare with Delta Net, GLA, Mamba.
- Test on generation tasks.
- Scale to 7B+.

## 11.5 Closing Remarks

RALA is an experimental contribution to linear attention literature. It is 
**not** production-ready and **not** a compression method. We release the 
code to enable replication and improvement.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** implemented RALA, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---