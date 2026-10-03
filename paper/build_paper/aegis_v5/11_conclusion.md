# 11. Conclusion

## 11.1 Summary

We presented **AEGIS-V5**, a hybrid attention module combining local 
sliding-window attention with global kernel linear attention through a 
split-head architecture. Base projections are SVD-factorized with energy-
based rank selection and frozen; LoRA adapters provide trainable capacity. 
Per-head gates fuse local and global branches.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | VRAM Δ |
|---|---|---|---|
| DistilBERT | -6.99% | 1.8% | 380 MB% |
| RoBERTa | +7.91% | 1.9% | 380 MB% |
| GPT-2 | -9.86% | 2.0% | 380 MB% |

## 11.3 Key Takeaways

1. **Split-head architecture works.**
2. **Softpick fix prevents collapse** (V4 vs V5).
3. **Parameter effect depends on `max_rank`.**
4. **Decoder fallback is essential** for causality.

## 11.4 Future Work

### Short-Term
- Fused Triton kernels for local + kernel attention
- Adaptive per-layer SVD rank
- INT8 + AEGIS-V5

### Long-Term
- Causal-safe hybrid attention
- Theoretical analysis of head split
- Cross-task transferability

## 11.5 Closing Remarks

Code and checkpoints released.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed AEGIS-V5, implemented, ran experiments, wrote 
  paper.
- **Collaborative Research Team** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---