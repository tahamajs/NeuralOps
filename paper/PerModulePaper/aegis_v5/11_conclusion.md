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
| DistilBERT | {{D_DELTA}}% | {{D_PARAM_PCT}}% | {{D_VRAM_DELTA}}% |
| RoBERTa | {{R_DELTA}}% | {{R_PARAM_PCT}}% | {{R_VRAM_DELTA}}% |
| GPT-2 | {{G_DELTA}}% | {{G_PARAM_PCT}}% | {{G_VRAM_DELTA}}% |

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

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed AEGIS-V5, implemented, ran experiments, wrote 
  paper.
- **{{AUTHOR_2}}** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---