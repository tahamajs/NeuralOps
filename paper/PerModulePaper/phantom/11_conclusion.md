<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **PHANTOM** (Parameter-Hashed Attention with Token-wise 
Orthogonal Mapping), which combines **leverage-score CUR factorization** for 
projection compression with **multi-round angular LSH** for sparse attention. 
The method achieves **{{D_RATIO}}×–{{G_RATIO}}×** parameter reduction with 
**{{D_DELTA}}%–{{G_DELTA}}%** accuracy drop on SST-2.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_RATIO}}×** | -{{VRAM_REDUCTION}}% |

## 11.3 Key Takeaways

1. **CUR with leverage scores** beats uniform SVD at low rank.
2. **4 LSH rounds** is optimal for LSH variance reduction.
3. **Rectified softpick** adds sparsity with minimal accuracy loss.
4. **Component interactions** produce super-additive benefits.

## 11.4 Future Work

### Short-Term
- Adaptive $R_\ell$ per layer
- Learned hash functions
- Fused LSH kernel

### Long-Term
- Extension to long-context benchmarks
- Theory of LSH + softpick
- Combined with quantization

## 11.5 Closing Remarks

PHANTOM is a step toward practical, multi-axis model compression. We release 
all code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed PHANTOM, implemented the code, ran experiments, 
  wrote the paper.
- **{{AUTHOR_2}}** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---