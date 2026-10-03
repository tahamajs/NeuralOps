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
The method achieves **50\%×–2.1\times×** parameter reduction with 
**-18.12%–-28.44%** accuracy drop on SST-2.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -18.12% | **50\%×** | -380 MB% |
| RoBERTa | -27.64% | **2.1\times×** | -380 MB% |
| GPT-2 | -28.44% | **2.1\times×** | -380 MB% |

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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed PHANTOM, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---