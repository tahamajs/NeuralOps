<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **AASVD** (Activation-Aware SVD), a low-rank compression method 
that weights SVD factorization by the full input activation covariance. AASVD 
uses streaming EMA accumulation, achieves real parameter reduction 
($d_\text{in} d_\text{out} \to r(d_\text{in} + d_\text{out})$), and includes 
a bundled multi-domain calibration corpus.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -1.26% | **4.0×** | -40% |
| RoBERTa | +9.51% | **3.5×** | -40% |
| GPT-2 | +1.37% | **3.5×** | -40% |

## 11.3 Key Takeaways

1. **Real compression works.**
2. **Full covariance beats diagonal** by 1.5%.
3. **Bundled corpus is sufficient** for out-of-the-box use.
4. **EMA with β=0.9 is robust.**

## 11.4 Future Work

### Short-Term
- Fused Triton kernels
- Multi-task calibration
- INT8 combined with AASVD
- Learned rank schedule

### Long-Term
- Theoretical error propagation bounds
- Cross-lingual calibration
- Recursive factorization

## 11.5 Closing Remarks

AASVD is a practical, honest model compression method. We release all code 
and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed AASVD, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
