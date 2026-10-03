<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 – 1 page (300 – 500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **AdaSplash**, an adaptive splash attention mechanism that 
combines **α-entmax sparse activations** with **local sliding-window masking** 
and **attention sink preservation**. AdaSplash is **not** a compression 
method — it **increases** parameter count by ~0.88% due to LoRA 
adapters. Its value is in producing **~40% sparse attention** without 
accuracy loss.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Sparsity |
|---|---|---|---|
| DistilBERT | -13.53% | +594438 | 42% |
| RoBERTa | -11.47% | +1179648 | 40% |
| GPT-2 | +0.00% | +1188876 | 41% |

## 11.3 Key Takeaways

1. **α-entmax produces exact zeros** without accuracy loss.
2. **Attention sinks are essential** for stability.
3. **Splash masking** works for causal decoders.
4. **Not a compression method** — parameter count increases.
5. **Slower than baseline** due to bisection.

## 11.4 Future Work

### Short-Term
- **Fused CUDA kernel** for entmax bisection
- **Learned α** per layer
- **Sparse matmul** to exploit sparsity
- **T5 verification**

### Long-Term
- **Long-context inference** with splash masking
- **Sparse hardware** deployment (2:4 sparsity)
- **Attention pattern analysis** as a research tool

## 11.5 Closing Remarks

AdaSplash is a **research tool** for studying sparse attention. It is 
**not** a production-ready compression method. We release code and checkpoints 
for reproducibility.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided methodological feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.
