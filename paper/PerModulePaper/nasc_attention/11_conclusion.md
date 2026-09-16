<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **NASC**, a three-branch hybrid attention that combines 
sliding-window, sparse block, and spectral RFF attention under a batch-level 
softmax gate. NASC achieves **{{D_ACC}}%** on DistilBERT, **{{R_ACC}}%** 
on RoBERTa, and **{{G_ACC}}%** on GPT-2.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ | Time |
|---|---|---|---|---|
| DistilBERT | Dense | {{D_ACC}}% | {{D_DELTA}}% | {{D_TIME}} s |
| RoBERTa | Dense | {{R_ACC}}% | {{R_DELTA}}% | {{R_TIME}} s |
| GPT-2 | **Cascade** | {{G_ACC}}% | {{G_DELTA}}% | {{G_TIME}} s |

## 11.3 Key Takeaways

1. **Three branches provide redundancy.** All three contribute.
2. **Batch-level gate is a simplification.** Per-token would be more 
   adaptive.
3. **Encoder default is dense.** Cascade is decoder-only by default.
4. **Overhead is significant.** ~{{G_TIME_R}}× slower.
5. **RFF causal cumsum is memory-heavy.** OOM at long sequences.

## 11.4 Future Work

### 11.4.1 Short-Term

- Test encoder cascade with `enable_cascade_on_encoders=True`.
- Vectorize sparse block loop.
- Fused Triton kernel for RFF.

### 11.4.2 Medium-Term

- Per-token gating instead of batch-level.
- Per-row `softpick` fallback.
- Causal RFF without materializing cumsum.

### 11.4.3 Long-Term

- Scale to $L > 4096$.
- Combine with quantization.
- Test on Qwen3.5.

## 11.5 Closing Remarks

NASC is a **sophisticated three-branch hybrid attention**. It works 
correctly on decoders and provides a cascade path for long-context 
scenarios. Its limitations — batch-level gate, RFF memory, and Python loop 
overhead — are implementation constraints, not fundamental. Future work 
should address these to unlock NASC's full potential.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.

---