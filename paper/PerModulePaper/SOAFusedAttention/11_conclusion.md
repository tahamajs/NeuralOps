# 11. Conclusion

## 11.1 Summary

We presented **SOA Fused Attention**, a zero-shot low-rank factorization of 
attention projections. The method replaces Q, K, V with factorized pairs 
initialized from the truncated SVD of the pretrained weights, achieving 
**{{D_RATIO}}×–{{G_RATIO}}×** parameter reduction on attention projections 
without any calibration data.

## 11.2 Results Recap

| Model | Acc Δ | Attention Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_RATIO}}×** | -{{VRAM_REDUCTION}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_RATIO}}×** | -{{VRAM_REDUCTION}}% |

## 11.3 Key Takeaways

1. **Zero-shot SVD works** — no calibration needed.
2. **$W_O$ should stay** — factorizing it compounds error.
3. **LoRA recovers** {{LORA_GAIN}}% of the accuracy drop.
4. **Attention parameters reduce by {{D_RATIO}}×.**

## 11.4 Future Work

### Short-Term
- Per-layer adaptive rank selection
- Fused kernels for factorized attention
- Combined with KV quantization

### Long-Term
- Composition with token compression
- Extension to cross-attention
- Theoretical error propagation bounds

## 11.5 Closing Remarks

SOA Fused Attention is a practical, zero-shot attention compressor. We 
release all code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented, ran experiments, wrote 
  the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---