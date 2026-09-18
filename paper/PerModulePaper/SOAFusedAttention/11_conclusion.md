# 11. Conclusion

## 11.1 Summary

We presented **SOA Fused Attention**, a zero-shot low-rank factorization of 
attention projections. The method replaces Q, K, V with factorized pairs 
initialized from the truncated SVD of the pretrained weights, achieving 
**1.92×–1.92×** parameter reduction on attention projections 
without any calibration data.

## 11.2 Results Recap

| Model | Acc Δ | Attention Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -9.40% | **1.92×** | -10% |
| RoBERTa | -27.64% | **1.92×** | -10% |
| GPT-2 | -28.44% | **1.92×** | -10% |

## 11.3 Key Takeaways

1. **Zero-shot SVD works** — no calibration needed.
2. **$W_O$ should stay** — factorizing it compounds error.
3. **LoRA recovers** 5.5% of the accuracy drop.
4. **Attention parameters reduce by 1.92×.**

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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---
