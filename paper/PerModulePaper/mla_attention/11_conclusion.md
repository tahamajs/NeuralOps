<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We reproduced **Multi-Head Latent Attention** from DeepSeek-AI (2024) and 
ported it to four transformer families: DistilBERT (flat), RoBERTa 
(nested), GPT-2 (causal Conv1D), and Qwen3.5 (GQA + RoPE). MLA replaces 
separate K and V projections with a shared low-rank latent vector, 
reducing KV-cache size by **2.1\times×–2.1\times×**.

## 11.2 Results Recap

| Model | Acc Δ | KV reduction | Params |
|---|---|---|---|
| DistilBERT | -39.79% | **2.1\times×** | 66.4MM |
| RoBERTa | -27.64% | **2.1\times×** | 124.6MM |
| GPT-2 | -22.14% | **2.1\times×** | 124.4MM |

## 11.3 Key Takeaways

1. **MLA ports successfully** to non-DeepSeek architectures.
2. **SVD-based initialization** with singular value normalization is 
   essential.
3. **RoBERTa nested encoder** requires careful `is_nested_encoder` handling.
4. **GPT-2 Conv1D** requires explicit transposition.
5. **GQA models** require mean-pooling before SVD.

## 11.4 Future Work

### Short-Term
- Long-context evaluation ($L = 2048, 8192$)
- Combined with parameter compression
- INT8 KV cache + MLA

### Long-Term
- Optimal latent dimension via spectral analysis
- Decoupled RoPE matching the original
- Extension to encoder-decoder models

## 11.5 Closing Remarks

MLA is a promising KV-cache compression technique that generalizes beyond 
its original architecture. We release all code and checkpoints for 
reproduction.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support.. This work reproduces DeepSeek-AI (2024).

## 11.7 Author Contributions

- **Taha Majs** implemented the reproduction, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---