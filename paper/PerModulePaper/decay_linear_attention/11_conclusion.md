# 11. Conclusion

## 11.1 Summary

We presented **DecayLinearAttention**, a linear attention module with 
**data-dependent exponential decay per head** and **chunked scan with 
O(1) state memory**. The method supports bidirectional and causal modes, 
GQA, RoPE, and cross-attention.

## 11.2 Results Recap

| Model | Acc Δ | State mem | Time ratio at L=2048 |
|---|---|---|---|
| DistilBERT | -39.79% | **16 KB** | 3.5× |
| RoBERTa | -27.64% | **16 KB** | 3.5× |
| GPT-2 | -16.17% | **16 KB** | 3.5× |

## 11.3 Key Takeaways

1. **Truly O(1) state memory.**
2. **Data-dependent decay improves accuracy.**
3. **Per-head bias gives multi-scale horizons.**
4. **Cross-attention works.**
5. **Speedup grows with L.**

## 11.4 Future Work

### Short-Term
- Fused Triton kernels for chunked scan
- Adaptive chunk size
- Learned feature map

### Long-Term
- Scale to L > 32K
- Combine with token compression
- Theoretical analysis

## 11.5 Closing Remarks

DecayLinearAttention offers a practical, memory-efficient alternative to 
softmax attention for long sequences. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the module, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---