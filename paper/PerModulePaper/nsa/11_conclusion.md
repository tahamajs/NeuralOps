<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Native Sparse Attention** (NSA), which combines a fixed 
sliding window with dynamic top-$k$ global routing. The mask is a union of 
the two, applied before softmax. The implementation is fully vectorized, 
parameter-free, and compatible with four pretrained architectures.

## 11.2 Results Recap

| Model | Acc Δ | Density |
|---|---|---|
| DistilBERT | +0.12% | 50\%% |
| RoBERTa | -27.64% | 0.85% |
| GPT-2 | -28.44% | 0.85% |

## 11.3 Key Takeaways

1. **Union of local + global** captures both regimes.
2. **Zero parameters added.**
3. **Density drops with sequence length.**
4. **Selection is content-dependent.**
5. **No time speedup without sparse kernels.**

## 11.4 Future Work

### Short-Term
- Triton kernels for sparse matmul
- Learned window and top-k per head
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Cross-layer selection transfer
- Integration with quantization and MoE

## 11.5 Closing Remarks

NSA provides a simple, effective, and parameter-free sparse attention 
mechanism for long-context inference.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed NSA, implemented the code, wrote the paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---