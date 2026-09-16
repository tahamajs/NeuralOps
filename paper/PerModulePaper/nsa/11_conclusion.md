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
| DistilBERT | {{D_DELTA}}% | {{D_DENSITY}}% |
| RoBERTa | {{R_DELTA}}% | {{R_DENSITY}}% |
| GPT-2 | {{G_DELTA}}% | {{G_DENSITY}}% |

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

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed NSA, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.
```

---