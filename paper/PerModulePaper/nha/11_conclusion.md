<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **NHA** (Native Hybrid Attention), a module that runs local 
sliding-window attention and global RFF attention in parallel, combined via 
a learned token-level gate.

## 11.2 Results Recap

| Model | Acc Δ | Complexity | Mean gate |
|---|---|---|---|
| DistilBERT | +0.00% | $\mathcal{O}(L(w+m))$ | 0.94 |
| RoBERTa | -27.64% | $\mathcal{O}(L(w+m))$ | 0.85 |
| GPT-2 | -28.44% | $\mathcal{O}(L(w+m))$ | 0.85 |

## 11.3 Key Takeaways

1. **Linear complexity** in $L$ for both streams.
2. **Learned gates** outperform fixed fusion.
3. **Orthogonal RFF** stabilizes training.
4. **Chunked scan** preserves causality with bounded memory.
5. **No parameter reduction** — compute-only optimization.

## 11.4 Future Work

### Short-Term
- Learned window size
- Per-head RFF count
- Fused kernels for local + RFF
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Cross-task gate transfer
- Integration with quantization

## 11.5 Closing Remarks

NHA provides an interpretable hybrid attention mechanism for linear-time 
inference. We release code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed NHA, implemented the code, wrote the paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---