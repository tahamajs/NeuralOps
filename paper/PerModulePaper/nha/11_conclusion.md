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
| DistilBERT | {{D_DELTA}}% | $\mathcal{O}(L(w+m))$ | {{D_GATE}} |
| RoBERTa | {{R_DELTA}}% | $\mathcal{O}(L(w+m))$ | {{R_GATE}} |
| GPT-2 | {{G_DELTA}}% | $\mathcal{O}(L(w+m))$ | {{G_GATE}} |

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

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed NHA, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---