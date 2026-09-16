<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **HALO-ReGLA**, a hybrid attention module that runs dense 
softmax and rectified linear streams in parallel, combined by per-head 
learnable gates. The gate is initialized to favor dense behavior and shifts 
toward linear during training.

## 11.2 Results Recap

| Model | Acc Δ | Mean gate | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{D_GATE}} | -{{D_VRAM_DELTA}}% |
| RoBERTa | {{R_DELTA}}% | {{R_GATE}} | -{{R_VRAM_DELTA}}% |
| GPT-2 | {{G_DELTA}}% | {{G_GATE}} | -{{G_VRAM_DELTA}}% |

## 11.3 Key Takeaways

1. **Learned gates are bimodal**: dense vs linear heads.
2. **Layer depth correlates with gate.**
3. **Learned fusion beats fixed fusion.**
4. **ELU+1 is optimal kernel.**
5. **No speedup without hard routing.**

## 11.4 Future Work

### Short-Term
- Hard routing via thresholded gates
- Chunked cumsum for long sequences
- Learned kernels
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Integration with GQA / KV cache
- Cross-task gate transfer

## 11.5 Closing Remarks

HALO-ReGLA provides an interpretable hybrid attention mechanism. We release 
code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed HALO-ReGLA, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---