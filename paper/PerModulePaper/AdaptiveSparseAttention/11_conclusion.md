<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Adaptive Sparse Attention** (ASA), a per-head learnable 
sparsification mechanism for transformer attention. Each head maintains a 
scalar logit that controls its retention budget, and top-$k$ selection is 
made differentiable via a sigmoid soft mask. Chunked computation bounds 
peak memory.

## 11.2 Results Recap

| Model | Acc Δ | Sparsity | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{D_SP}} | -{{MEM_REDUCTION}}× |
| RoBERTa | {{R_DELTA}}% | {{R_SP}} | -{{MEM_REDUCTION}}× |
| GPT-2 | {{G_DELTA}}% | {{G_SP}} | -{{MEM_REDUCTION}}× |

## 11.3 Key Takeaways

1. **Learned per-head budgets are bimodal**: dense vs sparse.
2. **Soft top-$k$ is differentiable and effective.**
3. **Chunking bounds memory** without hurting quality.
4. **No parameter reduction** — compute-only optimization.
5. **Attention is highly concentrated**: top 10% of keys captures {{MASS10}}%.

## 11.4 Future Work

### Short-Term
- Learned chunk size
- Cross-layer sparsity transfer
- Fused kernels for sparse matmul
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Integration with GQA and compression
- Cross-task transfer

## 11.5 Closing Remarks

ASA provides a flexible, interpretable sparsity mechanism for transformer 
attention. We release code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed ASA, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.
