<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We introduced **Primal Attention**, an asymmetric kernel SVD formulation 
of self-attention. By allowing different landmarks for queries and keys, 
Primal Attention generalizes symmetric Nyström approximation. The pseudoinverse 
is computed via differentiable Schulz iteration.

## 11.2 Results Recap

| Model | Acc Δ | Attn mem ratio |
|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{MEM_REDUCTION}}×** |
| RoBERTa | {{R_DELTA}}% | **{{MEM_REDUCTION}}×** |
| GPT-2 | {{G_DELTA}}% | **{{MEM_REDUCTION}}×** |

## 11.3 Key Takeaways

1. **Asymmetric landmarks** give +{{ASYM_GAIN}}% over symmetric Nyström.
2. **$m = 64$** is optimal for $L = 128$.
3. **Schulz from SVD pinv** is best practice.
4. **Long $L$ benefits most.**
5. **No parameter reduction.**

## 11.4 Future Work

### Short-Term
- True causal approximation
- Learned landmark pooling
- Fused kernels
- Long-context evaluation

### Long-Term
- Theoretical error bounds
- Integration with MoE
- Cross-task transfer

## 11.5 Closing Remarks

Primal Attention provides a flexible approximation primitive for attention. 
We release code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed Primal Attention, implemented the code, wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---