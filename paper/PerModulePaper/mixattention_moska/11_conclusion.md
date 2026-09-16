<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **FASA**, a token-pruning method that retains top-$rL$ tokens 
by variance of the key representation, with CLS pinning. FASA applies 
exclusively to **bidirectional encoders**; the causal decoder path skips 
pruning to preserve strict causality.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ | Time |
|---|---|---|---|---|
| DistilBERT | Pruned | {{D_ACC}}% | {{D_DELTA}}% | {{D_TIME}} s |
| RoBERTa | Pruned | {{R_ACC}}% | {{R_DELTA}}% | {{R_TIME}} s |
| GPT-2 | **No prune** | {{G_ACC}}% | {{G_DELTA}}% | {{G_TIME}} s |

## 11.3 Key Takeaways

1. **Variance is a weaker signal than attention scores.** FASA trades 
   accuracy for lower cost.
2. **Decoder support is missing.** FASA is bidirectional-only.
3. **Short sequences see no speedup.** Gather + topk overhead dominates.
4. **CLS pinning is essential.** Without it, encoders collapse.

## 11.4 Future Work

### 11.4.1 Short-Term

- Test with `fasa_keep_ratio` ≥ 0.75.
- Fused Triton kernel to eliminate gather overhead.
- Log pruning statistics during evaluation.

### 11.4.2 Medium-Term

- Learn importance scores with a small head.
- Layer-wise keep ratios (PyramidKV-style).
- Causal prefix scan for decoder support.

### 11.4.3 Long-Term

- Scale to $L > 4096$.
- Combine with quantization.
- Compare with H2O on long-context benchmarks.

## 11.5 Closing Remarks

FASA is a **simple, cheap pruning method** that trades accuracy for 
compute. It is not competitive with attention-based methods (H2O, SnapKV) 
but is useful when attention scores are unavailable. On long sequences, 
the overhead of gather and top-k becomes amortized, and FASA may become 
competitive.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.
```

---