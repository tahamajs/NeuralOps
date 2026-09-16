<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented a **production-hardened port** of **Performer** to pretrained 
transformers. The implementation supports four architecture families with 
exact weight transfer, GQA, RoPE, and both bidirectional and causal paths.

## 11.2 Results Recap

| Model | Accuracy | Δ | Time |
|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_DELTA}}% | {{D_TIME}} s |
| RoBERTa | {{R_ACC}}% | {{R_DELTA}}% | {{R_TIME}} s |
| GPT-2 | {{G_ACC}}% | {{G_DELTA}}% | {{G_TIME}} s |

## 11.3 Key Takeaways

1. **FAVOR+ works** but with moderate accuracy drop.
2. **Numerical stabilization is essential** for FP16.
3. **The `inject_performer` raise bug** violates the docstring contract.
4. **`output_scale` is unnecessary** and should be removed.

## 11.4 Future Work

### 11.4.1 Immediate Fixes

- **Fix the `inject_performer` raise** to return `None`.
- **Remove `output_scale`** from `LoRALinear`.

### 11.4.2 Medium-Term

- **Test larger $F$** on long sequences.
- **Fused Triton kernel** for the feature map.

### 11.4.3 Long-Term

- **Combine with quantization.**
- **Scale to 7B+ models.**

## 11.5 Closing Remarks

Performer remains a strong kernel attention baseline. With the two bugs 
fixed, it is a solid choice for long-context inference.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.
```

---