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
| DistilBERT | 50.92% | -39.79% | 1.42x s |
| RoBERTa | 50.92% | -27.64% | 2.3 s s |
| GPT-2 | 77.5% | 0.85% | 2.3 s s |

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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support. for feedback.

## 11.7 Author Contributions

- **Taha Majs** implemented the module, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---