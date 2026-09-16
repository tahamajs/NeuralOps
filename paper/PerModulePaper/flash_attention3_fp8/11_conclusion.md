<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented a **hardened, hardware-aware** implementation of FlashAttention-3 
FP8 that runs on any CUDA GPU by falling back to PyTorch SDPA when FP8 
tensor cores are unavailable. The implementation handles four transformer 
families and preserves accuracy within **{{D_DELTA}}%** (DistilBERT), 
**{{R_DELTA}}%** (RoBERTa), and **{{G_DELTA}}%** (GPT-2) on SST-2.

## 11.2 Results Recap

| Model | Accuracy | Δ | Params Δ | FP8 Path |
|---|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | No |
| RoBERTa | {{R_ACC}}% | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | No |
| GPT-2 | {{G_ACC}}% | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | No |

## 11.3 Key Takeaways

1. **Portability matters.** Hardware detection prevents crashes.
2. **FP8 is hardware-limited.** Without Hopper, no speedup.
3. **SDPA is a strong fallback.** Accuracy is preserved.
4. **Honest reporting is essential.** We did not measure FP8; we say so.

## 11.4 Future Work

### 11.4.1 Short-Term

- Test on H100 to validate FP8 speedups.
- Profile FP8 cast overhead.
- Add CUDA capability detection.

### 11.4.2 Long-Term

- Support cross-attention.
- Integrate with TransformerEngine.
- Test on 7B+ models.

### 11.4.3 Methodological

- Standardize hardware reporting.
- Publish FP8 accuracy curves.
- Compare with FP8 training.

## 11.5 Closing Remarks

This work is a **cautionary example**: the original FlashAttention-3 paper 
reports impressive speedups on H100. On consumer hardware, those speedups 
are unavailable. We provide the code that works everywhere and report 
honestly where it does not.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---