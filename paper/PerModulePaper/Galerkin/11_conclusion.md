<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **Galerkin Attention**, a Petrov-Galerkin projection of 
transformer self-attention onto a fixed DCT basis. The method was evaluated 
on three pretrained models on SST-2. **Results are negative for encoders**: 
DistilBERT drops {{D_DELTA}}% and RoBERTa collapses to {{R_ACC}}%. GPT-2 
shows no change because the causal path falls back to standard attention.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ |
|---|---|---|---|
| DistilBERT | Galerkin | {{D_ACC}}% | {{D_DELTA}} |
| RoBERTa | Galerkin | {{R_ACC}}% | {{R_DELTA}} |
| GPT-2 | Fallback | {{G_ACC}}% | {{G_DELTA}} |

## 11.3 Key Takeaways

1. **Fixed basis projection fails.** Learned or data-dependent bases are 
   necessary.
2. **Attention is fundamentally data-dependent.** No fixed basis can capture 
   content-selective, position-sensitive patterns.
3. **Decoder fallback is not a success.** It is a no-op.

## 11.4 Future Work

### 11.4.1 Short-Term

- **Test learned bases** (Linformer-style projection).
- **Test data-dependent DCT** (basis from each batch).
- **Test hybrid approaches** (DCT + learned).

### 11.4.2 Long-Term

- **Theoretical analysis:** prove that fixed bases cannot approximate 
  trained attention.
- **Attention sharpness:** measure this as a function of layer depth.
- **Alternative decompositions:** wavelet, learned dictionary.

## 11.5 Closing Remarks

This paper documents a **negative result**. We hope it saves the community 
time by showing that fixed-basis attention projection is not viable. The 
lesson: attention is fundamentally adaptive, and any successful 
approximation must preserve that adaptivity.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.
```

---