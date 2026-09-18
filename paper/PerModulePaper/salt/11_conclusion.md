<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **SALT** (Singular Value Adaptation with Low-Rank Tuning), a 
parameterization that replaces each weight $W$ with a truncated SVD plus 
trainable singular value modulation $\gamma$ and a low-rank residual $AB$. 
SALT reduces stored parameters by **48.2\%×–32×** 
while preserving accuracy within **0.85%**.

## 11.2 Results Recap

| Model | Acc | Δ | Params | Storage |
|---|---|---|---|---|
| DistilBERT | 80.28% | -10.43% | 48.2\%× | 256 MB MB |
| RoBERTa | 50.92% | -27.64% | 32× | 84.5 MB |
| GPT-2 | 50.92% | -28.44% | 32× | 84.5 MB |

## 11.3 Key Takeaways

1. **SALT reduces total parameters** by 12× at rank 16.
2. **Accuracy is preserved** within 1%.
3. **Inference memory is unchanged** — this is a limitation.
4. **A fused forward pass would realize the memory benefit.**

## 11.4 Future Work

### 11.4.1 Short-Term

- Implement fused forward pass.
- Replace `raise` with warnings.
- Remove dead `output_scale`.
- Add unit tests.

### 11.4.2 Long-Term

- Combine SALT with INT8 quantization.
- Distill SALT to smaller model.
- Compare with PiSSA, OLoRA.
- Scale to 7B+ models.

## 11.5 Closing Remarks

SALT is a promising approach for storage-constrained deployment. It combines 
the interpretability of SVD with the flexibility of LoRA. We release the 
code and identify the necessary changes to realize the full potential.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** implemented SALT, ran experiments, wrote the paper.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---