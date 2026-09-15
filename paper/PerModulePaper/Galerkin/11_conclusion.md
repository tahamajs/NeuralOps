# 11. Conclusion

## 11.1 Summary

We presented **Galerkin Attention**, which formulates attention as a 
Petrov-Galerkin projection onto a DCT basis, achieving $\mathcal{O}(Nrd)$ 
complexity for encoder models. Decoders fall back to standard softmax.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Path |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_DELTA}}M | Galerkin |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_DELTA}}M | Galerkin |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_DELTA}}M | Fallback |

## 11.3 Key Takeaways

1. **Encoder linear complexity** is achievable via basis projection.
2. **DCT is the natural basis** for attention smoothness.
3. **Decoder fallback** is necessary for causality.
4. **Not a compression method.**

## 11.4 Future Work

### Short-Term
- Learned basis
- Causal-compatible basis
- Cross-attention evaluation

### Long-Term
- Higher-order DCT variants
- Adaptive rank per layer
- Integration with FlashAttention

## 11.5 Closing Remarks

Galerkin Attention is a mathematically principled encoder efficiency method. 
We release code under {{LICENSE}}.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented the code, ran experiments.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.
