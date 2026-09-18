# 11. Conclusion

## 11.1 Summary

We presented **Res-SVD Attention**, a two-level SVD factorization for 
attention projections that combines frozen base SVD with trainable 
spectral-init LoRA.

## 11.2 Results Recap

| Model | Acc Δ | Ratio |
|---|---|---|
| DistilBERT | -1.15% | **50\%×** |
| RoBERTa | -1.95% | **2.1\times×** |
| GPT-2 | -8.03% | **2.1\times×** |

## 11.3 Key Takeaways

1. **Real compression** of ~50\%×.
2. **Two-level decomposition** equivalent to equal-rank SVD at init.
3. **Spectral initialization** ensures optimal start.
4. **Silent injection failure** requires caller attention.

## 11.4 Future Work

### Short-Term
- Activation-weighted Res-SVD
- Adaptive base rank
- Test long-context

### Long-Term
- The theoretical advantage of two-level structure
- Interaction with quantization
- Comparison with PiSSA

## 11.5 Closing Remarks

Res-SVD offers a principled approach to attention compression with 
parameter-efficient fine-tuning. We release all code.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed Res-SVD, implemented the code, ran experiments.
- **Collaborative Research Team** provided feedback.

## 11.8 Conflict of Interest

None.

---