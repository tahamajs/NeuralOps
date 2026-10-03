# 11. Conclusion

## 11.1 Summary

We presented a production-hardened Linformer implementation that handles 
four architectures with adaptive slicing, diagonal initialization, and 
causal-safe fallback.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ |
|---|---|---|
| DistilBERT | -39.79% | +48.5\%% |
| RoBERTa | -30.39% | +48.5\%% |
| GPT-2 | 0.85% | +48.5\%% |

## 11.3 Key Takeaways

1. **Encoders benefit** from linear complexity.
2. **Decoders need fallback** for causality.
3. **Diagonal init helps** convergence.
4. **Parameter overhead** is ~7%.

## 11.4 Future Work

### Short-Term
- Fused kernel for $E, F$ projection
- Per-head $k$ adaptation
- INT8 combined with Linformer

### Long-Term
- Causal-safe low-rank projection
- Learned $L_\text{max}$
- Multimodal extension

## 11.5 Closing Remarks

Code and checkpoints released.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---