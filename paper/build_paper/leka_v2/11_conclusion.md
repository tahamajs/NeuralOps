# 11. Conclusion

## 11.1 Summary

We presented **LEKAv2**, a linear-time kernel attention with landmark 
distillation, dual-scale RFF, softpick, and local-global fusion. The 
module achieves **86.01%** on DistilBERT, **50.92%** on RoBERTa, 
and **50.92%** on GPT-2, with modest parameter cost.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Landmark ratio |
|---|---|---|---|
| DistilBERT | -4.70% | +48.5\%% | 0.85 |
| RoBERTa | -27.64% | +48.5\%% | 0.85 |
| GPT-2 | -28.44% | +48.5\%% | 0.85 |

## 11.3 Key Takeaways

1. **Learned landmark distillation** helps.
2. **Dual-scale RFF** helps.
3. **Softpick** stabilizes masked attention.
4. **Local-global fusion** improves quality.
5. **Not truly linear at ρ = 0.5.**

## 11.4 Future Work

### Short-Term
- Truly linear version with fixed $L_\text{max}$.
- Adaptive ρ learning.
- INT8 quantization.

### Long-Term
- Cross-attention support.
- Scaling to 1B+.
- Landmark transfer across tasks.

## 11.5 Closing Remarks

LEKAv2 is a step toward richer linear attention. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---