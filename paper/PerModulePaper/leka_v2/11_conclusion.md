# 11. Conclusion

## 11.1 Summary

We presented **LEKAv2**, a linear-time kernel attention with landmark 
distillation, dual-scale RFF, softpick, and local-global fusion. The 
module achieves **{{D_ACC}}%** on DistilBERT, **{{R_ACC}}%** on RoBERTa, 
and **{{G_ACC}}%** on GPT-2, with modest parameter cost.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Landmark ratio |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_LR}} |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_LR}} |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_LR}} |

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

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented, ran experiments, 
  wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---