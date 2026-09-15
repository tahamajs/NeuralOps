# 11. Conclusion

## 11.1 Summary

We presented **HymbaTransX**, a hybrid differential attention + SSM block. 
The block combines two independent query-key pairs (differential attention) 
with a state space model, fused by a learned scalar gate.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | α final |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_ALPHA}} |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_ALPHA}} |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_ALPHA}} |

## 11.3 Key Takeaways

1. **Differential attention reduces noise** by {{NOISE_REDUCTION}}%.
2. **SSM fusion helps encoders** more than decoders.
3. **Learned α is informative** per model.
4. **Parameter cost is real** (~{{PARAM_INCREASE}}%).

## 11.4 Future Work

### Short-Term
- Fused Triton kernels for differential + SSM.
- Per-head α.
- λ scheduling.

### Long-Term
- Compression of the SSM branch.
- Cross-attention support.
- Scaling to 1B+ models.

## 11.5 Closing Remarks

HymbaTransX is a step toward richer attention blocks. We release all code 
and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the block, implemented, ran experiments, wrote 
  the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---
