# 11. Conclusion

## 11.1 Summary

We presented **LEKAv1**, a linear-time kernel attention that uses 
QR-orthogonalized Random Fourier Features to approximate the Gaussian 
RBF kernel. The module includes a preservation gate for residual safety, 
head-wise LayerNorm for stability, and supports GQA and RoPE.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Speed @ L=2048 |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{S2048}}× |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_S2048}}× |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_S2048}}× |

## 11.3 Key Takeaways

1. **RFF works.**
2. **QR projection improves accuracy.**
3. **Preservation gate is essential.**
4. **Linear scaling for long context.**

## 11.4 Future Work

### Short-Term
- Blocked cumsum for lower memory.
- Adaptive F.
- Combined with weight compression.

### Long-Term
- Combined with Performer.
- Extension to cross-attention.
- Theoretical bounds.

## 11.5 Closing Remarks

LEKAv1 is a practical linear-time attention. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented, ran experiments, 
  wrote the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---