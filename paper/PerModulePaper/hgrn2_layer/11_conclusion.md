# 11. Conclusion

## 11.1 Summary

We presented **HGRN2Layer**, an element-wise linear recurrent module that 
replaces multi-head self-attention with a chunked parallel scan in 
$\mathcal{O}(L d)$ time and $\mathcal{O}(d)$ memory per step. It supports 
bidirectional (encoder) and causal (decoder) modes and transfers pretrained 
weights from four architectures.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Speed @ L=128 | Speed @ L=1024 |
|---|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{SLOWDOWN}}× | {{S1024}}× |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_SLOWDOWN}}× | {{R_S1024}}× |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_SLOWDOWN}}× | {{G_S1024}}× |

## 11.3 Key Takeaways

1. **Element-wise recurrence works** at long L.
2. **Bounded forget gate is essential.**
3. **Input modulation prevents drift.**
4. **Python loop limits short-sequence speed.**
5. **Parameters increase** — we are honest about this.

## 11.4 Future Work

### Short-Term
- Fused Triton kernel for chunked scan.
- Larger chunk size experiments.
- Cross-layer gate sharing.

### Long-Term
- Theoretical bound on accuracy loss.
- Cross-task transferability.
- Integration with attention (hybrid layers).

## 11.5 Closing Remarks

HGRN2Layer is a practical, element-wise recurrent alternative to attention. 
We release all code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the module, implemented, ran experiments, wrote 
  the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.
```

---