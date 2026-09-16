# 11. Conclusion

## 11.1 Summary

We presented **HGRN2-GatedDeltaNet**, a hybrid linear recurrent attention 
module combining HGRN-2 and Gated DeltaNet with a learned gate. The module 
runs in $\mathcal{O}(L)$ time and memory, replacing the $\mathcal{O}(L^2)$ 
of standard attention.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Speed @ L=128 | Speed @ L=1024 |
|---|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{SLOWDOWN}}× | {{S1024}}× |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_SLOWDOWN}}× | {{R_S1024}}× |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_SLOWDOWN}}× | {{G_S1024}}× |

## 11.3 Key Takeaways

1. **Linear recurrence works** at long L.
2. **Two branches beat one** ({{BRANCH_GAIN}}%).
3. **Python loop is the bottleneck** at short L.
4. **Fused kernels are the future direction.**

## 11.4 Future Work

### Short-Term
- Fused Triton kernel for DeltaNet recurrence.
- Parallel scan for HGRN-2.
- Cross-layer gate sharing.

### Long-Term
- Replace HGRN-2 with Mamba.
- Extend to cross-attention.
- Multimodal integration.

## 11.5 Closing Remarks

HGRN2-GatedDeltaNet is a step toward practical linear attention. We 
release all code and checkpoints.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the module, implemented, ran experiments, wrote 
  the paper.
- **{{AUTHOR_2}}** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---