# 11. Conclusion

## 11.1 Summary

We presented **HGRN2-GatedDeltaNet**, a hybrid linear recurrent attention 
module combining HGRN-2 and Gated DeltaNet with a learned gate. The module 
runs in $\mathcal{O}(L)$ time and memory, replacing the $\mathcal{O}(L^2)$ 
of standard attention.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Speed @ L=128 | Speed @ L=1024 |
|---|---|---|---|---|
| DistilBERT | -37.15% | +48.5\%% | 1.0x× | 86.4× |
| RoBERTa | -25.58% | +48.5\%% | 84.5× | 84.5× |
| GPT-2 | -28.33% | +48.5\%% | 84.5× | 84.5× |

## 11.3 Key Takeaways

1. **Linear recurrence works** at long L.
2. **Two branches beat one** (0.85%).
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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the module, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---