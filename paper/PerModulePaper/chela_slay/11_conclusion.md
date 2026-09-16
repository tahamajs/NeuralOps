# 11. Conclusion

## 11.1 Summary

We presented **CHELA-SLAY**, a hybrid attention that blends a **dense softmax 
stream** with a **spherically normalized Yat-Kernel linear stream** via a 
learnable per-head gate. The gate is initialized to favor the dense path 
($\sigma({{GATE_INIT_BIAS}}) \approx {{GATE_INIT}}$), preserving pretrained 
behavior at step 0.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | Time Ratio | VRAM |
|---|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | +{{D_PARAM_PCT}}% | {{D_TIME_RATIO}}× | {{D_VRAM}} MB |
| RoBERTa | {{R_DELTA}}% | +{{R_PARAM_PCT}}% | {{R_TIME_RATIO}}× | {{R_VRAM}} MB |
| GPT-2 | {{G_DELTA}}% | +{{G_PARAM_PCT}}% | {{G_TIME_RATIO}}× | {{G_VRAM}} MB |

## 11.3 Key Takeaways

1. **Dense anchor is essential** — pure linear loses ~{{ANCHOR_GAIN}}%.
2. **Spherical normalization stabilizes** — {{SPH_GAIN}}% improvement.
3. **Yat-Kernel outperforms ELU+1.**
4. **Cumsum memory is the main limitation.**
5. **Speedup only at long L** ($L \geq {{BREAKEVEN}}$).

## 11.4 Future Work

### Short-Term
- Chunked cumsum to eliminate memory bottleneck
- Learned $\Omega$ for better feature map
- Per-dimension gate

### Long-Term
- Combine with token compression
- Larger-scale evaluation
- Theoretical error bounds

## 11.5 Closing Remarks

CHELA-SLAY demonstrates a stable hybrid of dense and linear attention. The 
dense anchor is essential for preserving pretrained accuracy. We release all 
code and checkpoints.

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