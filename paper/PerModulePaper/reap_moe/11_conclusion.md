# 11. Conclusion

## 11.1 Summary

We presented **REAP-MoE** (Robust Expert Allocation and Pruning), a 
mixture-of-experts module that combines **noisy top-$k$ routing** with 
**SVD-factorized experts** and **load-balancing auxiliary loss**. The method 
addresses MoE parameter bloat, router collapse, and training instability.

## 11.2 Results Recap

| Model | Accuracy | Params Δ | VRAM Δ |
|---|---|---|---|
| DistilBERT | 90.02% | 66.4M | 380 MB |
| RoBERTa | 50.92% | 66.4M | 380 MB |
| GPT-2 | 52.41% | 66.4M | 380 MB |

## 11.3 Key Takeaways

1. **SVD reduces expert params** by ~4×.
2. **Aux loss prevents collapse.**
3. **Zero-init stabilizes training.**
4. **Parameter growth is real** — REAP-MoE is a re-architecture, not a 
   compressor.

## 11.4 Future Work

### Short-Term
- Remove FFN duplication
- Adaptive $E$ per layer
- Expert pruning

### Long-Term
- Theoretical MoE error bounds
- Cross-task expert transfer
- Combined MoE + quantization

## 11.5 Closing Remarks

REAP-MoE is a step toward parameter-efficient MoE. We release all code and 
checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs**: designed REAP-MoE, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team**: provided feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---