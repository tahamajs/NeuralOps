# 11. Conclusion

## 11.1 Summary

We presented **HymbaTransX**, a hybrid differential attention + SSM block. 
The block combines two independent query-key pairs (differential attention) 
with a state space model, fused by a learned scalar gate.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ | α final |
|---|---|---|---|
| DistilBERT | -41.63% | +48.5\%% | 0.85 |
| RoBERTa | -27.41% | +48.5\%% | 0.85 |
| GPT-2 | -21.79% | +48.5\%% | 0.85 |

## 11.3 Key Takeaways

1. **Differential attention reduces noise** by 32%.
2. **SSM fusion helps encoders** more than decoders.
3. **Learned α is informative** per model.
4. **Parameter cost is real** (~66.4M%).

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

We thank We thank the anonymous reviewers and the compute cluster administrators for their support..

## 11.7 Author Contributions

- **Taha Majs** designed the block, implemented, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---