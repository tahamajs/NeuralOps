# 11. Conclusion

## 11.1 Summary

We presented **BAM-UMoE**, a capacity expansion method for post-hoc MoE 
application to pretrained transformers. The method introduces a three-part 
step-0 preservation mechanism (zero-init experts, negative gating bias, 
frozen base) that guarantees exact forward-pass equivalence at 
initialization.

## 11.2 Results Recap

| Model | Acc Δ | Params Δ |
|---|---|---|
| DistilBERT | -0.69% | +226M |
| RoBERTa | +6.07% | +445M |
| GPT-2 | +2.06% | +445M |

## 11.3 Key Takeaways

1. **Step-0 preservation works** — verified empirically.
2. **Load balancing is stable** — auxiliary loss effective.
3. **Parameter cost is high** — inherent to MoE.
4. **Not a compression method.**

## 11.4 Future Work

### Short-Term
- Sparse MoE (1 expert per token)
- Interaction with compression
- Multi-task evaluation

### Long-Term
- Theoretical step-0 bounds
- Multi-modal extension
- Sparse hardware deployment

## 11.5 Closing Remarks

BAM-UMoE is a **research tool** for post-hoc capacity expansion. It is 
**not** a compression method. We release code and checkpoints for 
reproducibility.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments.
- **Collaborative Research Team** provided methodological feedback and edited the manuscript.

## 11.8 Conflict of Interest

None.

---