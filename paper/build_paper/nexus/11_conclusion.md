# 11. Conclusion

## 11.1 Summary

We presented NEXUS, a modular framework for combining six sparse attention 
mechanisms. NEXUS achieves sub-quadratic attention for encoders while 
falling back to dense for causal models.

## 11.2 Results Recap

| Model | Acc Δ | Path |
|---|---|---|
| DistilBERT | -39.79% | sparse |
| RoBERTa | -27.64% | sparse |
| GPT-2 | -27.75% | dense |

## 11.3 Key Takeaways

1. **Modular design enables experimentation.**
2. **Encoder-only sparsity** in our implementation.
3. **Two components (stripe, RFF) inactive** under default SST-2.
4. **Softpick needs care.**
5. **Speedup grows with $N$.**

## 11.4 Future Work

### Short-Term
- Causal block-sparse path
- Enable stripe for shorter sequences
- Alternative to softpick

### Long-Term
- Component interaction study
- Theoretical analysis
- Multi-task evaluation

## 11.5 Closing Remarks

NEXUS is a research framework for studying combined sparse attention. We 
release code and checkpoints, and we explicitly document which components 
are active in our experiments.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed, implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---