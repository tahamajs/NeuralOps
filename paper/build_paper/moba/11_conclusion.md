# 11. Conclusion

## 11.1 Summary

We presented **Mamba-Inspired Gated State-Space Attention**, a drop-in 
replacement for multi-head attention that:
- Reuses pretrained Q/K/V/O weights
- Applies learned exponential decay
- Uses a preservation gate
- Supports causal and bidirectional modes

We are explicit that the implementation is **not** true Mamba and remains 
$\mathcal{O}(L^2 d)$.

## 11.2 Results Recap

| Model | Acc Δ |
|---|---|
| DistilBERT | -17.09% |
| RoBERTa | +5.61% |
| GPT-2 | -28.44% |

## 11.3 Key Takeaways

1. **Pretrained weight reuse is possible** in SSM-inspired structures.
2. **Learned decay is effective.**
3. **Preservation gate is essential.**
4. **Complexity remains quadratic** — a key limitation.

## 11.4 Future Work

### Short-Term
- True $\mathcal{O}(L)$ implementation via chunked scan
- Selective decay (input-dependent $\gamma$)
- Per-head preservation gates

### Long-Term
- Hybrid Mamba + attention
- Theoretical analysis of the two-branch mapping
- Cross-task transfer of learned decay

## 11.5 Closing Remarks

Mamba-Inspired SSM Attention provides a pretrained-friendly baseline for 
SSM research. Code released. **We encourage users to be aware of the 
quadratic complexity despite the SSM naming.**

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed, implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---