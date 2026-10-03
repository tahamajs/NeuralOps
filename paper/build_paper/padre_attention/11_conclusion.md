# 11. Conclusion

## 11.1 Summary

We presented **PADRe**, a polynomial-kernel attention module that replaces 
softmax with $(1 + s/d)^d$. The module supports four architectures with 
Frobenius validation and output equivalence checking. We are explicit that 
kernelization — the source of theoretical speedup — is left to future work.

## 11.2 Results Recap

| Model | Acc Δ |
|---|---|
| DistilBERT | -0.34% |
| RoBERTa | -14.00% |
| GPT-2 | -28.44% |

## 11.3 Key Takeaways

1. **Polynomial substitution works.**
2. **Max-centering is essential.**
3. **Order 2 is sufficient.**
4. **Kernelization is future work.**

## 11.4 Future Work

### Short-Term
- Kernelize the polynomial: $\phi(x) = (1, x, x^2, \ldots, x^d)$
- Learned order
- Comparison with Performer

### Long-Term
- Theory: bound downstream error
- Multi-task evaluation
- Cross-model transfer

## 11.5 Closing Remarks

PADRe is a research prototype for polynomial attention in pretrained 
transformers. Code and checkpoints released.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed, implemented, ran experiments, wrote paper.
- **Collaborative Research Team** provided feedback, edited.

## 11.8 Conflict of Interest

None.

---