<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Conclusion
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We reproduced **Nyströmformer** (Xiong et al., 2021) and ported it to 
four transformer families: DistilBERT, RoBERTa, GPT-2, and Qwen3.5. 
The Nyström approximation reduces attention compute via landmark pooling 
and iterative pseudoinverse, achieving $\mathcal{O}(N m d)$ complexity.

## 11.2 Results Recap

| Model | N=128 | N=512 |
|---|---|---|
| DistilBERT | Dense (baseline) | Nyström (86.2× speedup) |
| RoBERTa | Dense (baseline) | Nyström (86.2× speedup) |
| GPT-2 | Dense causal | Dense causal |
| Qwen3.5 | Dense causal | Dense causal |

## 11.3 Key Takeaways

1. **Short sequences don't benefit.**
2. **Causal decoders don't benefit.**
3. **Nyströmformer ports to all four architectures.**
4. **Adaptive pooling + Newton-Schulz pinv** work as in the original.
5. **Nyströmformer is orthogonal** to weight/KV compression.

## 11.4 Future Work

### Short-Term
- Causal Nyström variant
- Combined with weight/KV compression
- Alternative landmark selection

### Long-Term
- Theoretical error bound
- Extension to cross-attention
- Long-context task evaluation

## 11.5 Closing Remarks

Nyströmformer is a viable compute-compression technique for long-sequence 
encoders. We release all code and checkpoints.

## 11.6 Acknowledgments

We thank We thank the anonymous reviewers and the compute cluster administrators for their support.. This work reproduces Xiong et al. (2021).

## 11.7 Author Contributions

- **Taha Majs** implemented the reproduction, ran experiments, wrote 
  the paper.
- **Collaborative Research Team** provided feedback, edited the manuscript.

## 11.8 Conflict of Interest

None.

---