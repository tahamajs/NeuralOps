# 11. Conclusion

## 11.1 Summary

We presented **Compressed MHA**, a joint SVD compression method for 
transformer attention. By sharing an input basis across Q, K, V via joint 
SVD and separately factorizing W_O, we achieve **1.38×–1.31×** 
parameter reduction on attention projections.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | -39.79% | **1.38×** | -18% |
| RoBERTa | -27.64% | **1.31×** | -380 MB% |
| GPT-2 | -28.44% | **1.31×** | -380 MB% |

## 11.3 Key Takeaways

1. **Joint SVD exploits Q/K/V correlation** — fewer parameters at equal 
   accuracy.
2. **Nested detection order is critical** — wrong order causes collapse.
3. **LoRA recovery is essential** — closes most of the accuracy gap.
4. **Multi-architecture support** works across four model families.

## 11.4 Future Work

### Short-Term
- Fused Triton kernel for joint SVD forward
- Per-layer rank selection
- INT8 combined with Compressed MHA

### Long-Term
- Cross-layer factor sharing
- Joint Q/K/V/O factorization
- Theoretical approximation bounds

## 11.5 Closing Remarks

Compressed MHA is a practical, honest approach to attention compression. We 
release all code and checkpoints.

## 11.6 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## 11.7 Author Contributions

- **Taha Majs** designed Compressed MHA, implemented, ran experiments, 
  wrote paper.
- **Collaborative Research Team** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---