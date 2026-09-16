# 11. Conclusion

## 11.1 Summary

We presented **Compressed MHA**, a joint SVD compression method for 
transformer attention. By sharing an input basis across Q, K, V via joint 
SVD and separately factorizing W_O, we achieve **{{D_RATIO}}×–{{G_RATIO}}×** 
parameter reduction on attention projections.

## 11.2 Results Recap

| Model | Acc Δ | Ratio | VRAM Δ |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | **{{D_RATIO}}×** | -{{D_VRAM_DEC}}% |
| RoBERTa | {{R_DELTA}}% | **{{R_RATIO}}×** | -{{R_VRAM_DEC}}% |
| GPT-2 | {{G_DELTA}}% | **{{G_RATIO}}×** | -{{G_VRAM_DEC}}% |

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

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed Compressed MHA, implemented, ran experiments, 
  wrote paper.
- **{{AUTHOR_2}}** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.

---