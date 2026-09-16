<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented a **production-hardened port of Cosformer** to four pretrained 
transformer families. The implementation supports bidirectional (encoder) 
and causal (decoder) paths, GQA, RoPE, and LoRA recovery.

## 11.2 Results Recap

| Model | Path | Accuracy | Δ | Time |
|---|---|---|---|---|
| DistilBERT | Encoder | {{D_ACC}}% | {{D_DELTA}}% | {{D_TIME}} s |
| RoBERTa | Encoder | {{R_ACC}}% | {{R_DELTA}}% | {{R_TIME}} s |
| GPT-2 | Decoder | {{G_ACC}}% | {{G_DELTA}}% | {{G_TIME}} s |

## 11.3 Key Takeaways

1. **Cosformer preserves most of the baseline accuracy** on flat encoders.
2. **Decoder overhead is significant** (~{{G_TIME_R}}×).
3. **RoBERTa needs hyperparameter tuning** (LR, reweighting).
4. **Not a compression method** — adds LoRA parameters.

## 11.4 Future Work

### 11.4.1 Short-Term

- Tune LR for RoBERTa
- Test removing cos/sin reweighting for RoBERTa
- Test chunk_size in {16, 64, 128}

### 11.4.2 Medium-Term

- Test on long sequences ($L > 1024$)
- Test on generation tasks
- Compare with Performer, Linformer

### 11.4.3 Long-Term

- Implement fused Triton kernel for chunked scan
- Extend to Qwen3.5 with RoPE + GQA
- Scale to 7B models

## 11.5 Closing Remarks

Cosformer is one of the **better linear attention mechanisms** for 
pretrained transformers. On flat encoders, it preserves most accuracy at 
modest overhead. Future work should focus on (a) reducing decoder overhead 
via fused kernels, (b) resolving RoBERTa instability, and (c) testing at 
longer sequences where linear attention has its main advantage.

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for feedback.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** implemented the module, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback.

## 11.8 Conflict of Interest

None.

---