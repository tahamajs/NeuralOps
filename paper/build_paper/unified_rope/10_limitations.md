<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Compression

UnifiedRoPE is **not** a compression method. It does not reduce:
- Parameter count
- KV-cache size
- Attention compute

## 10.2 Added RoPE Changes Behavior

For DistilBERT, RoBERTa, and GPT-2, RoPE is **added** to a model trained 
with absolute positions. The attention pattern changes, and accuracy may 
drop.

## 10.3 LoRA Recovery Needed

Adding RoPE requires recovery fine-tuning. Without it, accuracy drops 
significantly.

## 10.4 Cross-Attention Not Supported

Encoder-decoder models (T5, BART) are not supported.

## 10.5 Task Coverage

Only SST-2 is evaluated. Long-context tasks may behave differently.

## 10.6 Model Coverage

Only four models tested. Larger models not tested.

## 10.7 Hardware

Only tested on RTX 4070 Laptop (8 GB).

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.9 Heuristic Hyperparameters

`max_seq_len`, `base`, `rotary_dim` are empirical.

## 10.10 Safe Injection Raises on Failure

`inject_unified_rope` raises `RuntimeError` on failure. Callers must 
handle this.

## 10.11 No Comparison with Performer/Linformer

We do not compare with other efficient attention methods.

## 10.12 Code Quality

Research-grade. Known issues:
- Broad exception handling
- `raise` in some paths should log

## 10.13 Ethical Considerations

See Appendix E.

---