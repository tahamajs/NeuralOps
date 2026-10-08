<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Inference Speedup

Squeezing Heads adds LoRA overhead. Inference is slower than baseline 
unless LoRA is folded into base weights.

## 10.2 Storage Overhead

SH stores both frozen weights AND LoRA adapters. Storage is slightly 
higher than baseline.

## 10.3 Head-Dim Reduction Loses Information

Reducing head dimension from $d_h$ to $d_h'$ cannot be perfectly reversed. 
There is an irreducible accuracy loss.

## 10.4 Task Coverage

Only SST-2 evaluated. Long-context tasks not tested.

## 10.5 Model Coverage

Not tested on:
- Large models ($H > 32$)
- Mixture-of-Experts
- Vision transformers

## 10.6 No Comparison with Per-Head Rank

Uniform LoRA rank may not be optimal.

## 10.7 Single Seed

All results from seed 42.

## 10.8 No Comparison with Some Baselines

We do not compare with:
- DistilBERT (distillation)
- TinyBERT
- Head pruning + full fine-tuning

## 10.9 Failure Modes

Long-context, inference-time LoRA folding

## 10.10 Code Quality

Research-grade implementation.

## 10.11 Ethical Considerations

Dual-use. See Appendix E.

---
