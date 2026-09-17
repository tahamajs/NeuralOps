# 10. Limitations

## 10.1 Parameter Count

HGRN2-GatedDeltaNet **increases** parameter count by ~48.5\%% per 
layer. This is a substantial limitation.

## 10.2 Short-Sequence Slowdown

At L = 128 (SST-2 standard), our method is **1.0x× slower** than 
baseline. This limits applicability to standard benchmarks.

## 10.3 Python Loop

The DeltaNet branch uses a Python-level loop. This is the primary 
performance bottleneck at short sequences.

## 10.4 Task Coverage

Only SST-2 is evaluated. Behavior on:
- Generation tasks: unknown
- Multi-class classification: unknown
- Token-level tasks: unknown
- Long-context tasks: we do not test L > 512 on SST-2.

## 10.5 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.6 Hardware

All experiments on RTX 4070. Behavior on other hardware untested.

## 10.7 No Theoretical Guarantee

We provide no formal approximation bound relative to standard attention.

## 10.8 Training Instability on RoBERTa

RoBERTa's loss oscillates during recovery, suggesting the optimization 
landscape is difficult. We do not fully characterize this.

## 10.9 Cross-Attention Not Implemented

If `encoder_hidden_states` is provided, we return identity. This breaks 
encoder-decoder models.

## 10.10 Single-Seed Results

Single-seed. Multi-seed would strengthen conclusions.

## 10.11 No Comparison with Mamba

We do not directly compare with Mamba, Jamba, or other modern SSMs.

## 10.12 The `is_nested_encoder` Issue

For RoBERTa, we set `is_nested_encoder = True`, which bypasses the internal 
output projection. If this is misconfigured, RoBERTa degrades. We do not 
fully diagnose this.

## 10.13 Code Quality

Research-grade:
- Broad exception handling
- No fused kernels
- No unit tests for edge cases

## 10.14 Ethical Considerations

Recurrent attention is dual-use. See Appendix E.

---