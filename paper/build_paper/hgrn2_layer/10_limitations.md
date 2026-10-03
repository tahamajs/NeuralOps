# 10. Limitations

## 10.1 Parameter Count

HGRN2Layer **increases** parameter count by ~48.5\%%.

## 10.2 Short-Sequence Slowdown

At L = 128, we are **1.0x× slower** than baseline.

## 10.3 Python Loop

The chunked scan still uses a Python loop over chunks, limiting parallelism.

## 10.4 Cross-Attention Not Implemented

Encoder-decoder models are not supported. We raise `NotImplementedError`.

## 10.5 Task Coverage

Only SST-2 is evaluated. Generation, long-context, and token-level tasks 
are untested.

## 10.6 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.8 No Theoretical Guarantee

We provide no formal approximation bound relative to standard attention.

## 10.9 Training Instability on RoBERTa

RoBERTa's loss curve is less stable than DistilBERT's. We do not fully 
diagnose this.

## 10.10 Single-Seed Results

Multi-seed analysis would strengthen conclusions.

## 10.11 No Comparison with Mamba

We do not compare with Mamba, GLA, or other modern SSMs.

## 10.12 `is_nested_encoder` Overreach

For RoBERTa, we set `layer.attention.output.dense = nn.Identity()`. If 
this is wrong, the output projection is skipped, causing degradation.

## 10.13 Code Quality

Research-grade implementation:
- Broad exception handling
- No fused kernels
- No unit tests for edge cases

## 10.14 Ethical Considerations

Recurrent attention is dual-use. See Appendix E.

---