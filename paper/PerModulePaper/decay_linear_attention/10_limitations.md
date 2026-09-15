# 10. Limitations

## 10.1 Feature Map Simplicity

We use ELU+1, which is simple but may not approximate softmax as well as 
higher-order feature maps (Performer's PORF, CHELA-SLAY's Yat-Kernel).

## 10.2 Cross-Attention Not Fully Tested

We implement cross-attention but only test it preliminarily on T5. 
Behavior on BART and other encoder-decoders is untested.

## 10.3 Task Coverage

Only SST-2 (classification) is evaluated. Generation, translation, and 
summarization are untested.

## 10.4 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.5 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.6 Chunk Size Choice

$C = 32$ is a fixed default. Adaptive chunk size might improve 
efficiency.

## 10.7 Log-Decay Range

$[{{LOG_DECAY_MIN}}, {{LOG_DECAY_MAX}}]$ is a fixed initialization. The 
optimal range may be model-dependent.

## 10.8 No Theoretical Bound

We provide no formal approximation bound on the linear attention output 
relative to softmax.

## 10.9 Single-Seed Results

Single-seed. Multi-seed would strengthen conclusions.

## 10.10 Bidirectional Average

We use a fixed $0.5$ weight for forward and backward scans. A learned 
weight might be better.

## 10.11 No Comparison with Some Methods

We do not compare with Mamba, RWKV, or other SSMs.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No fused kernels
- No unit tests for edge cases

## 10.13 Ethical Considerations

Linear attention is dual-use. See Appendix E.

---
