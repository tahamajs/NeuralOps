# 10. Limitations

## 10.1 Parameter Cost

JambaInfo increases parameter count by ~66.4M%.

## 10.2 Training Slower

~84.5× slower than baseline.

## 10.3 Inference Latency

~84.5× slower.

## 10.4 Cross-Attention Not Supported

`encoder_hidden_states` raises `NotImplementedError`.

## 10.5 Task Coverage

Only SST-2. Untested:
- Generation
- Long-context
- Token-level tasks
- T5-style seq2seq

## 10.6 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

Only RTX 4070.

## 10.8 Soft Fallback

`copy_weights` logs warnings on missing projections but does not raise. 
Risk of silent degradation.

## 10.9 Sparsity Parameter

The `sparsity` parameter is currently a placeholder. No sparsity penalty is 
actually applied in this version.

## 10.10 Bottleneck Initialization

We use default PyTorch init. PCA or SVD init might improve.

## 10.11 No Theoretical Bound

We provide no formal bound on the information-theoretic benefit of the 
bottleneck.

## 10.12 Single-Seed

Single-seed results. Multi-seed would strengthen.

## 10.13 α Saturation

At some initializations, α saturates and the model degenerates.

## 10.14 Code Quality

Research-grade.

## 10.15 Ethical Considerations

See Appendix E.

---