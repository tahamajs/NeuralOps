# 10. Limitations

## 10.1 Parameter Cost

HymbaTransX increases parameter count by ~66.4M%.

## 10.2 Training Slower

The two Q-K pairs and SSM branch slow training by 84.5×.

## 10.3 Inference Latency

Inference is 84.5× slower than baseline.

## 10.4 Cross-Attention Not Supported

Encoder-decoder models raise `NotImplementedError`.

## 10.5 Task Coverage

Only SST-2 is evaluated. Behavior on:
- Generation
- Long-context
- Token-level tasks

is untested.

## 10.6 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

All experiments on RTX 4070.

## 10.8 Soft Fallback Risk

Our `copy_weights` logs a warning on missing weights but does not raise. This 
can lead to silent degradation on unusual architectures.

## 10.9 α Saturation

If α saturates to 0 or 1, the model degenerates. We observe this on 
0.85.

## 10.10 No Theoretical Guarantee

We provide no bound on attention noise reduction or task accuracy.

## 10.11 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.12 No Comparison with Hymba

We don't run head-to-head against Hymba or Mamba.

## 10.13 Code Quality

Research-grade implementation.

## 10.14 Ethical Considerations

Hybrid attention is dual-use. See Appendix E.

---