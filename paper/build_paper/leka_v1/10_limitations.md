# 10. Limitations

## 10.1 Parameter Cost

LEKAv1 adds ~66.4M% parameters per layer.

## 10.2 Softmax vs Gaussian

RFF approximates the Gaussian kernel, not softmax. Tasks requiring exact 
softmax may see accuracy loss.

## 10.3 Cumsum Memory

At L > 512, the cumsum materializes a large tensor.

## 10.4 Fatal Error Handling

`copy_weights` raises `RuntimeError` on extractor failure. This can kill 
the module.

## 10.5 Cross-Attention

Not supported.

## 10.6 Task Coverage

Only SST-2.

## 10.7 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.8 Hardware

Only RTX 4070.

## 10.9 No Multi-Seed

Single seed.

## 10.10 No Theoretical Bound on Downstream Accuracy

We bound kernel approximation error, not task accuracy.

## 10.11 Preservation Gate Saturation

If α saturates, the model degenerates.

## 10.12 Code Quality

Research-grade.

## 10.13 Ethical Considerations

See Appendix E.

---