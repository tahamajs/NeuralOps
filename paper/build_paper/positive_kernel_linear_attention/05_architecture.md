# 5. Architecture and Implementation Details

The `positive_kernel_linear_attention` module is implemented in the `neuralops` package, inheriting from standard architectural interfaces.

```
       [ Input Sequence: X (T x d) ]
                     |
       +-------------+-------------+
       |                           |
[ Projection Stage ]       [ Calibration Monitor ]
       |                           |
[ Optimized Kernel: positive_kernel_linear_attention ]   |
       |                           |
       +-------------+-------------+
                     |
         [ Output Sequence: Y ]
```

The underlying kernel execution supports automatic fallback between custom Triton JIT kernels, compiled CUDA C++ extensions, and high-precision native PyTorch references.
