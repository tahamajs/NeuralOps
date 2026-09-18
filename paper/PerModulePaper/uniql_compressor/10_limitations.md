# 10. Limitations

## 10.1 Operational Boundaries
While `uniql_compressor` provides competitive efficiency gains, certain constraints apply:
1. **Calibration Sensitivity**: Performance depends upon the diversity and distribution of calibration tokens.
2. **Architecture Specificity**: Certain parameter configurations require tuning when moving from bidirectional encoder models to autoregressive decoders.
3. **Hardware Alignment**: Semi-structured 2:4 sparsity requires Ampere or newer NVIDIA GPU architectures to realize maximum tensor core acceleration.
