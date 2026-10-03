# 10. Limitations

This method does not eliminate the need for model-specific tuning. In some architectures, the interaction with RoPE or grouped-query attention may require additional calibration. The approach is also not universally optimal for very short sequences, where the overhead of decomposition or gating may outweigh the savings.
