# Compression Pipeline Tutorial

NeuralOps provides production quantizers and pruners:

- **GPTQ**: Second-order inverse Hessian compensation.
- **AWQ**: Activation-aware salient channel protection.
- **SmoothQuant**: Migration of activation outliers into weights.
- **Wanda**: Pruning by $|W| \cdot ||X||_2$.

```python
from neuralops.compression_engine.quantization.gptq import quantize_model_gptq

quantized_model = quantize_model_gptq(model, calib_batches, bits=4)
```
