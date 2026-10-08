# Inference Engine Tutorial

Execute high-throughput token decoding with sampling, speculative decoding, and continuous batching:

```python
from neuralops.inference.engine import NeuralOpsInferenceEngine

engine = NeuralOpsInferenceEngine(model)
output_ids = engine.generate(input_ids, top_k=50, top_p=0.9, repetition_penalty=1.1)
```
