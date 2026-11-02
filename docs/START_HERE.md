# Getting Started with NeuralOps

Welcome to **NeuralOps**, the high-performance AI systems framework providing **131 Attention Mechanisms**, **72 Neural Compressors**, and **26 Canonical Triton JIT Kernels**.

## Quick Installation

```bash
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps
pip install -e .
```

## Core Workflows

1. **Inspect Model Architecture**:
   ```python
   from neuralops import inspect_architecture
   inspect_architecture(model)
   ```

2. **Attention Surgery**:
   ```python
   from neuralops import replace_attention
   replace_attention(model, attention_type="cosformer", calibrate=True)
   ```

3. **Model Compression**:
   ```python
   from neuralops import compress
   engine = compress("hnc_svd", rank_ratio=0.5)
   compressed_model = engine.compress(model)
   ```
