# Getting Started with NeuralOps

Welcome to **NeuralOps**, the high-performance AI systems framework providing **131 Attention Mechanisms**, **72 Neural Compressors**, and **26 Canonical Triton JIT Kernels**.

## Quick Installation

```bash
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps
pip install -e .
```

## Core Workflows

1. **Inspect Model Architecture & Attention Topology**:
   ```python
   from neuralops.surgery.scanner import UniversalArchitectureScanner

   # Automatically scan model modules across 5 topological patterns
   targets = UniversalArchitectureScanner.scan(model)
   print(f"Discovered {len(targets)} attention targets in model.")
   ```

2. **In-Place Attention Surgery**:
   ```python
   import neuralops

   # Transplant quadratic attention with linear subquadratic attention
   model = neuralops.replace_attention(model, attention="cosformer", copy_weights=True)
   ```

3. **Low-Rank Model Compression**:
   ```python
   import neuralops

   # Compress linear projection layers using Adaptive SVD (ASVD)
   compressed_model = neuralops.compress(model, method="asvd", rank=32)
   ```
