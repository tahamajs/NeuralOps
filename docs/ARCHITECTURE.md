# Architecture & Design

## System Hierarchy
NeuralOps is structured into core modular layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuralOps Public APIs                    │
│    (neuralops.main, eval_all_models_datasets, eval_pipeline)│
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│     Attention Modules        │        │     Compression Engines      │
│  (130 Canonical Mechanisms)  │        │   (72 Compression Models)    │
│  - Adaptive (10)             │        │  - SVD / Adaptive SVD        │
│  - Linear / Kernel (15)      │        │  - Tensor Decomposition      │
│  - Modern / GQA (11)         │        │  - Pruning / Sparsification  │
│  - Gated / DeltaNet (8)      │        │  - Quantization / FP8        │
│  - Hybrid / Sliding (15)     │        └──────────────┬───────────────┘
│  - Sparse / MoE (12)         │                       │
│  - SSM / Recurrent (19)      │                       │
│  - SVD Projections (17)      │                       │
│  - Fast Kernels / Dense (23) │                       │
└──────────────┬───────────────┘                       │
               │                                       │
               └───────────────────────┬───────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │     neuralops.core.utils     │
                       │  - rope, gqa, masks, config  │
                       │  - lora, shared helpers      │
                       └──────────────┬───────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │  Underlying PyTorch / CUDA   │
                       └──────────────────────────────┘
```

## Attention Data Flow
```
Host Transformer Layer
         │
         ▼
[Input Hidden States: (B, S, D)]
         │
         ├──> [Q, K, V Projections (Dense or SVD Low-Rank)]
         │
         ├──> [RoPE / Position Encoding (core.utils.rope)]
         │
         ├──> [GQA Group Expansion if MQA/GQA (core.utils.gqa)]
         │
         ├──> [Attention Kernel: Causal / Linear / Sparse / Hybrid]
         │
         ├──> [Mask Handling (core.utils.masks)]
         │
         ▼
[Output Projection (O_proj)] ──> [Residual & LayerNorm]
```
