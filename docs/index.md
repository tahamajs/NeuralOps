# 🚀 NeuralOps: Unified Platform for Subquadratic Attention and Compression

**NeuralOps** (`SVD_linear_Attention` v1.3.7) is a unified, high-performance deep learning acceleration platform bringing together:
- **131 Canonical Attention Mechanisms**: Linear attention, state-space models (Mamba/SSM), gated retention, RWKV, Cosformer, Longformer, and hybrid architectures.
- **72 Compression Engines**: Low-rank matrix decomposition (FWSVD, AdaSVD, ASVD, Direct SVD), gradient/Hessian pruning (SparseGPT, Wanda), and INT4/FP8 quantization (GPTQ, SmoothQuant, QuaRot).
- **26 Triton JIT Kernel Modules**: High-performance fused operators delivering up to $4.60\times$ speedups over standard PyTorch on NVIDIA Ada Lovelace and Hopper GPUs.

## Key Capabilities
1. **One-Line In-Place Surgery**: Drop-in attention and linear layer replacement without re-architecting your model.
2. **Bit-Exact Hugging Face Roundtrip**: Zero weight drift between exported and reloaded model checkpoints.
3. **Serving Adapters**: Production integrations for vLLM, Text Generation Inference (TGI), SGLang, and TensorRT-LLM.
