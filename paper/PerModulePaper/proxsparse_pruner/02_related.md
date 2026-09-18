# 2. Related Work and Theoretical Foundations

## 2.1 Model Compression Paradigms
Compressing deep neural networks without sacrificing task accuracy has produced three predominant methodologies:
1. **Low-Rank Factorization**: Exploiting the low intrinsic dimensionality of trained weight matrices by decomposing dense projections $W$ into compact matrix products $U V^\top$ via truncated Singular Value Decomposition (SVD), randomized range finders, or tensor-train factorizations.
2. **Structural and Semi-Structured Sparsity**: Enforcing structured zero patterns (such as block pruning or NVIDIA 2:4 semi-structured sparsity) that map directly onto hardware tensor cores to yield tangible wall-clock acceleration rather than theoretical unindexed sparsity.
3. **Quantization and Codebook Decomposition**: Mapping full-precision floating-point parameters (FP32/BF16) into reduced integer representations (INT8/INT4) or clustered codebooks (k-means, vector quantization) to alleviate memory bandwidth pressure.

`proxsparse_pruner` builds upon these foundational principles, tailoring transformations specifically to the sensitivity profiles of transformer attention and projection sublayers.

## 2.2 Linear and Efficient Attention Mechanisms
In attention acceleration, extensive literature addresses the quadratic bottleneck of softmax attention. Kernel-based linear attention methods approximate the softmax similarity $\exp(Q K^\top / \sqrt{d})$ via feature map inner products $\phi(Q) \phi(K)^\top$, transforming the attention operation from $\mathcal{O}(N^2)$ into $\mathcal{O}(N d^2)$ through matrix associativity. State space models (SSMs) and gated linear recurrence formulate token interactions as continuous-time dynamical systems or parallel prefix scans. `proxsparse_pruner` incorporates these advances to sustain high token throughput across extended sequence contexts.
