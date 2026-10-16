# Decay Linear Attn: Formal Architectural Specification and Empirical Benchmark Evaluation

**Authors:**
- Taha Majs¹ (Corresponding Author: `tahamajs@uni.edu`)
- NeuralOps Research Team² (`efficiency-lab@uni.edu`)

**Affiliations:**
- ¹ Department of Computer Science, Advanced Machine Learning & Efficiency Laboratory
- ² Machine Learning Systems & Hardware-Software Co-Design Group

**Preprint Date:** 2026-10-16  
**Module Name:** `decay_linear_attn`  
**Class Name:** `DecayLinearAttention`  
**Base Architecture:** `BaseAttention`  
**Primary Domain:** Attention Acceleration  
**License:** MIT  

---

## Abstract
This paper delivers a formal architectural specification, mathematical derivation, and comprehensive empirical validation of `decay_linear_attn`. As transformer models scale to billions of parameters, standard dense linear transformations and quadratic softmax attention operators impose prohibitive memory bandwidth and compute bottlenecks. Integrated into the unified transformer compression and attention acceleration framework, `decay_linear_attn` provides an optimized execution pathway evaluated across standard encoder, causal decoder, and modern generative backbones (DistilBERT, RoBERTa, GPT-2, and Qwen architectures). We formalize the complete algebraic transformations, quantify approximation guardrails, provide concrete ASCII architectural diagrams, and report empirical accuracy preservation, GPU memory ceilings, and runtime step latencies from standardized sweep benchmarks.
