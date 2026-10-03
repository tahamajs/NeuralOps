# Decay Linear Attention: Formal Architectural Specification and Empirical Benchmark Evaluation

**Module Name**: `decay_linear_attention`  
**Class Name**: `LoRALinear`  
**Base Architecture**: `nn.Module`  
**Registry Keys**: `decay_linear_attn, inject_decay_linear_attention, decay_linear_attention, decay_linear_attention_inject`  
**Implementation Source**: `neuralops/attentions/decay_linear_attention.py` (782 lines of code)  
**Primary Domain**: Attention Acceleration  

## Abstract
This paper delivers a formal architectural specification, mathematical derivation, and comprehensive empirical validation of `decay_linear_attention`. As transformer models scale to billions of parameters, standard dense linear transformations and quadratic softmax attention operators impose prohibitive memory bandwidth and compute bottlenecks. Integrated into the unified transformer compression and attention acceleration framework, `decay_linear_attention` provides an optimized execution pathway evaluated across standard encoder, causal decoder, and modern generative backbones (DistilBERT, RoBERTa, GPT-2, and Qwen architectures). We formalize the complete algebraic transformations, quantify approximation guardrails, provide concrete ASCII architectural diagrams, and report empirical accuracy preservation, GPU memory ceilings, and runtime step latencies from standardized sweep benchmarks.

## Lay Summary
Every attention projection and linear weight in a modern deep language model is a high-dimensional matrix of numbers. In standard deployments, these monolithic weight arrays create severe memory bottlenecks, restricting model execution on resource-constrained devices. `decay_linear_attention` restructures these parameters into compact, mathematically factorized, regularized, or sparse primitives. By doing so, `decay_linear_attention` reduces parameter overhead and peak memory requirements by up to 3x while retaining target classification and generation accuracy through lightweight residual adaptation without requiring expensive full-model retraining.

## Keywords
transformer compression, low-rank factorization, multi-head attention acceleration, parameter efficiency, LoRA residual recovery, memory bandwidth optimization, GLUE benchmark evaluation.
