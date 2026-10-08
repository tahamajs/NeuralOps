# 2. Related Work

## 2.1 Efficient Attention Mechanisms

Sparse attention methods such as Longformer and BigBird reduce the attention pattern to a subset of tokens while keeping a small number of global connections. Low-rank approaches such as Linformer and Nyströmformer compress the projection space to approximate the attention matrix. Kernel methods such as Performer and Linear Transformer replace the softmax kernel with a more tractable feature map, enabling near-linear complexity in some settings.

## 2.2 KV-Cache Compression

Methods such as H2O, StreamingLLM, and KV quantization focus on reducing the cost of caching keys and values during generation. These methods are generally complementary to attention approximation and can be combined with architectural changes when memory constraints are dominant.

## 2.3 Parameter-Efficient Fine-Tuning

LoRA freezes the base weights and learns low-rank adapters, making it a natural mechanism for recovery after replacing a pretrained attention module. AdaLoRA and DoRA improve the choice of rank or decomposition, but the core principle remains the same: adapt only a small trainable subset while preserving the base model.

## 2.4 Position of Our Work

{{MODULE_NAME}} differs from these methods in three ways. First, it targets architecture portability across flat, nested, and causal models without retraining the original module. Second, it preserves the pretrained attention weights and uses LoRA-like recovery only for adaptation. Third, it is explicitly benchmarked under a fixed evaluation protocol with honest failure reporting and implementation guardrails.
