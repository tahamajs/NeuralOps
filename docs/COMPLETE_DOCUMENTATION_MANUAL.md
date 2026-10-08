# NeuralOps Complete API & Architecture Reference Manual

**Package:** `neuralops`  
**Current Release:** `v1.1.1`  
**Author:** Taha Majlesi (<tahamaj4@gmail.com>)  
**PyPI Distribution:** [https://pypi.org/project/neuralops/1.1.1/](https://pypi.org/project/neuralops/1.1.1/)  
**GitHub Repository:** [https://github.com/tahamajs/SVD_linear_Attention](https://github.com/tahamajs/SVD_linear_Attention)  

---

## 1. Executive Framework Architecture

`NeuralOps` is a high-performance PyTorch library engineered for subquadratic attention mechanisms, post-training model compression (PTQ, SVD factorizations, structured pruning, KV cache optimizations), hardware-aware compilation, and high-throughput inference serving.

```
                               ┌────────────────────────────────────────────────────────┐
                               │             Pretrained Transformer Backbone            │
                               │        (BERT, RoBERTa, DistilBERT, GPT-2, LLaMA)       │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                                                     ▼
                  ┌───────────────────────────┐                         ┌───────────────────────────┐
                  │    ATTENTION REGISTRY     │                         │    COMPRESSION REGISTRY   │
                  │       (159 Modules)       │                         │       (73 Engines)        │
                  ├───────────────────────────┤                         ├───────────────────────────┤
                  │ • Linear / Recurrent (36) │                         │ • Low-Rank Factor (16)    │
                  │ • Hybrid Attention (26)   │                         │ • SVD Decomposition (7)   │
                  │ • Modern Attention (27)   │                         │ • Quantization (PTQ) (13) │
                  │ • Fast Triton Kernels (17)│                         │ • Pruning / Wanda (11)    │
                  │ • Sparse / Windowed (16)  │                         │ • KV Cache Opt (11)       │
                  │ • SVD Attention (9)       │                         │ • Hybrid Pipelines (10)   │
                  │ • State Space / SSM (6)   │                         │ • Distillation (KD) (5)   │
                  │ • Adaptive / Hardware (12)│                         └─────────────┬─────────────┘
                  └─────────────┬─────────────┘                                       │
                                │                                                     │
                                └──────────────────────────┬──────────────────────────┘
                                                           ▼
                                ┌─────────────────────────────────────────────────────┐
                                │             Runtime & Serving Execution             │
                                │ • ContinuousBatcher, PagedKVCache, SpeculativeDec   │
                                │ • Triton Kernels + 100% Numerical PyTorch Fallbacks │
                                └─────────────────────────────────────────────────────┘
```

---

## 2. Core Registries

All modules are registered dynamically using string identifiers:

```python
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

# Inspect registered counts
print(f"Attention modules: {len(ATTENTION_REGISTRY.keys())}")   # 159
print(f"Compressor engines: {len(COMPRESSOR_REGISTRY.keys())}") # 73
```

### Module Retrieval & Instantiation
```python
# Instantiate an attention module
attn_cls = ATTENTION_REGISTRY["svd_linear_attention"]
attn_layer = attn_cls(hidden_size=768, num_heads=12, rank=32)

# Instantiate a compression engine
comp_cls = COMPRESSOR_REGISTRY["svd_linear"]
compressor = comp_cls(target_rank=32)
```

---

## 3. Attention Zoo Family Breakdown (159 Modules)

| Family | Count | Notable Modules | Mathematical Foundations |
| :--- | :---: | :--- | :--- |
| **Linear Attention** | 36 | `linear_attention`, `cosformer`, `kathryn`, `chela_slay`, `performer`, `rfa` | Approximates $\mathrm{softmax}(QK^T)V$ via kernel feature maps $\phi(Q)\phi(K)^TV$ achieving $\mathcal{O}(N)$ memory & compute. |
| **Hybrid Attention** | 26 | `spectron_fp8_attention`, `gated_hybrid_kernel_attention`, `fusion_attn` | Combines global subquadratic linearized heads with sliding-window localized softmax heads. |
| **Modern Attention** | 27 | `rope_attention`, `alibi_attention`, `gqa`, `mqa`, `cross_attention` | Implements Rotary Position Embedding (RoPE), ALiBi bias, Grouped Query Attention (GQA), and Multi-Query Attention. |
| **Fast Triton Kernels**| 17 | `flash_attention_v2`, `flash_decoding`, `fused_rope`, `paged_attention` | Tiled memory-hierarchy fused GPU execution via Triton JIT kernels with PyTorch CPU fallbacks. |
| **Sparse Attention** | 16 | `sliding_window_attention`, `block_sparse`, `cluster_attention`, `moe_deltakv_paged` | Restricts computation to local bands, dilated strides, or dynamic top-$k$ clustered attention tokens. |
| **SVD Attention** | 9 | `svd_linear_attention`, `adaptive_svd_attention`, `asvd_attention`, `fused_svd` | Factorizes projection weights ($W_q, W_k, W_v, W_o$) into low-rank representations $(U \Sigma V^T)$. |
| **Adaptive Attention** | 7 | `adaptive_span_attention`, `adaptive_sparse_attention`, `block_transformer_hiba` | Dynamically adjusts context window length or sparsity thresholds per head based on token entropy. |
| **State Space (SSM)** | 6 | `mamba_svd_flash`, `nexus`, `state_update` | Implements selective state updates $h_t = \bar{A}h_{t-1} + \bar{B}x_t$ for linear recurrence with zero attention matrix storage. |
| **Hardware-Aware** | 5 | `hopper_tma`, `distributed_ring_attention`, `tensor_core_gemm` | Asynchronous memory copy, warp-specialized pipeline, and ring-based sequence parallelism. |
| **Approximate** | 6 | `random_projection_attention`, `nystromformers`, `linformer` | Low-rank Nyström approximations and Johnson-Lindenstrauss random projections. |

---

## 4. Model Compression Zoo Breakdown (73 Engines)

| Family | Count | Primary Methods | Typical Compression & Speedup |
| :--- | :---: | :--- | :--- |
| **Low-Rank Factorization** | 16 | `a3_compressor`, `lora`, `dobi_compressor`, `asvd`, `factored_svd` | $2.0\times - 4.0\times$ parameter reduction, $1.5\times - 3.2\times$ latency speedup. |
| **Quantization (PTQ)** | 13 | `gptq`, `awq`, `smoothquant`, `astro_quant`, `fp8_quantizer` | INT8 / INT4 weight & activation quantization with second-order Hessian error compensation. |
| **Pruning (Structured/Sparse)** | 11 | `wanda`, `sparsegpt`, `structured_magnitude`, `movement_pruning` | $50\% - 70\%$ structured 2:4 or unstructured sparsity without retraining. |
| **KV Cache Compression** | 11 | `delta_kv`, `paged_cache`, `snapkv`, `h2o_heavy_hitter`, `streaming_llm` | $3.5\times - 8.0\times$ KV footprint reduction in high-concurrency long-context serving. |
| **Hybrid Compression** | 10 | `svd_quant_hybrid`, `prune_and_rank`, `quantized_low_rank` | Fuses singular-value rank reduction with non-uniform low-bit integer quantization. |
| **SVD Algorithms** | 7 | `adasvd_compressor`, `adaptive_ranks`, `energy_svd`, `direct_svd` | Optimal truncated singular value decomposition with cumulative energy thresholding. |
| **Knowledge Distillation** | 5 | `cross_entropy_kd`, `hidden_state_alignment`, `attention_transfer` | Transfers teacher representations into student compressed linear backbones. |

---

## 5. Triton Kernels & Fallbacks Architecture

NeuralOps ships with **19 custom GPU kernels** in `neuralops.core.triton_kernels`. 
Every kernel includes a validated **PyTorch CPU/CUDA fallback** enabling seamless development on Apple Silicon (macOS) and CI/CD without hardware lock-in.

```python
from neuralops.core.triton_kernels.linear_attn import linear_attn_pytorch_fallback
from neuralops.core.triton_kernels.gqa import gqa_pytorch_fallback
from neuralops.core.triton_kernels.softplus import softplus_fn

# PyTorch fallback works out-of-the-box on CPU / MPS
import torch
q = torch.randn(2, 4, 128, 64)
k = torch.randn(2, 4, 128, 64)
v = torch.randn(2, 4, 128, 64)

out = linear_attn_pytorch_fallback(q, k, v)
assert out.shape == q.shape
```

---

## 6. End-to-End Tutorials & Code Examples

### 6.1 Attention Injection into HuggingFace Models
```python
import torch
from transformers import AutoModelForSequenceClassification
from neuralops.core.adapters import inject_attention

# 1. Load HuggingFace model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 2. Swap standard attention with SVD Linear Attention
model = inject_attention(model, attn_type="svd_linear_attention", rank=32)

# 3. Standard forward pass
x = torch.randint(0, 1000, (2, 64))
logits = model(x).logits
print("Output logits shape:", logits.shape)
```

### 6.2 Post-Training Compression (SVD / AWQ / Wanda)
```python
from transformers import AutoModelForSequenceClassification
from neuralops.compression_engine import get_compressor

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Instantiate SVD compressor
compressor = get_compressor("adaptive_ranks")(energy_threshold=0.95)

# Compress model weights
compressed_model = compressor.compress(model)
print("Compression complete!")
```

### 6.3 High-Throughput Streaming & Speculative Inference Engine
```python
from neuralops.inference.engine import NeuralOpsInferenceEngine

engine = NeuralOpsInferenceEngine(model, tokenizer)

# Speculative decoding
output_text = engine.speculative_generate(
    prompt="Explain linear attention in transformers:",
    draft_model=draft_model,
    max_new_tokens=100,
    temperature=0.7,
)
print("Generated:", output_text)
```

---

## 7. Verification & Testing

Verify installation and run the full non-GPU test suite:
```bash
# Verify registries
python3 -c "
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions, neuralops.compression_engine
assert len(ATTENTION_REGISTRY) == 159
assert len(COMPRESSOR_REGISTRY) == 73
print('Registry verified: 159 Attentions + 73 Compressors')
"

# Run pytest test suite
pytest neuralops/tests/ -v -m "not gpu and not slow"
```

---

## 8. Citation

```bibtex
@software{neuralops2026,
  title  = {NeuralOps: A Production-Grade Framework for Subquadratic Attention, Model Compression, and Hardware-Aware Transformer Optimization},
  author = {Majlesi, Taha},
  year   = {2026},
  url    = {https://github.com/tahamajs/SVD_linear_Attention}
}
```
