# 🏛️ NEURALOPS PART 4 — SURGICAL MODEL TRANSPLANT & LOSSLESS SERIALIZATION ENGINE

**Architecture**: `NeuralOps` (`SVD_linear_Attention` v1.3.7)  
**Core Modules**: [`neuralops/models/surgical_io.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/models/surgical_io.py), `neuralops/surgery/`  
**Target Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU 8GB (Ada Lovelace, SM 8.9)  

---

## 1. The One-Line Surgical Transplant Engine

The surgical transplant engine enables in-place replacement of standard $O(T^2)$ multi-head self-attention mechanisms with any of the 131 subquadratic or compressed attention modules in `NeuralOps`:

```python
from neuralops import replace_attention_layer

# Surgically transplants all attention layers to subquadratic SVD attention in-place
model = replace_attention_layer(
    model,
    target_attention="svd_linear_attention",
    target_layers="all",
    rank=32,
    hnc_align=True,
)
```

### Module Tree Traversal & Wrapping
1. **Recursive Scan**: Traverses the `nn.Module` tree, matching known attention module signatures (`LlamaAttention`, `MistralAttention`, `Qwen2Attention`, `BertSelfAttention`, `GPT2Attention`, etc.).
2. **Dimension Alignment**: Automatically extracts `hidden_size`, `num_heads`, `head_dim`, and positional embedding configuration.
3. **Weight Transfer**: Decomposes or copies existing projection weights ($W_q, W_k, W_v, W_o$) into the new attention module.
4. **AttentionWrapper Invariant**: Wraps the transplanted module in an [`AttentionWrapper`](file:///Users/tahamajs/Documents/uni/Project/neuralops/models/surgical_io.py), which intercepts HuggingFace keyword arguments (`attention_mask`, `position_ids`, `past_key_value`, `use_cache`, `output_attentions`) and normalizes tuple output conventions.

---

## 2. Lossless Bit-Exact Serialization (`surgical_io.py`)

### The State Dict Mismatch Problem
When a surgically modified model is saved using standard `torch.save(model.state_dict())`, standard loading fails because:
- Module hierarchies have changed (`model.layers.0.self_attn` is now `model.layers.0.self_attn.wrapped_attn`).
- Projection parameter shapes are factorized ($[d, d] \to [d, r]$ and $[r, d]$).
- HuggingFace's `from_pretrained` expects original dense weights.

### The `surgical_io` Solution
[`surgical_io.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/models/surgical_io.py) introduces paired lossless serialization:
- **`save_surgical_checkpoint(model, save_directory)`**:
  1. Emits model weights into `pytorch_model.bin` or `model.safetensors`.
  2. Emits an exhaustive JSON manifest (`surgical_manifest.json`) recording every architectural modification.
- **`load_surgical_checkpoint(base_model, save_directory)`**:
  1. Parses `surgical_manifest.json`.
  2. Re-executes the exact surgical transplants on the fresh base model.
  3. Loads weights into the re-constituted architecture.
  4. Restores transient wrapper flags (`is_nested_encoder`, `is_decoder`, `return_tuple`, `layer_idx`).
  5. **Guarantees bit-exact output parity**:
     $$\|y_{\text{reloaded}} - y_{\text{original}}\|_\infty < 10^{-6}$$

### `surgical_manifest.json` Schema
```json
{
  "version": "1.3.7",
  "base_architecture": "LlamaForCausalLM",
  "transplants": [
    {
      "layer_name": "model.layers.0.self_attn",
      "target_attention": "svd_linear_attention",
      "rank": 32,
      "use_hnc": true,
      "wrapper_flags": {
        "is_decoder": true,
        "return_tuple": true,
        "layer_idx": 0
      }
    }
  ]
}
```

---

## 3. Foundation Model Architecture Adapters

Pre-built adapters handle specific parameter layouts across all major foundation models:

| Architecture | Attention Class | QKV Layout | Positional Encoding | Special Features |
|:---|:---|:---|:---|:---|
| **LLaMA 1/2/3** | `LlamaAttention` | Decoupled `q_proj`, `k_proj`, `v_proj`, `o_proj` | Rotary Position Embeddings (RoPE) | SwiGLU MLP, RMSNorm |
| **Mistral / Mixtral**| `MistralAttention`| Grouped-Query Attention | Sliding Window RoPE | Sparse MoE routing |
| **Qwen 2.5 / 3.5** | `Qwen2Attention` | GQA + Latent Attention | Dynamic Context Scaling | Tied input/output embeddings |
| **GPT-2 / NeoX** | `GPT2Attention` | 1D Conv Fused `c_attn` $[d, 3d]$ | Learned Absolute 1D | Transposed GEMM weights |
| **RoBERTa / BERT** | `BertSelfAttention`| Separate `query`, `key`, `value` | Absolute Positional Embeddings | Bidirectional non-causal masking |
| **T5 / Flan-T5** | `T5Attention` | Split Q, K, V + Relative Bias | Relative Position Buckets | Encoder-Decoder cross-attention |

---
*Part 4 certified under `tests/test_huggingface_export_roundtrip.py` and `tests/test_exhaustive_surgery_8_architectures.py`.*
