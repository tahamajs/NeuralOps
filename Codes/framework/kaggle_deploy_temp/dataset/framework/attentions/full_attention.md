# FlashAttention‑3: Fast and Memory‑Efficient Exact Attention

**Abstract**

We present a drop‑in attention module that leverages the **FlashAttention** algorithm (Dao et al., 2022) to compute exact self‑attention with significantly reduced memory footprint and faster runtime. When the dedicated `flash_attn` library is available, the module calls its optimised CUDA kernel; otherwise it falls back to PyTorch’s built‑in `scaled_dot_product_attention` with FlashAttention‑compatible patterns. The module also supports optional FP8 computation for further speed‑ups on next‑generation hardware. It wraps the standard query, key, value, and output projections, making it a direct replacement for any Transformer attention layer. This paper describes the architectural integration, the underlying algorithms, and the practical considerations for using the module in a DistilBERT pipeline.

---

## 1. Introduction

Standard multi‑head self‑attention requires computing and storing an \(N \times N\) attention matrix, which is the main bottleneck for long sequences. The FlashAttention algorithm (Dao et al., 2022) restructures the computation by tiling over the sequence dimension and recomputing the softmax on‑the‑fly, thereby avoiding materialisation of the full attention matrix. This reduces memory usage from \(O(N^2)\) to \(O(N)\) and accelerates inference and training. The **FlashAttention‑3** module (named after the library version, though it invokes `flash_attn_func` generically) encapsulates this technique into a simple interface that can be injected into existing Transformer models like DistilBERT.

When the `flash_attn` Python package is not installed, the module gracefully degrades to PyTorch’s `scaled_dot_product_attention`, which also implements memory‑efficient attention (including a FlashAttention‑inspired kernel) when running on compatible GPUs. Thus, users obtain efficient attention in all environments without any code changes. An experimental FP8 mode is included to further push the performance envelope on hardware with native support (e.g., NVIDIA H100 GPUs).

---

## 2. Method

### 2.1 Standard Multi‑Head Projections

The module takes hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\) and applies linear projections:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q^\top, \quad
\mathbf{K} = \mathbf{H} \mathbf{W}_K^\top, \quad
\mathbf{V} = \mathbf{H} \mathbf{W}_V^\top,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{(H d_h) \times d_{\text{model}}}\) are square matrices (in DistilBERT, \(H d_h = d_{\text{model}}\)). The outputs are reshaped into the multi‑head layout \((B, H, L, d_h)\).

### 2.2 FlashAttention Algorithm (Background)

FlashAttention computes the exact attention output without storing the full attention matrix. It operates by splitting the sequence into blocks and accumulating statistics:

Let \(\mathbf{q}_i\), \(\mathbf{k}_j\), \(\mathbf{v}_j\) be blocks of queries and keys/values. For each query block, the algorithm iterates over key‑value blocks, maintaining a running maximum \(m\) and denominator \(\ell\) for numerical stability, and incrementally updates the output:

\[
m_{i}^{\text{new}} = \max(m_i, \text{rowmax}(\mathbf{S}_{ij})),
\]
\[
\mathbf{O}_i = \operatorname{diag}(e^{m_i - m_i^{\text{new}}}) \mathbf{O}_i + e^{ \text{rowmax}(\mathbf{S}_{ij}) - m_i^{\text{new}} } \mathbf{P}_{ij} \mathbf{V}_j,
\]
\[
\ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + e^{ \text{rowmax}(\mathbf{S}_{ij}) - m_i^{\text{new}} } \operatorname{rowsum}(\mathbf{P}_{ij}),
\]

where \(\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^\top / \sqrt{d_h}\) and \(\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - \text{rowmax}(\mathbf{S}_{ij}))\). This tiled approach yields exact softmax attention with constant memory in \(N\). The `flash_attn_func` kernel implements this for all heads and batches in parallel.

### 2.3 Integration in the Module

In the forward pass, after obtaining \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\):

- **If `flash_attn` is available**: the tensors are transposed from \((B, H, L, d_h)\) to \((B, L, H, d_h)\) (the expected layout), made contiguous, and passed to `flash_attn_func` with `causal=False`. The result is transposed back to \((B, H, L, d_h)\).  
  *Note:* The attention mask is **not** passed to `flash_attn_func` in this implementation; therefore, padding is not explicitly masked. This limitation is discussed later.

- **If `flash_attn` is unavailable**: PyTorch’s `scaled_dot_product_attention` is called with `q, k, v` in the \((B, H, L, d_h)\) layout. An additive mask is constructed from `attention_mask` (which is expected to be a tensor of 1s (valid) and 0s (padding)): `(1 - mask) * -1e9`. This provides correct padding behaviour.

### 2.4 Optional FP8 Mode

When `use_fp8=True` and CUDA is available, the query, key, and value tensors are cast to the FP8 data type `torch.float8_e4m3fn` before calling `flash_attn_func`. This can deliver further throughput gains on GPUs with FP8 tensor cores (e.g., H100), provided the `flash_attn` library supports FP8 inputs. The flag is off by default and should only be enabled on supported hardware.

### 2.5 Output Projection

After attention, the output tensor is reshaped to \((B, L, H d_h)\) and projected back to \(d_{\text{model}}\) via \(\mathbf{W}_O^\top\). Dropout is not explicitly applied inside this module; the surrounding model typically handles residual dropout.

---

## 3. Implementation Details

- **Class**: `FlashAttention3`, inheriting from `BaseAttention`, registered as `"flash3"`.
- **Constructor**: 
  - Accepts `config` (model configuration) and optional `rank` (unused) and `use_fp8`.
  - Defines four `nn.Linear` layers for Q, K, V, and output.
- **Forward**:
  - Input: `hidden_states` \((B, L, D)\), `attention_mask` (optional) for SDPA fallback.
  - Computes Q, K, V and reshapes to multi‑head.
  - If `FLASH_ATTN_AVAILABLE`: calls `flash_attn_func` (without mask); else uses `torch.nn.functional.scaled_dot_product_attention` with mask.
  - Returns `(output, None)`.

- **Mask handling**:
  - For the SDPA fallback, the mask is assumed to be a binary tensor with 1 for valid tokens. The code constructs `(1 - mask) * -1e9`, which is the correct additive mask for SDPA.
  - For the FlashAttention path, the mask is **ignored**, meaning all tokens attend to each other regardless of padding. This can be problematic for padded sequences. The user must either ensure no padding or extend the code to pass a mask to `flash_attn_func`, which does support an optional `key_padding_mask` argument.

- **Dependencies**: Requires `flash_attn` to be installed for the optimised path. Without it, PyTorch ≥ 1.13 (which includes `scaled_dot_product_attention`) is sufficient.

---

## 4. Usage and Configuration

The module is injected via the framework’s registry using the key `"flash3"`. Configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | `None` | Ignored; kept for interface compatibility. |
| `use_fp8` | `False` | Whether to cast Q, K, V to FP8 before FlashAttention (requires compatible hardware). |

Example manual insertion:

```python
attn = FlashAttention3(config=model_config, use_fp8=False)
layer.attention = attn
```

To enable FP8, set `use_fp8=True` and ensure the `flash_attn` version supports FP8.

---

## 5. Discussion

The FlashAttention‑3 module brings state‑of‑the‑art memory efficiency and speed to attention computation without any approximation. The integration is straightforward, preserving the original model’s projection weights (these can be initialised from pretrained weights if the injection code copies them, though not shown here). The fallback to PyTorch SDPA ensures broad compatibility.

**Limitations**:
- **Masking in FlashAttention path**: The current code does not pass the attention mask to `flash_attn_func`. This can lead to leakage from padding tokens. A simple fix would be to create a boolean padding mask of shape \((B, L)\) with `True` for valid tokens and pass it as `key_padding_mask=~mask_bool`. Most deployment scripts for DistilBERT use padded sequences, so this should be addressed.
- **FP8 constraints**: FP8 support is experimental and hardware‑dependent. It also introduces potential precision loss; it is intended for inference or large‑batch training on H100 GPUs.
- **No attention weights**: The module returns `None` for attention weights, consistent with the efficient attention paradigm where the full matrix is not stored. This is compatible with most downstream tasks, but some interpretability methods may require adaptation.

Despite these issues, the module is a valuable tool for scaling Transformers to longer sequences while retaining exact attention.

---

## 6. Conclusion

We have described the FlashAttention‑3 module, a drop‑in replacement for standard self‑attention that harnesses the FlashAttention algorithm for exact, memory‑efficient computation. With a robust fallback to PyTorch’s native efficient attention and an experimental FP8 mode, it offers a practical path to faster training and inference for Transformer models. Future improvements will incorporate proper masking support in the FlashAttention path and explore the integration of more advanced FlashAttention variants.