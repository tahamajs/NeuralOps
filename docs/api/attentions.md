# Attentions API Reference

NeuralOps registers **131 canonical subquadratic and linear attention mechanisms** unified under `neuralops.core.registry.ATTENTION_REGISTRY`.

Every attention mechanism conforms to the base interface `neuralops.core.base.BaseAttention` and can be instantiated directly or injected in-place into pre-trained transformer architectures.

---

## 1. Architectural Taxonomy

| Family | Count | Key Modules | Computational Complexity | Memory Footprint |
| :--- | :---: | :--- | :---: | :---: |
| **Linear Attention** | 18 | `cosformer`, `performer`, `gla_attention`, `rpc_attention`, `regla_attention`, `linear_attention`, `rfa` | $\mathcal{O}(L \cdot D^2)$ | $\mathcal{O}(D^2)$ |
| **Gated & Recurrent** | 16 | `gated_delta_net`, `hgrn2_gated_deltanet`, `decay_linear_attention`, `gated_linear_attention` | $\mathcal{O}(L \cdot D^2)$ | $\mathcal{O}(D^2)$ |
| **SVD & Low-Rank** | 25 | `svd_linear_attention`, `asvd_attention`, `residual_svd_attention`, `minimal_svd`, `fused_svd_kernel` | $\mathcal{O}(L \cdot r)$ | $\mathcal{O}(r)$ |
| **State Space (SSM)** | 14 | `mamba_block`, `mamba_svd_flash`, `selective_ssm`, `samba_m3`, `upi_stuffed_mamba` | $\mathcal{O}(L \cdot D \cdot N)$ | $\mathcal{O}(D \cdot N)$ |
| **Hybrid & MoE** | 20 | `tri_branch_hybrid`, `bam_umoe`, `granite_hybrid`, `expert_choice_moe`, `samba_moe` | $\mathcal{O}(L \cdot D)$ | $\mathcal{O}(D)$ |
| **Sparse & Windowed** | 14 | `adaptive_sparse_attention`, `cluster_attention`, `nsa_attention`, `routing_attention`, `sliding_window` | $\mathcal{O}(L \cdot W)$ | $\mathcal{O}(W)$ |
| **Modern & Positional**| 18 | `alibi`, `mla_attention`, `gqa_attention`, `roformer_attention`, `chela_slay` | $\mathcal{O}(L^2)$ or $\mathcal{O}(L)$ | $\mathcal{O}(L)$ |
| **Dense Baselines** | 6 | `full_attention`, `standard_mha`, `zeros_attention` | $\mathcal{O}(L^2)$ | $\mathcal{O}(L^2)$ |

---

## 2. Instantiating Attention Modules

```python
import torch
import neuralops

# Create any attention mechanism by canonical name
attn = neuralops.create_attention(
    "gated_delta_net",
    embed_dim=768,
    num_heads=12,
)

# Forward pass
x = torch.randn(2, 64, 768)
output = attn(x)
print("Output shape:", output.shape)
```

---

## 3. In-Place Model Surgery

Transplant any attention mechanism into an existing Hugging Face model on-the-fly:

```python
from transformers import AutoModelForCausalLM
import neuralops

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Replace standard self-attention across all layers with Gated DeltaNet
student = neuralops.replace_attention(model, attention="gated_delta_net", calibrate=True)
```

---

## 4. Constant-Memory Autoregressive Decoding

```python
cache = neuralops.NeuralOpsStateCache()
new_token = torch.randn(1, 1, 768)

# Flat O(1) state accumulation per token
out_step = attn(new_token, past_key_value=cache, use_cache=True)
print(f"Cache memory footprint: {cache.memory_mb():.2f} MB")
```
