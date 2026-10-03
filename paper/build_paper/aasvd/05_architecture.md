# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support

| Architecture | Target layers | Conv1D? | Nested? |
|---|---|---|---|
| DistilBERT-base | `q_lin, k_lin, v_lin, out_lin` | No | No |
| RoBERTa-base | `query, key, value, output.dense` | No | Yes |
| GPT-2 | `c_attn, c_proj` | Yes | No |
| Qwen3.5-0.8B | `q_proj, k_proj, v_proj, o_proj` | No | No |

By default, AASVD targets **attention projections only** 
(`target_attention_only = True`). Set `target_attention_only = False` to 
also compress FFN layers.

## 5.2 Layer Targeting Rules

AASVD protects the following layers from compression:

| Pattern | Reason |
|---|---|
| `lm_head` | Output projection tied to embeddings |
| `classifier` | Task-specific head, small |
| `head` | Same as above |
| `output_layer` | Task-specific |
| `score` | GPT-2 classification head |
| `pooler` | BERT pooler, not a projection |

The default targeting uses `should_compress_linear` for additional checks.

## 5.3 Handling Conv1D (GPT-2)

GPT-2 uses HuggingFace's `Conv1D`, which stores weights as `(in, out)` 
instead of `(out, in)`:

```python
# Standard PyTorch Linear
W = raw_w  # shape (out, in)

# GPT-2 Conv1D
W = raw_w.t()  # transpose to (out, in)
```

AASVD detects Conv1D via:

```python
is_conv1d = (
    "conv1d" in cls_name 
    or "conv1d" in target_cls 
    or hasattr(module, "nf") 
    or hasattr(target_mod, "nf")
)
```

The transposed weight is used throughout. The factorized `SVDLinear` 
receives the same input/output shapes as the original Conv1D, so the 
rest of the model's forward pass works unchanged.

## 5.4 Handling LoRA-Wrapped Modules

When the model has been wrapped with LoRA adapters (`LoRALinear`), AASVD 
extracts the `base_layer`:

```python
target_mod = getattr(module, "base_layer", module)
```

The factorized layer replaces `module.base_layer`, preserving the outer 
LoRA parameters. This ensures that any existing LoRA adapters remain 
attached and trainable.

## 5.5 Replacing a Layer

Replacement uses `replace_module` from the framework:

```python
if hasattr(module, "base_layer"):
    module.base_layer = svd_layer
else:
    replace_module(self.model, name, svd_layer)
```

After replacement, all parameters in fallback layers are frozen 
(`requires_grad = False`) to prevent accidental training.

## 5.6 Causality and Streaming

For decoder models (GPT-2, Qwen), AASVD's calibration is compatible with 
causal forward passes:

- Forward hooks fire during the standard causal forward pass.
- No future-token leakage occurs because the model's own attention 
  masking is respected.

**Verification.** We check that gradient of output at position $i$ w.r.t. 
input at position $j > i$ is zero, both before and after injection.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject AASVD into a full model
```
Input:  model M, rank r, threshold τ
        calibration_texts D
Output: compressed model M'

1:  targets ← [(name, module) for name, module in M if is_target(module)]
2:  hooks ← []
3:  for (name, module) in targets do
4:      hooks.append(register_hook(module, accumulate_C_X(name)))
5:  end for
6:
7:  # Calibration forward pass
8:  tok_inputs ← tokenize(D)
9:  M.eval()
10: with no_grad(): M(tok_inputs)
11:
12: for h in hooks: h.remove()
13:
14: # Precompute C_X^{1/2}, C_X^{-1/2} for each layer
15: cov_cache ← {}
16: for (name, module) in targets do
17:     cov_cache[name] ← compute_sqrt_and_inv(C_X[name])
18: end for
19:
20: # Compress each layer
21: for (name, module) in targets do
22:     svd_layer, err, r_eff ← compress_linear(module, cov_cache[name])
23:     if svd_layer is not None and err ≤ τ:
24:         replace(module, svd_layer)
25:     else:
26:         freeze_all(module)
27: end for
28:
29: # Free covariances
30: cov_cache.clear()
31: torch.cuda.empty_cache()
32:
33: return M'
```

## 5.8 The `SVDLinear` Module

```python
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, ...):
        super().__init__()
        self.V = nn.Parameter(torch.empty(rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
    
    def forward(self, x):
        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        if self.bias is not None:
            out += self.bias
        return out * self.output_scale
```

## 5.9 Interaction with LoRA Recovery

After AASVD compression, the model is passed to the LoRA recovery pipeline:

1. **AASVD** compresses → factorized `SVDLinear` layers.
2. **Framework** attaches LoRA adapters (`lora_A`, `lora_B`).
3. **Recovery** trains only LoRA parameters.
4. **Evaluation** uses the fine-tuned model.

**Note.** AASVD can also instantiate LoRA adapters *inside* the `SVDLinear` 
module (via the `lora_rank` parameter). This is used when the framework 
expects LoRA to be part of the compression module.

## 5.10 Registry

```python
@COMPRESSOR_REGISTRY.register("aasvd")
@COMPRESSOR_REGISTRY.register("aa_svd")
@COMPRESSOR_REGISTRY.register("aa_svd_compressor")
class AASVDCompressor(BaseCompressor):
    ...
```

## 5.11 Failure Modes and Fallbacks

| Failure | Detection | Response |
|---|---|---|
| Reconstruction error > $\tau$ | Frobenius check | Fallback to original |
| Covariance singular | `eigh` error | Skip weighting |
| No tokenizer | `tokenizer is None` | Use unweighted SVD |
| No calibration texts | Empty list | Use default corpus |
| CUDA OOM during SVD | `RuntimeError` | Retry on CPU |
| NaN in covariance | `torch.isnan` | Skip layer |
