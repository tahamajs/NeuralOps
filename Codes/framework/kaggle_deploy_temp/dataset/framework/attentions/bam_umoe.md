# BAM UMoE (Unified Mixture of Experts) Attention Block

A radical replacement for standard self‑attention: token‑wise Mixture of Experts for QKV projections and a second MoE for feed‑forward, skipping pairwise token interactions entirely.

---

## What it does

Instead of computing attention scores between all token pairs, this block processes each token **independently** through two successive Mixture‑of‑Expert layers:

1. **QKV‑MoE** – takes the hidden state \(\mathbf{x} \in \mathbb{R}^{B \times L \times d}\) and outputs a combined \(\mathbf{QKV} \in \mathbb{R}^{B \times L \times 3d_h H}\) using a top‑\(k\) gated mixture of \(E\) experts.  
2. **FFN‑MoE** – projects the intermediate representation back to \(d\) using another top‑\(k\) MoE.

Residual connections and layer normalisation are applied around both blocks, mimicking a standard Transformer layer but with **no token‑token interaction**. The result is a linear‑complexity, highly parameter‑efficient module that acts as a drop‑in replacement for DistilBERT’s attention layer.

---

## Mathematical Formulation

### Unified Mixture of Experts (`UnifiedMoE`)

For an input \(\mathbf{x} \in \mathbb{R}^{B \times L \times d}\), the router computes logits:

\[
\mathbf{R} = \mathbf{x} \mathbf{W}_{\text{router}}^\top \quad \in \mathbb{R}^{B \times L \times E},
\]

where \(\mathbf{W}_{\text{router}} \in \mathbb{R}^{E \times d}\). For each token position \((b,\ell)\), the top‑\(k\) largest logits are selected:

\[
\mathcal{T} = \operatorname{TopK}(\mathbf{R}_{b,\ell}, k).
\]

Softmax normalisation is applied to these \(k\) scores:

\[
g_i = \frac{\exp(R_{b,\ell,i})}{\sum_{j \in \mathcal{T}} \exp(R_{b,\ell,j})}, \quad i \in \mathcal{T}.
\]

Each expert is a two‑layer MLP:

\[
E_i(\mathbf{x}) = \mathbf{W}_2^{(i)} \operatorname{GELU}(\mathbf{W}_1^{(i)} \mathbf{x}),
\]

where \(\mathbf{W}_1^{(i)} \in \mathbb{R}^{(2d_{\text{out}}) \times d}\) and \(\mathbf{W}_2^{(i)} \in \mathbb{R}^{d_{\text{out}} \times (2d_{\text{out}})}\), with \(d_{\text{out}}\) being the output dimension.

The final output for a token is the weighted sum of the selected expert outputs:

\[
\mathbf{y} = \sum_{i \in \mathcal{T}} g_i \, E_i(\mathbf{x}).
\]

All tokens are processed in parallel via tensor gathering and weighting (no loops).

### BAM‑UMoE Block

The block receives hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times d}\):

1. **QKV‑MoE** produces \(\mathbf{QKV} = \text{UnifiedMoE}_{\text{qkv}}(\mathbf{H})\) with output size \(3 \cdot H \cdot d_h\).  
2. **Residual + Norm**: \(\tilde{\mathbf{H}} = \text{LayerNorm}(\mathbf{H} + \text{Dropout}(\mathbf{QKV}))\).  
3. **FFN‑MoE**: \(\mathbf{F} = \text{UnifiedMoE}_{\text{ffn}}(\tilde{\mathbf{H}})\) with output size \(d\).  
4. **Residual + Norm**: \(\mathbf{O} = \text{LayerNorm}(\tilde{\mathbf{H}} + \text{Dropout}(\mathbf{F}))\).

No attention mask is needed because there is no pairwise interaction; padding tokens are simply processed as any other token.

---

## Implementation

- **`UnifiedMoE`**: holds a linear router `self.router` and an `nn.ModuleList` of expert MLPs. The forward method uses `torch.topk`, softmax over selected experts, computes all expert outputs via a `torch.stack`, then gathers and combines.
- **`BAMUMoEBlock`**: inherits from `BaseAttention`. Contains:
  - `self.qkv_moe`: a `UnifiedMoE` with output `3 * all_head_size`.
  - `self.ffn_moe`: a `UnifiedMoE` with output `hidden_size`.
  - `self.norm1`, `self.norm2`: `nn.LayerNorm`.
  - `self.dropout`: dropout.
  - `self.output_proj`: a final linear projection (currently unused in the forward pass? Actually it’s defined but not used in the provided code; the block returns the output of `norm2` directly). Possibly a leftover.

The class is registered as `"bam_umoe"` in the `ATTENTION_REGISTRY`.

---

## How to Use

Configure your compression config with the MoE parameters:

```python
comp_config.num_experts = 8        # number of experts in each MoE
comp_config.top_k = 2              # top‑k experts selected
comp_config.dropout = 0.1
```

The injection function (to be provided) would instantiate `BAMUMoEBlock` and replace `layer.attention`.

Manual creation:

```python
block = BAMUMoEBlock(
    config=model_config,          # e.g., DistilBertConfig
    rank=64,                      # (kept for compatibility, not used internally)
    num_experts=8,
    top_k=2,
    layer_id=layer_idx,
    num_layers=total_layers,
    old_attn=original_attention   # not used, but accepted
)
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 8 | Number of experts in each MoE layer |
| `top_k` | 2 | Number of experts activated per token |
| `dropout` | 0.1 | Dropout applied after each residual branch |

The block uses no attention mask, no heads, and no sequence‑length‑dependent operations—time complexity is \(\mathcal{O}(L \cdot k \cdot d^2)\) with a constant factor depending on experts.

---

## Important Notes

- **No token mixing**: This block completely removes the attention mechanism. Every token is processed independently; the model must rely on positional embeddings (already present in the input) for any sequence‑level information. This is extremely fast but may severely limit the ability to model long‑range dependencies.
- **Parameter efficiency**: The MoE structure allows a large total expert capacity (many parameters) while only a subset is activated per token, keeping the inference FLOPs moderate.
- **Zero‑shot initialisation**: Not performed; the experts are randomly initialised. For best results, the whole model should be fine‑tuned after injection.
- **`output_proj`**: Present but not used; might be intended for further projection.
- **Base class**: Inherits from `BaseAttention`, which may provide additional hooks (e.g., `get_ortho_loss` returning zero).

---

*Ideal for extreme throughput scenarios where local token processing is sufficient, and the original attention can be sacrificed.*