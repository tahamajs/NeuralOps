# GoldFinch-AHN (Adaptive Hybrid Network)

## Overview

**GoldFinch-AHN** is a production-hardened hybrid attention architecture combining short-term multi-head sliding window attention with long-term GatedDeltaNet recurrent associative memory. The mechanism leverages a learned, input-dependent dynamic gating layer initialized to provide stable representation transfer while smoothly arbitrating between local high-resolution contexts and persistent sequence-level states.

### Architectural Highlights
- **Local Branch:** Multi-head sliding window self-attention capturing fine-grained local syntactic dependencies with $O(N \cdot W)$ computational and memory complexity.
- **Global Recurrent Memory:** GatedDeltaNet memory update mechanism maintaining continuous sequence representations across long horizons.
- **Dynamic Input-Dependent Gating:** Token-wise learned routing mechanism $\mathbf{g}_t = \sigma(\mathbf{W}_g \mathbf{x}_t + \mathbf{b}_g)$ with conservative bias initialization ($\text{bias} \approx 1.8$) for robust fine-tuning convergence.
- **Cross-Model Compatibility:** Hardened weight-transfer support across DistilBERT, RoBERTa/BERT nested encoders, GPT-2 causal decoders, and modern GQA architectures (LLaMA/Qwen).

---

## Mathematical Formulation

Given input sequence $\mathbf{X} \in \mathbb{R}^{B \times L \times D}$:

1. **Sliding Window Projection:**
   $$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
   $$\mathbf{Y}_{\text{local}} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{window}}\right)\mathbf{V}$$

2. **GatedDeltaNet Recurrent Memory Update:**
   $$\mathbf{S}_t = \beta_t \mathbf{S}_{t-1} + (1 - \beta_t)(\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^T$$
   $$\mathbf{Y}_{\text{global}, t} = \mathbf{S}_t \mathbf{q}_t$$

3. **Dynamic Fusion:**
   $$g_t = \sigma(\mathbf{x}_t \mathbf{W}_g + b_g)$$
   $$\mathbf{Y}_t = g_t \mathbf{Y}_{\text{local}, t} + (1 - g_t) \mathbf{Y}_{\text{global}, t}$$
   $$\mathbf{O} = \mathbf{Y}\mathbf{W}_O$$

---

## Configuration & Hyperparameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_model` | `int` | `768` | Hidden model dimensionality |
| `num_heads` | `int` | `12` | Number of attention heads |
| `window_size` | `int` | `64` | Sliding window receptive field for local attention |
| `gate_init_bias` | `float` | `1.8` | Initial bias for gating logits (biases initial routing toward local window) |
| `is_causal` | `bool` | `False` | Enforces strictly causal masking for autoregressive decoders |
| `rank` | `int` | `None` | Optional low-rank adapter bottleneck rank |

---

## Usage Examples

```python
import torch
from framework.attentions.goldfinch_ahn import GoldFinchAHNBlock, inject_goldfinch_ahn

# Direct instantiation
block = GoldFinchAHNBlock(
    d_model=768,
    num_heads=12,
    window_size=64,
    gate_init_bias=1.8,
    is_causal=False
)

x = torch.randn(2, 128, 768)
out, attn_weights = block(x)

# Model injection
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
injected_model = inject_goldfinch_ahn(model)
```

---

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **91.06%** | +0.00% | 0.9128 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-goldfinch](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-goldfinch) | 2026-09-19 |
