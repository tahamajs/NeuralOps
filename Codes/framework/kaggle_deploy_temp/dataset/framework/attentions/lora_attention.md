# LoRA Attention: Low‑Rank Adaptation of Attention Projections for Parameter‑Efficient Fine‑Tuning

**Abstract**

We present **LoRA Attention**, a module that applies Low‑Rank Adaptation (LoRA) to the query, key, value, and output projections of a standard multi‑head self‑attention layer. Instead of updating the full pretrained weight matrices, LoRA inserts trainable low‑rank matrices \(\mathbf{A}\) and \(\mathbf{B}\) that are added to the frozen pretrained weights, drastically reducing the number of trainable parameters while preserving the original model’s knowledge through a zero‑shot initialisation (the adapters initially contribute zero output). The module is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT, using the original linear layers as the frozen base. A scaling factor \(\alpha / r\) controls the magnitude of the adaptation. This paper provides a rigorous mathematical description of the LoRA adaptation, the architecture of the attention module, the initialisation strategy, and its integration into the compression framework via a registry‑based injection function.

---

## 1. Introduction

Fine‑tuning large pre‑trained Transformer models on downstream tasks is computationally expensive due to the massive number of parameters. Parameter‑efficient fine‑tuning methods address this by updating only a small fraction of the model’s parameters. **Low‑Rank Adaptation (LoRA)** (Hu et al., 2021) has emerged as a leading technique: it injects trainable low‑rank matrices into the linear layers of the model while keeping the original weights frozen. The adaptation is expressed as:

\[
\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x},
\]

where \(\mathbf{W}_0\) is the frozen pretrained weight, \(\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}\) and \(\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}\) are learned, and \(\alpha\) is a hyperparameter controlling the adaptation strength. The rank \(r\) is chosen much smaller than the dimensions, e.g., \(r = 8\).

In the **LoRA Attention** module, we apply this idea to all four linear projections within the multi‑head self‑attention block—query, key, value, and output. The original pretrained weights are wrapped inside a `LoRALinear` layer that freezes them and adds the trainable low‑rank adapters. The adapters are initialised so that \(\mathbf{B}\) is zero, ensuring that the model initially outputs exactly the same as the pretrained model (zero‑shot property). This enables fine‑tuning with minimal overfitting and strong performance on downstream tasks.

The module is designed to be injected into a DistilBERT layer by replacing the entire attention sub‑layer. It retains the standard multi‑head attention forward pass, with the only change being the linear projections. The result is a highly parameter‑efficient adaptation that can be trained with a fraction of the original parameters.

---

## 2. Method

### 2.1 Low‑Rank Adaptation for a Linear Layer

Given an original linear layer with weight matrix \(\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) and bias \(\mathbf{b}_0\) (optional), the LoRA‑adapted output is defined as:

\[
\mathbf{y} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B} \mathbf{A} \mathbf{x} + \mathbf{b}_0.
\]

The term \(\mathbf{B} \mathbf{A}\) is a low‑rank matrix of rank at most \(r\). The scaling factor \(\alpha / r\) ensures that the adaptation magnitude does not blow up with increasing rank and allows tuning of the adaptation strength independently of rank. The forward pass computes the original output first, then the adapter output using two small matrix multiplications, and sums them.

To achieve zero‑shot initialisation, \(\mathbf{A}\) is initialised with Kaiming uniform (for good gradient flow), while \(\mathbf{B}\) is initialised to **zero**. Thus, at the start of training, \(\mathbf{B}\mathbf{A} = \mathbf{0}\), and the layer behaves exactly as the pretrained model. This is crucial for stable fine‑tuning, as the model does not suffer from an initial perturbation.

Dropout can be applied to the input before the adapter (as in the original LoRA paper), though it is often set to zero. Here a dropout layer is added before \(\mathbf{A}\) if a non‑zero dropout is specified.

### 2.2 Application to Multi‑Head Self‑Attention

The standard multi‑head self‑attention layer projects the input \(\mathbf{H} \in \mathbb{R}^{B \times N \times D}\) to queries, keys, and values using learned weight matrices \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times (H d_h)}\), and the output projection \(\mathbf{W}_O \in \mathbb{R}^{(H d_h) \times D}\). In the LoRA Attention module, each of these matrices is replaced by a `LoRALinear` wrapper around the pretrained linear layer. Consequently, the projections become:

\[
\mathbf{Q} = \mathbf{W}_Q \mathbf{H}^\top + \frac{\alpha}{r} \mathbf{B}_Q \mathbf{A}_Q \mathbf{H}^\top, \quad \text{etc.}
\]

The multi‑head reshaping and attention computation remain unchanged: the queries, keys, and values are split into \(H\) heads, scaled dot‑product attention is performed, and the heads are concatenated before the output projection.

The attention scores are:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}},
\]

followed by optional masking, softmax, dropout, and aggregation:

\[
\mathbf{C} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}) \mathbf{V}.
\]

The output is then projected with the adapted output layer.

### 2.3 Parameter Efficiency

Each LoRA adapter adds only \(r \times (d_{\text{in}} + d_{\text{out}})\) parameters per linear layer. For a typical DistilBERT with \(D=768\) and \(H=12\), the full Q, K, V, and O matrices each contain \(768 \times 768 = 589,824\) parameters, totaling \(4 \times 589,824 = 2,359,296\) parameters per attention block. With LoRA rank \(r=8\), the added parameters are \(4 \times 8 \times (768 + 768) = 4 \times 8 \times 1536 = 49,152\)—a **48× reduction** compared to the full matrices. Moreover, only these adapters are trained; the original weights remain frozen, significantly reducing the optimizer state and memory footprint.

---

## 3. Implementation Details

### 3.1 LoRALinear Class

- **Constructor**:  
  - `original_layer`: an `nn.Linear` that is stored and frozen.  
  - `rank`: the decomposition rank \(r\).  
  - `alpha`: scaling factor (default 1.0).  
  - `dropout`: dropout probability on the adapter input (default 0.0).  
  - It creates two trainable parameters: `lora_A` (shape \(r \times d_{\text{in}}\)) and `lora_B` (shape \(d_{\text{out}} \times r\)).  
- **Initialisation**:  
  - `lora_A` is initialised with Kaiming uniform (to provide non‑zero gradients).  
  - `lora_B` is initialised to zeros.  
- **Forward**:  
  - Computes `orig_out = original_layer(x)`.  
  - Computes `lora_out = lora_B @ lora_A @ x` (with dropout on `x` before `lora_A`).  
  - Scales `lora_out` by `alpha / rank`.  
  - Returns `orig_out + lora_out`.  

### 3.2 LoRAAttention Class

- **Constructor**:  
  - `model_config`: provides `num_attention_heads`, `hidden_size`.  
  - `rank`: LoRA rank.  
  - `alpha`: LoRA alpha.  
  - `dropout`: attention dropout (for the attention weights, not the adapter).  
  - `pretrained_attn`: the original attention module containing `q_lin`, `k_lin`, `v_lin`, `out_lin`.  
  - Wraps each of the four linear layers in a `LoRALinear` instance.  
- **Forward**:  
  1. Projects hidden states through the adapted Q, K, V layers and reshapes to multi‑head.  
  2. Computes attention scores, applies mask (converts 2D mask to additive), softmax, dropout on attention weights.  
  3. Aggregates values.  
  4. Reshapes and applies adapted output projection.  
  5. Returns output and optionally attention weights.  

### 3.3 Injection Function

The injection function `inject_lora_attn` reads parameters from `comp_config` (defaults: `lora_rank=8`, `lora_alpha=16.0`, `dropout=0.1`) and creates a new `LoRAAttention` instance, passing the original attention module as `pretrained_attn`. It then replaces `layer.attention` with the new module. This ensures the original model is seamlessly extended with LoRA adapters without any change to the rest of the model.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"lora_attn"` in the compression configuration. Relevant parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 8 | Rank \(r\) of the low‑rank adapters. |
| `lora_alpha` | 16.0 | Scaling factor \(\alpha\) for the adaptation. |
| `dropout` | 0.1 | Dropout rate applied to the attention weights. |

Example configuration:
```python
comp_config.attention_type = "lora_attn"
comp_config.lora_rank = 16
comp_config.lora_alpha = 32.0
```

Manual instantiation:
```python
attn = LoRAAttention(model_config, rank=16, alpha=32.0, dropout=0.1, pretrained_attn=old_attn)
layer.attention = attn
```

The original weights are automatically frozen; only the LoRA parameters are trained.

---

## 5. Discussion

LoRA Attention offers an excellent trade‑off between parameter efficiency and model quality. By freezing the pretrained attention weights and learning only small, low‑rank corrections, it preserves the rich linguistic knowledge of the base model while adapting to the target task. The zero‑shot initialisation ensures no initial performance degradation, and the scaling factor \(\alpha\) provides a simple knob to control the adaptation strength.

The method is particularly effective in the attention module because attention projections are often heavily over‑parameterised for downstream tasks; low‑rank updates can capture task‑specific shifts without requiring full re‑training. In a multi‑layer model, applying LoRA to all attention layers yields a significant reduction in total trainable parameters—often to less than 1% of the full model—while still achieving competitive results.

One limitation is that the rank \(r\) and \(\alpha\) must be tuned. Too small a rank may limit capacity, while too large a rank can increase memory usage and risk overfitting. The optimal values depend on the task complexity and dataset size. Additionally, LoRA does not modify the attention pattern itself; the computational cost of attention remains the same as the base model. For extreme efficiency, LoRA could be combined with other linear‑time attention mechanisms, though that is beyond the scope of this module.

---

## 6. Conclusion

We have presented LoRA Attention, a parameter‑efficient adaptation of the multi‑head self‑attention mechanism using low‑rank adapters. The module wraps the original pretrained linear layers, freezes them, and adds trainable low‑rank matrices that are initialised to zero, guaranteeing a zero‑shot start. Integrated into the DistilBERT compression framework, it provides a straightforward way to fine‑tune attention layers with a fraction of the parameters, making it a valuable tool for resource‑constrained applications. Future extensions could combine LoRA with efficient attention patterns to further reduce the overall computational cost while maintaining the benefits of parameter‑efficient adaptation.