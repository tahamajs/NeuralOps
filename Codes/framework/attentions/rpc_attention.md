# RPC Attention: Random Projection Augmented Self‑Attention

**Abstract**

We present **RPC Attention**, a simple baseline attention module that pre‑processes the query and key vectors of each attention head with a fixed, randomly initialised linear transformation before computing the scaled dot‑product attention. The projection matrices are stored as non‑trainable buffers, one per head, and serve as a fixed random feature map. This design can be seen as a lightweight form of feature decorrelation or as a static approximation to more sophisticated random feature attention mechanisms. The module retains the exact quadratic complexity of standard self‑attention but provides a drop‑in replacement for the self‑attention sub‑layer in DistilBERT within the compression framework. This paper describes the mathematical formulation, implementation, and intended role as a baseline for efficient attention experiments.

---

## 1. Introduction

Standard multi‑head self‑attention computes pairwise similarities between queries and keys in the original feature space. Various efficient attention methods use random projections to approximate the softmax kernel (e.g., Performer, RALA). **RPC Attention** (Random Projection Attention) takes a minimalistic step in this direction: it applies a **fixed, random linear transformation** to the query and key tensors of each head before the dot product, without any subsequent non‑linearity or kernel approximation. The transformation is a square matrix (same dimension as the head), initialised from a standard normal distribution and never updated during training. This can be viewed as a form of static feature mixing that decorrelates the input dimensions, potentially improving gradient flow and acting as a regulariser.

The module is intentionally simple, serving as a controlled baseline for more complex random‑feature‑based attention variants. It is integrated into the compression framework via the `ATTENTION_REGISTRY` as a drop‑in replacement for DistilBERT’s self‑attention. This paper provides the mathematical details, implementation notes, and discusses the role of RPC Attention in the broader landscape of efficient Transformers.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first applies standard linear projections to obtain queries, keys, and values for all heads:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times (H d_h)}\), where \(H\) is the number of heads and \(d_h = D/H\). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\) and transposed to \((B, H, L, d_h)\).

### 2.2 Per‑Head Fixed Random Projection

For each head \(h\), a random matrix \(\mathbf{R}_h \in \mathbb{R}^{d_h \times d_h}\) is created once and stored as a non‑trainable buffer. The entries of \(\mathbf{R}_h\) are drawn from a standard normal distribution (using `torch.randn`). This matrix is then used to project the queries and keys of head \(h\):

\[
\tilde{\mathbf{Q}}_h = \mathbf{Q}_h \mathbf{R}_h, \qquad
\tilde{\mathbf{K}}_h = \mathbf{K}_h \mathbf{R}_h.
\]

The values \(\mathbf{V}_h\) remain unchanged. Since the projection is applied within the \(d_h\)-dimensional head space, it does not reduce dimensionality; it merely rotates (and potentially scales) the feature vectors. The projection is fixed after initialisation and does not receive gradient updates.

### 2.3 Scaled Dot‑Product Attention

The attention scores are computed using the projected queries and keys:

\[
\mathbf{S}_h = \frac{\tilde{\mathbf{Q}}_h \tilde{\mathbf{K}}_h^\top}{\sqrt{d_h}}.
\]

Softmax is applied to obtain attention weights \(\mathbf{A}_h = \operatorname{softmax}(\mathbf{S}_h)\), and the output context is:

\[
\mathbf{C}_h = \mathbf{A}_h \mathbf{V}_h.
\]

The per‑head outputs are concatenated, reshaped to \((B, L, D)\), and passed through the output projection \(\mathbf{W}_O\):

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

### 2.4 Rationale

If the random matrices were orthonormal, the dot product would be preserved: \((\mathbf{Q}\mathbf{R})(\mathbf{K}\mathbf{R})^\top = \mathbf{Q}\mathbf{K}^\top\). Since they are general random matrices, the dot product is altered. This alteration can be seen as a fixed, data‑independent feature mixing that may help break pathological initialisations or create a more diverse set of attention patterns across heads. It also serves as a minimal augmentation to test whether random projections provide any benefit in a standard attention setting, analogous to the use of fixed random encoders in some neural network architectures.

---

## 3. Implementation Details

- **Class `RPCAttention`** inherits from `nn.Module` and is registered under `"rpc"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size`.
  - `rank`: ignored (for interface compatibility).
  - `pretrained_attn`: accepted but not used; does not perform weight transfer.
  - Creates linear layers `W_q`, `W_k`, `W_v`, `out_proj` (with biases).
  - Initialises a buffer `proj` of shape `(num_heads, head_dim, head_dim)` from a standard normal distribution. This buffer is moved to the appropriate device automatically.
- **Forward**:
  1. Project hidden states to Q, K, V and reshape to `(B, H, L, d_h)`.
  2. Retrieve `proj` and apply to Q and K using `einsum('bhld,hde->bhle', ...)`.
  3. Compute attention scores with scaling, apply softmax.
  4. Multiply by V, transpose, reshape, output projection.
  5. Return `(output, None)`.
- **Mask handling**: The forward method does **not** accept or use an `attention_mask`. Padding tokens will be processed like regular tokens, potentially causing issues. This omission should be addressed in a production version.
- **Bug note**: The constructor incorrectly calls `super().__init__(config, rank)`, but the class is defined as inheriting from `nn.Module`, which accepts no arguments. This likely causes an error at runtime; a correct version would call `super().__init__()`. Additionally, the forward method has an unused `pretrained_attn` argument that shadows the constructor parameter.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"rpc"` in the compression configuration. No additional hyperparameters are required. Example injection (assuming the bug is fixed):

```python
comp_config.attention_type = "rpc"
```

Manual instantiation (fixed):
```python
attn = RPCAttention(config=model_config)  # rank will default to 64 internally
layer.attention = attn
```

Since the module does not initialise from pretrained weights, the linear layers are randomly initialised, and the model must be fine‑tuned.

---

## 5. Discussion

RPC Attention is a minimal modification of the standard attention mechanism. By applying a fixed random transformation to queries and keys, it introduces a static, per‑head mixing of features. This can be interpreted as a form of **attention ensembling** where each head operates on a different random rotation of the feature space, potentially increasing the diversity of attention patterns without any learnable parameters. In the context of the compression framework, it serves as a **baseline** for evaluating whether learned random features (as in Performer or RALA) outperform a simple fixed random projection.

However, the module’s utility is limited by several factors:
- The random matrices are not orthogonal, so the dot product is distorted, which may harm performance.
- The projection is applied to the same dimension, so there is no reduction in computational complexity; it remains quadratic.
- The lack of mask support and the `super()` bug render it currently unsuitable for real use without fixes.
- No pretrained weight transfer means it cannot be used zero‑shot.

If the projection matrices were made learnable, the module would resemble a simple low‑rank adaptation in the feature space, but they are fixed. Thus, RPC Attention is best viewed as a **pedagogical** or **debugging** tool within the framework, rather than a practical efficient attention method.

---

## 6. Conclusion

We have described RPC Attention, a simple baseline attention module that applies fixed random projections to queries and keys before the dot product. While it retains the full quadratic complexity of standard attention, it introduces a minimal form of feature mixing that can serve as a reference for more advanced random‑feature‑based attention mechanisms. The module is integrated into the DistilBERT compression framework, though it requires minor fixes to be fully functional. Future work could extend this idea by making the projections learnable or orthogonal, bridging the gap toward true random feature attention.