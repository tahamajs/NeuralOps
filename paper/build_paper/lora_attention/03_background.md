# 3. Background

## 3.1 Attention Projections

For input $X \in \mathbb{R}^{B \times L \times d}$:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\tag{3.1}
$$

with $W_Q \in \mathbb{R}^{d \times H d_h}$ and $W_K, W_V \in \mathbb{R}^{d \times H_{kv} d_h}$.

The output projection:

$$
\text{Out} = \text{Attn}(Q, K, V) \cdot W_O, \quad W_O \in \mathbb{R}^{H d_h \times d}
\tag{3.2}
$$

Total attention parameters: $4 d^2$ for standard MHA.

## 3.2 LoRA

LoRA (Hu et al., 2021) parameterizes weight updates as:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.3}
$$

with $B \in \mathbb{R}^{d_\text{out} \times r_L}$, $A \in \mathbb{R}^{r_L \times d_\text{in}}$.

**Initialization.** $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$. So 
$\Delta W = 0$ at step 0.

**Forward.** For input $x$:

$$
y = W_\text{frozen} x + \frac{\alpha}{r_L} B A x
\tag{3.4}
$$

**Mergeability.** After training, we can compute:

$$
W_\text{merged} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.5}
$$

and discard $A, B$, giving zero inference overhead.

## 3.3 Trainable Parameter Count

For each attention projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

$$
\text{Trainable}_\text{LoRA} = r_L (d_\text{out} + d_\text{in})
\tag{3.6}
$$

For $d = 768$, $r_L = 16$:

| Projection | Frozen | Trainable (LoRA) |
|---|---|---|
| $W_Q$ | 589,824 | $16 \cdot 1536 = 24,576$ |
| $W_K$ | 589,824 | 24,576 |
| $W_V$ | 589,824 | 24,576 |
| $W_O$ | 589,824 | 24,576 |
| **Total** | 2,359,296 | **98,304** |

**Reduction:** ~24× fewer trainable parameters.

## 3.4 Memory Savings

Full fine-tuning requires:
- Gradients: `params × 2 bytes` (FP16)
- AdamW state: `params × 8 bytes` (two FP32 moments)
- Total: `params × 10 bytes`

For DistilBERT attention (14.2M params):
- Full FT: ~142 MB of optimizer state per layer
- LoRA FT: ~1 MB

**Total reduction: ~100×**

## 3.5 Multi-Architecture Weight Formats

### 3.5.1 GPT-2 Conv1D

```python
c_attn.weight.shape = (d, 3d)  # (in, out) — must transpose
```

### 3.5.2 RoBERTa Nested

```python
layer.attention.self.query.weight  # Linear (out, in)
layer.attention.output.dense.weight
```

### 3.5.3 Qwen GQA

```python
q_proj.weight.shape = (H·d_h, d)
k_proj.weight.shape = (H_kv·d_h, d)  # smaller
```

### 3.5.4 DistilBERT Flat

```python
q_lin.weight.shape = (d, d)  # standard Linear
```

## 3.6 Nested vs Flat Encoders

| Model | Structure | Output projection |
|---|---|---|
| DistilBERT | Flat | `out_lin` inside attention |
| RoBERTa | Nested | `output.dense` outside attention |
| GPT-2 | Flat | `c_proj` inside attention |

**Critical.** For nested encoders, the LoRA module must **not** apply 
`out_proj` — the outer layer handles it.

## 3.7 Grouped-Query Attention (GQA)

$$
K, V \in \mathbb{R}^{B \times L \times H_{kv} \times d_h}
\tag{3.7}
$$

Expanded via `repeat_kv` before attention.

## 3.8 Rotary Position Embeddings (RoPE)

$$
\tilde{q}_i = q_i \cos(\theta_i) + \text{rot}(q_i) \sin(\theta_i)
\tag{3.8}
$$

Applied to $Q, K$ after projection.

---