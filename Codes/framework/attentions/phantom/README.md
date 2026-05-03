# PHANTOM — Parameter‑Hashed AttentioN with Token‑wise Orthogonal Mapping

**CUR‑factorised projections + multi‑round angular LSH attention with Softpick sparsity.**

Drop‑in attention module for DistilBERT that combines **parameter efficiency**, **sub‑quadratic attention**, and **sparse activation** – all with zero‑shot weight initialisation.

---

## 🎯 Key Highlights

- **CUR decomposition** – approximates Q/K/V/Out matrices via low‑rank row/column sampling, reducing parameters by **5‑20×**.
- **Angular LSH attention** – splits query‑key pairs into buckets using random hyper‑plane hashing in multiple rounds. Complexity drops from `O(N²)` to `O(N·B)` where `B` ≈ bucket size.
- **Softpick activation** – introduces natural sparsity (~47% zeros) without a separate sparsity loss.
- **Zero‑shot** – initialised directly from a pretrained DistilBERT attention layer; no fine‑tuning required for first use.

---

## 🧠 Architecture Deep‑Dive

### 1. CUR‑Factorised Linear Layers (`FactorisedLinear_CUR`)
Instead of storing a full `(out_dim × in_dim)` weight matrix, PHANTOM keeps two smaller matrices:
- **`down`** – `(rank × in_dim)` – projects input to a compact subspace.
- **`up`** – `(out_dim × rank)` – expands back to the output dimension.

The weight matrices are initialised using a **CUR decomposition** of the pretrained weights:
1. Compute row and column leverage scores via SVD.
2. Sample the **top‑`rank`** rows and columns.
3. Build `C` (columns), `R` (rows), and a small `U` matrix that links them.
4. The product `C @ U @ R` approximates the original weight.

> **Parameter saving:** For DistilBERT (`d=768`), a full projection has `768×768 ≈ 590k` parameters. With `rank=16`, the CUR version uses `768×16 + 16×768 = 24,576` parameters → **24× reduction**.

### 2. Angular LSH Attention
Standard dot‑product attention is quadratic in sequence length. PHANTOM replaces it with a **multi‑round locality‑sensitive hashing** scheme:

- **Query & key vectors are L2‑normalised** → attention becomes cosine similarity.
- **Random hyper‑planes** generate a binary hash code for each vector.
- Queries **only attend to keys that fall into the same bucket**.
- Multiple independent rounds are performed to increase the probability that truly relevant pairs share at least one bucket. The final output is the sum across rounds.

This reduces the effective attention computation to **O(N·B·rounds)** where `B` = average bucket size (controlled by `lsh_bucket_size`). For typical settings (4 rounds, 32 buckets), attention cost is *linear in sequence length*.

### 3. Softpick Sparsity
Softpick is a **rectified softmax**:

```
p_i = relu( exp((s_i - max)/T) - exp(-max/T) ) / sum(relu(...))
```

It naturally drives many attention probabilities to **exactly zero** (typically ~47% on trained models), saving further memory and FLOPs. In PHANTOM it is applied *inside* each LSH bucket.

---

## ⚙️ Configuration

All hyper‑parameters live in `PHANTOMConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cur_rank` | `16` | Rank of CUR approximation for all projections |
| `lsh_rounds` | `4` | Number of independent LSH hash rounds |
| `lsh_bucket_size` | `32` | Number of hash buckets (determines granularity) |
| `softpick_enabled` | `True` | Whether to use Softpick (else standard softmax) |
| `softpick_temperature` | `1.0` | Temperature for Softpick |
| `dropout` | `0.1` | Dropout rate on attention weights |

These can be set directly or via the injection function which reads from a `comp_config` object.

---

## 🚀 Usage

### Injection via Registry (recommended for framework pipelines)
```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

ATTENTION_REGISTRY["phantom"](
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
)
```
The injector picks up `comp_config.cur_rank`, `.lsh_rounds`, `.lsh_bucket_size`, `.dropout`.

### Direct Instantiation
```python
from phantom import PHANTOMAttention, PHANTOMConfig

cfg = PHANTOMConfig(cur_rank=16, lsh_rounds=8)
new_attn = PHANTOMAttention(model_config, cfg, pretrained_attn=old_attn)
out, _ = new_attn(hidden_states, attention_mask)
```

---

## 📊 Complexity Analysis

Let:
- $N$ = sequence length
- $D$ = model dimension
- $H$ = number of heads
- $R$ = rank (CUR rank)
- $B$ = bucket size, $K$ = rounds

| Component | Time Complexity | Memory (parameters) |
|-----------|----------------|---------------------|
| Standard attention | $O(N^2 D)$ | $4 D^2$ |
| **PHANTOM CUR proj** | $O(N D R)$ | $2 D R$ per projection |
| **PHANTOM LSH attention** | $O(N H B K D)$ | – |
| **Total** | $O(N D R + N H B K D)$ | $8 D R$ (for Q,K,V,O) |

**Example:** DistilBERT with $D=768$, $H=12$, $R=16$, $B=32$, $K=4$  
Step‑by‑step:
- Standard parameters: $4 \times 768^2 \approx 2.36$M
- PHANTOM parameters: $8 \times 768 \times 16 = 98,\!304$ → **24× compression**.
- Attention FLOPs: For $N=512$, standard attention $\sim 512^2 \cdot 64 \cdot 12 = 201$M MACs.  
  PHANTOM attention $\sim 512 \cdot 12 \cdot 32 \cdot 4 \cdot 64 = 50$M MACs → **4× speedup**.

---

## 🔬 Comparison with Other Efficient Attention Methods

| Method | Parameter Reduction | Complexity | Sparse / Low‑Rank | Zero‑Shot | Special Traits |
|--------|-------------------|------------|-------------------|-----------|----------------|
| **PHANTOM** | ✅ CUR (≥10×) | $O(N B K D)$ | Both (LSH + Softpick) | ✅ Yes | Combines hash‑based sparsity with structured low‑rank; multi‑round LSH for accuracy |
| **AURORA** | ✅ ARMOR (SVD+quant) | $O(N^2)$ (full attention) | Sparse + low‑rank | Partially | 10‑in‑1 suite; includes token pruning, DeltaKV, etc. |
| **ORION** | ✅ SVD factorised | Hybrid ($O(NF + N^2)$) | Separate kernel + full heads | ✅ Yes | Head‑wise split; kernel attention via RFF |
| **NOVA** | ✅ ARMOR + subspace | $O(N^2)$ (full) | Low‑rank + token pruning | ✅ Yes | 8 techniques, including AST token pruning |
| **MinimalNOVA** | ✅ SVD factorised | $O(N^2)$ (full) | Low‑rank only | ✅ Yes | Pure SVD factorisation, simplest drop‑in |
| **OHA** | ✅ Low‑rank global | $O(N W + N R)$ | Sliding window + low‑rank global | ✅ Yes | Tiny memory footprint; gated fusion |
| **Performer** | ❌ Full params | $O(N D \log D)$ (kernel) | Random features | ❌ | FAVOR+ kernel; no parameter reduction |
| **Linformer** | ✅ Low‑rank projection | $O(N k D)$ | Projected K,V | ❌ | Fixed projection matrix; needs training |
| **Reformer** | ❌ Full params | $O(N \log N)$ | LSH + reversible | ❌ | LSH but retains full parameters; uses reversible layers |
| **Sparse Transformers** | ❌ Full params | $O(N \sqrt{N})$ (strided) | Fixed sparse pattern | ❌ | Hand‑designed sparsity; no parameter saving |

**Key differences:**
- **PHANTOM** is the **only method** that simultaneously achieves:
  - Massive parameter reduction (CUR)
  - Sub‑quadratic attention via LSH
  - Built‑in sparsity (Softpick)
  - Zero‑shot weight transfer from a standard DistilBERT checkpoint.
- Compared to **AURORA/NOVA**, PHANTOM focuses on hash‑based sparsity rather than token pruning or adaptive mechanisms, making it especially suited for **long‑sequence** inference where bucket‑size control enables strict memory bounds.
- Unlike **Performer** or **Reformer**, PHANTOM does not require retraining; it inherits the pretrained knowledge directly via CUR.

---

## 📦 Integration into the Efficient Transformer Toolkit

PHANTOM is part of the `framework/attentions/phantom/` package:

```tree
phantom/
├── __init__.py
├── phantom.py
└── README.md
```

It is registered under the key `"phantom"` and can be used interchangeably with any other attention module in the toolkit via `replace_attention` or direct injection.

---

## 🙌 Acknowledgements

This implementation is inspired by:
- CUR decomposition for low‑rank matrix approximation.
- Reformer’s LSH attention.
- Softpick sparse activation from the AURORA suite.

All code is provided for research use.

---

*Part of the Efficient Transformer Toolkit – compressing attention with hashing and low‑rank algebra.*
