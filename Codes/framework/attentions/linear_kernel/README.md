# Linear & Low‑Rank Attention Suite

A collection of **seven** attention modules that replace quadratic self‑attention with **linear** or **low‑rank** approximations.  
All modules are drop‑in compatible with DistilBERT / BERT models and registered in `ATTENTION_REGISTRY` (except RALA).

---

## 📦 Modules

| Module | Registry key | Complexity | Core idea |
|--------|--------------|------------|-----------|
| **ChelaSlay** | `chela_slay` | `O(N·F)` | Spherical normalisation + Yat kernel (random Fourier features) |
| **Nyströmformer** | `nystrom` | `O(N·R)` | Landmark tokens + Nyström approximation of attention matrix |
| **Performer** | `performer` | `O(N·R)` | FAVOR+ with orthogonal random features |
| **RALA** | – (standalone) | `O(N·R)` | Rank‑augmented linear attention (elu + low‑rank correction) |
| **ReGLA** | `regla` | `O(N·R)` | Gated linear attention + depthwise convolution |
| **RPC** | `rpc` | `O(N²)` (projection only) | Kernel PCA via fixed random projection |
| **ZeroS** | `zeros` | `O(N·R)` | Zero‑mean linear attention (centered keys) |

*`N` = sequence length, `F` = number of Fourier features, `R` = rank / landmark count.*

---

## 🚀 Quick Start

### Using the Registry

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

# Replace a layer with Nyströmformer
ATTENTION_REGISTRY["nystrom"](layer, model_config, comp_config, ...)
```

### Direct Instantiation (e.g., Performer)

```python
from framework.attentions.performer_attention import PerformerAttention

attn = PerformerAttention(config, rank=256, redraw_projection=False)
output, _ = attn(hidden_states, attention_mask)
```

---

## 🔬 Detailed Module Descriptions

### 1. ChelaSlay – Spherical Yat‑kernel Attention
- **Paper:** ChelaSlay (spherical linearized attention)
- **How it works:**  
  Projects queries and keys onto the unit sphere. Then applies a **Yat kernel**: random Fourier features (`cos`, `sin`) are used to approximate a kernel function.  
  Linear attention is computed as:  
  `output = φ(Q) · (φ(K)^T V) / (φ(Q) · sum(φ(K)))`  
  This avoids the `N × N` attention matrix.
- **Parameters:**  
  `feature_dim` – number of random Fourier features (default 256).  
  `random_proj` matrix is fixed after initialisation.
- **When to use:** When you need a simple, deterministic linear attention without learned kernels.

### 2. Nyströmformer – Landmark‑based Nyström Approximation
- **Paper:** [Nyströmformer: A Nyström‑based Algorithm for Approximating Self‑Attention](https://arxiv.org/abs/2102.03902)
- **How it works:**  
  Selects `R` landmark tokens (by segment averaging). Computes three matrices:  
  - `F1 = Q @ K_land^T`  
  - `F2 = Q_land @ K_land^T`  
  - `F3 = Q_land @ K^T`  
  Then approximates the full attention matrix as `F1 @ pinv(F2) @ F3`.  
  Uses iterative Moore‑Penrose pseudo‑inverse for stability.
- **Parameters:**  
  `rank` – number of landmarks (default 64).  
  `pinv_iterations` – iterations for pseudo‑inverse refinement (default 6).  
  `residual` – adds V to the output (helpful for fine‑tuning).
- **When to use:** When you want a mathematically grounded low‑rank approximation that works well with pre‑trained models.

### 3. Performer – FAVOR+ (Fast Attention Via Orthogonal Random Features)
- **Paper:** [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
- **How it works:**  
  Approximates the softmax kernel using **orthogonal random features**.  
  `φ(x) = ReLU( x @ proj / scale )` – positive, orthogonal projections.  
  Linear attention is performed as `φ(Q) · (φ(K)^T V) / (φ(Q) · sum(φ(K)))`.  
  The projection matrix can be redrawn at each forward pass (`redraw_projection=True`).
- **Parameters:**  
  `rank` – number of random features (default 256).  
  `redraw_projection` – whether to resample the orthogonal matrix every forward pass (default True).
- **When to use:** When you need a proven, stable linear attention with strong theoretical guarantees.

### 4. RALA – Rank‑Augmented Linear Attention
- **Paper:** [Rank‑Augmented Linear Attention](https://arxiv.org/abs/2311.17374)
- **How it works:**  
  Standard linear attention using `elu(x)+1` kernel.  
  Adds a **learned low‑rank correction** term: `U @ V^T` (one per head) to capture higher‑order interactions that linear attention alone may miss.  
  A learnable scalar `alpha` blends the correction.
- **Parameters:**  
  `rala_rank` (from config) – rank of the augmentation matrices (default 8).  
  Trains `U`, `V`, and `alpha`.
- **When to use:** When you want to improve the expressiveness of linear attention with very few extra parameters (suitable for fine‑tuning).

### 5. ReGLA – Refined Gated Linear Attention
- **Paper:** ReGLA (NAACL 2025)
- **How it works:**  
  Multiplies keys by a **sigmoid gate** before linear attention.  
  Linear attention is properly normalised: `Q (K^T V) / (Q sum(K))`.  
  Adds a **depthwise convolution** (local enhancement) on the original input as a residual.
- **Parameters:**  
  `kernel_size` – convolution kernel size (default 3).  
  `gate_proj` – learned gating projection.
- **When to use:** When you want gating to control the flow of information and a local positional boost.

### 6. RPC – Random Projection Kernel PCA Attention
- **How it works:**  
  Projects queries and keys using a **fixed random Gaussian matrix** (one per head).  
  Standard softmax attention is then performed – this is **not linear**; it’s a feature transformation that can improve robustness.  
  The projection matrix is frozen after initialisation.
- **Complexity:** Still `O(N²)` – use only if you need a different feature space.
- **When to use:** As a lightweight alternative to learned projections; can be combined with other linear methods.

### 7. ZeroS – Zero‑Mean Linear Attention
- **How it works:**  
  Centers the keys by subtracting their mean along the sequence dimension → attention has a **zero‑sum property**.  
  Applies ReLU activation on queries and centered keys, then performs linear attention with proper normalisation.
- **When to use:** When you want a simple, normalisation‑friendly linear attention with reduced numerical variance.

---

## ⚙️ Shared Configuration

All modules (except RALA) accept a standard HuggingFace `config` with:
- `hidden_size`
- `num_attention_heads`
- `attention_probs_dropout_prob` (where applicable)

Additional parameters can be passed directly to the constructor or through `comp_config`.

For RALA, extra settings like `rala_rank` are read from `config`.

---

## 📊 Comparison Table

| Module | Complexity | Memory (KV cache) | Requires Training? | Learned Parameters | Numeric Stability |
|--------|------------|-------------------|--------------------|--------------------|--------------------|
| ChelaSlay | O(N·F) | O(N·F) | No (fixed proj) | None | Good |
| Nyström | O(N·R) | O(N·R) | No | None | Requires pinv iterations |
| Performer | O(N·R) | O(N·R) | No (optional redraw) | None | Very good |
| RALA | O(N·R) | O(N·R) | Yes (U, V, α) | Small (U,V,α) | Good |
| ReGLA | O(N·R) | O(N·R) | Yes (gate, conv) | Small | Good |
| RPC | O(N²) | O(N²) | No (fixed proj) | None | Standard |
| ZeroS | O(N·R) | O(N·R) | No | None | Good |

*All linear methods except RPC avoid the N² memory bottleneck during attention.*

---

## 🛠️ Integration into Your Pipeline

1. **Direct replacement** – instantiate the attention class and assign it to the transformer layer:
   ```python
   from performer_attention import PerformerAttention
   layer.attention.self = PerformerAttention(config, rank=256)
   ```
2. **Via Registry** – use the injection function registered under the module’s key.
3. **With HuggingFace** – you can replace individual attention layers in a `DistilBertModel` by iterating over `model.distilbert.transformer.layer` and swapping the `attention` sub‑module.

All modules produce the standard output signature: `(context, attention_weights)` where `attention_weights` may be `None` for linear methods.

---

## 📜 References

- ChelaSlay: Spherical linearised attention with Yat kernel
- Nyströmformer: *Nyströmformer – A Nyström‑based Algorithm for Approximating Self‑Attention* (2021)
- Performer: *Rethinking Attention with Performers* (FAVOR+) (2020)
- RALA: *Rank‑Augmented Linear Attention* (ICLR 2024)
- ReGLA: *Refined Gated Linear Attention* (NAACL 2025)
- ZeroS: zero‑mean linear attention (several works)

---

*Linear attention for the long run – select your favourite kernel!*
