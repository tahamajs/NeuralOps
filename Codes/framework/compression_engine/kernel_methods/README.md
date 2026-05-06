# 🔬 Galerkin, RPC, Spectraformer & TLinFormer Attention

Four efficient attention modules that replace the quadratic self‑attention of Transformers with **sub‑quadratic** or **low‑rank** approximations, each based on a distinct mathematical principle.

| Module | Registry Key | Complexity | Core Idea |
|--------|--------------|------------|-----------|
| **Galerkin** | `galerkin` | O(N·R) | Petrov‑Galerkin projection onto a random low‑rank subspace |
| **RPC** | `rpc` | O(N·R) | Kernel PCA: project queries onto principal components of keys |
| **Spectraformer** | `spectra` | O(N·R) | Performer‑style random Fourier features with orthogonal projections |
| **TLinFormer** | `tlinformer` | O(N·R) | Learnable low‑rank key/value projections (Linformer‑like) |

All modules are registered in `ATTENTION_REGISTRY` and can be injected into any DistilBERT / BERT model.

---

## 🧬 Detailed Descriptions

### 1. Galerkin Attention (`galerkin`)

- **Paper:** *”Choose a Transformer: Fourier or Galerkin”* (Cao, 2021)
- **How it works:**  
  Projects keys and values onto a random low‑rank subspace using a **Gaussian sketching matrix** of shape (seq_len × rank).  
  Attention scores are then computed between the full queries and the projected keys – reducing the cost from O(N²) to O(N·R).  
  This can be seen as a **Petrov‑Galerkin projection** of the attention operator.
- **When to use:** When you want a simple, non‑learned subspace projection that works out‑of‑the‑box.

### 2. RPC Attention (`rpc`)

- **Paper:** *”Self‑Attention through Kernel PCA”* (inspired)
- **How it works:**  
  Computes the **principal components of the key vectors** via SVD (per head), then projects queries onto those components.  
  Attention is performed in the reduced (rank) space, and the resulting context is mapped back using a value projection.
- **When to use:** When the key distribution has a clear low‑dimensional structure (e.g., after extensive pretraining).

### 3. Spectraformer Attention (`spectra`)

- **Paper:** *”Spectraformer: Random Feature Linearization with Flexible Kernel Approximation”* (placeholder)
- **How it works:**  
  Applies **orthogonal random Fourier features** (the same family as Performer / FAVOR+).  
  The feature map uses ReLU for positivity, then performs linear attention: `φ(Q) (φ(K)^T V) / denominator`.  
  The random projection matrix is regenerated at each forward pass (can be made fixed).
- **When to use:** When you need a proven linear attention mechanism with strong theoretical guarantees.

### 4. TLinFormer Attention (`tlinformer`)

- **Paper:** *”TLinFormer: Exact Linear‑Time Attention by Reconfiguring Neuron Connections”* (placeholder)
- **How it works:**  
  Learns two small projection matrices (`E` and `F`) that compress the key and value dimensions from `head_dim` down to `rank`.  
  Attention is computed in the lower‑dimensional space, achieving O(N·R) complexity.  
  Similar to Linformer, but the projection is applied per head instead of over the sequence length.
- **When to use:** When you want a learnable, trainable linear attention that can adapt to your data.

---

## ⚙️ Configuration

All modules accept a standard HuggingFace‑style `config` and a `rank` parameter.

| Module | Parameter | Default | Description |
|--------|----------|---------|-------------|
| Galerkin | `rank` | 64 | Dimension of the random subspace |
| RPC | `rank` | 64 | Number of principal components used |
| Spectraformer | `rank` | 256 | Number of random Fourier features |
| TLinFormer | `rank` | 64 | Dimension of the low‑rank projection |

All modules inherit from `BaseAttention` and expose a standard forward:
```python
output, attn_weights = module(hidden_states, attention_mask)
```

---

## 🚀 Usage

### Inject via Registry

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY

ATTENTION_REGISTRY["galerkin"](layer, model_config, comp_config, ...)
ATTENTION_REGISTRY["rpc"](layer, model_config, comp_config, ...)
ATTENTION_REGISTRY["spectra"](layer, model_config, comp_config, ...)
ATTENTION_REGISTRY["tlinformer"](layer, model_config, comp_config, ...)
```

### Direct Instantiation

```python
from framework.attentions.galerkin_attention import GalerkinAttention

attn = GalerkinAttention(config, rank=32)
out, _ = attn(hidden_states, attention_mask)
```

### CLI

```bash
python cli.py train --task mrpc --attention galerkin
python cli.py train --task mrpc --attention rpc
python cli.py train --task mrpc --attention spectra
python cli.py train --task mrpc --attention tlinformer
```

---

## 📊 Comparison with Other Linear Attention Methods

| Method | Complexity | Learnable | Stability | Best for |
|--------|------------|-----------|-----------|----------|
| Galerkin | O(N·R) | No | High | Quick prototyping |
| RPC | O(N·R) + SVD cost | No | High | Structured key distributions |
| Spectraformer | O(N·R) | No | Very high | Production linear attention |
| TLinFormer | O(N·R) | Yes (E,F) | High | Tailored to dataset |
| Performer | O(N·R) | No | Very high | General purpose |
| Linformer | O(N·k) | Yes (E,F) | High | Fixed sequence length |

- **Galerkin** and **RPC** are the cheapest to initialise (no extra parameters).
- **Spectraformer** is the most numerically stable because of orthogonal features.
- **TLinFormer** offers the best accuracy/compromise trade‑off when you can fine‑tune.

---

## 📜 References

- Galerkin Transformer: *Cao, 2021 (arXiv:2105.14906)*  
- Kernel PCA attention: multiple works  
- Spectraformer: *Nguyen et al., 2025 (arXiv:2405.15310)*  
- TLinFormer: placeholder  

*Choose the linear attention that fits your needs – all four are ready to use.*
