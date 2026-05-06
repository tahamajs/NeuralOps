# 🧊 Quantisation & Joint Low-Rank Compression

This sub‑package combines **low‑rank matrix factorisation** with **quantisation** to achieve extreme model compression while preserving accuracy.  
All compressors are registered in `COMPRESSOR_REGISTRY` and can be used via the CLI, in training scripts, or as standalone layers.

## 📦 Available Compressors

| Registry Key | Compressor Class      | Core Technique |
|--------------|-----------------------|----------------|
| `uniql`      | `UniQLCompressor`      | SVD + uniform quantisation of factors |
| `mlorq`      | `MLoRQCompressor`      | Multi‑Layer optimal rank & bit assignment under a memory budget |
| `kmeans`     | `KMeansCompressor`     | Weight clustering via k‑means centroids |
| `faq`        | `FAQQuantizer`         | Future‑Aware Quantisation (activation‑driven) |
| `coopq`      | `CoopQQuantizer`       | Shapley‑inspired per‑layer bit allocation |
| `astro`      | `AstroQuantizer`       | Simple uniform quantisation |

Plus helpful standalone utilities:

- `HuffmanCoder` – lossless entropy coding for weights.
- `nli_softmax` – a quantisation‑friendly softmax approximation.

---

## 🚀 Quick Start

### Compress via Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Uniform quantisation + low‑rank
CompressorClass = COMPRESSOR_REGISTRY.get("uniql")
compressor = CompressorClass(model, rank=32, bits=4)
compressed_model, stats = compressor.compress()
print(stats)
```

### Compress from the CLI

```bash
python cli.py compress --method uniql --target-rank 32 --custom-param 4
python cli.py compress --method mlorq --total-memory-budget-mb 400
python cli.py compress --method kmeans --n-clusters 128
python cli.py compress --method faq --bits 6
```

---

## 🧬 Detailed Descriptions

### 1. UniQL – Unified Low‑Rank + Quantisation (`uniql`)
- **What it does:** Performs truncated SVD on each linear layer, then uniformly quantises the factors `U`, `S`, `Vt` and the bias (if present) to a specified bit width.
- **Parameters:** `rank`, `bits` (default 8)
- **When to use:** A solid baseline for joint compression; fast and deterministic.

### 2. MLoRQ – Multi‑Layer Optimal Rank & Quantisation (`mlorq`)
- **What it does:** Given a total memory budget (in MB), automatically searches over per‑layer ranks and bit widths to maximise overall compression while staying within the budget.
- **Parameters:** `total_memory_budget_mb`, `bits_candidates` (list of ints, e.g., `[2,4,8]`)
- **When to use:** When you have a hard memory constraint (e.g., deploying on a mobile device) and need the best possible accuracy under that budget.

### 3. KMeans – Weight Clustering (`kmeans`)
- **What it does:** Replaces each weight matrix with a set of cluster centroids and an index matrix. The forward pass reconstructs weights by looking up the centroids.
- **Parameters:** `n_clusters` (number of centroids, typically 64–256)
- **When to use:** For extreme compression when a small set of distinct weight values is acceptable; popular in post‑training quantisation pipelines.

### 4. FAQ – Future‑Aware Quantisation (`faq`)
- **What it does:** Runs a forward pass with a few calibration texts, captures activation ranges from the **next** layer, and uses those ranges to set the quantisation scale for the current layer. This improves accuracy compared to naïve uniform quantisation.
- **Parameters:** `bits`, `calibration_texts` (list of strings), requires a `tokenizer` at initialisation.
- **When to use:** When you want to preserve task accuracy after quantisation and have a few representative samples.

### 5. CoopQ – Cooperative Quantisation (`coopq`)
- **What it does:** Assigns bit widths to layers proportionally to their **importance** (approximated by the max absolute weight value). Layers with larger weights get more bits.
- **Parameters:** `bits` (base bit width, adjusted per layer)
- **When to use:** When different layers have different sensitivity to quantisation (typical in Transformers).

### 6. Astro – Simple Uniform Quantisation (`astro`)
- **What it does:** Quantises all linear layers and biases to a uniform grid – the simplest possible quantiser.
- **Parameters:** `bits`
- **When to use:** As a baseline or for quick experiments.

---

## ⚙️ Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` / `target_rank` | int | 64 | Number of singular components kept (for low‑rank methods) |
| `bits` | int | 8 | Quantisation bit‑width |
| `n_clusters` | int | 256 | Number of centroids for k‑means |
| `total_memory_budget_mb` | int | 500 | Maximum memory budget for MLoRQ |
| `bits_candidates` | list[int] | [2,4,8] | Candidate bit‑widths for MLoRQ |
| `calibration_texts` | list[str] | ["Quant"] | Texts for FAQ activation measurement |

---

## 📊 Parameter Savings (DistilBERT‑base, 768×768 linear layer)

| Method | Rank / Clusters | Bits | Parameters (approx) | Reduction |
|--------|-----------------|------|--------------------|-----------|
| Original | – | 32 | 590,000 | 1× |
| UniQL | 64 | 8 | 98,304 × 0.25 (8‑bit) ≈ 24k effective | ~25× |
| MLoRQ | auto | auto | depends on budget | up to 100× |
| KMeans | 256 clusters | – | 256 + index overhead | ~10× (varies) |
| FAQ / Astro / CoopQ | – | 4‑8 | same param count, smaller storage | ~4‑8× storage reduction |

*Storage reduction ≠ theoretical parameter count; quantisation reduces the bits per weight, not the number of weights.*

---

## 🔌 Integration with the CLI

All compressors can be used from the command line:

```bash
# UniQL
python cli.py compress --method uniql --target-rank 32 --custom-param 4

# MLoRQ
python cli.py compress --method mlorq --total-memory-budget-mb 400

# KMeans
python cli.py compress --method kmeans --n-clusters 128

# FAQ (requires a tokenizer – pass model name)
python cli.py compress --method faq --model distilbert-base-uncased --bits 6

# Astro
python cli.py compress --method astro --bits 4
```

After compression, the model is saved and can be loaded normally with `from_pretrained`.

---

## 📂 File Structure

```
framework/compression/
├── ...
├── uniql_compressor.py
├── mlorq_compressor.py
├── kmeans_quantize.py
├── faq_quantizer.py
├── coopq_quantizer.py
├── astro_quantizer.py
├── huffman_coder.py
├── nli_softmax.py
└── README.md
```

---

## 🛠 Creating Your Own Quantiser

1. Inherit from `BaseCompressor`.
2. In `compress()`, iterate over `nn.Linear` modules and replace or modify their weights.
3. Decorate with `@COMPRESSOR_REGISTRY.register("your_key")`.
4. Return `(model, stats)`.

---

## 📜 License

Part of the **Efficient Transformer Toolkit** – MIT License.
