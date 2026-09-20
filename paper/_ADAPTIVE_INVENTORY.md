# AdaptiveRanks Paper Code and Result Inventory

**Canonical Compressor Name:** `AdaptiveRanks`  
**Registry Aliases:** `adaptive_ranks`, `adaptive_svd`, `adaptive`, `adaptive_compressor`, `adaptive_factory`, `adaptive_compressor_factory`, `adaptive_ranks_factory`, `adaptive_svd_factory`  
**Classes:** `AdaptiveRanksCompressor`, `AdaptiveTransformerCompressor`  
**Linear Layer:** `SVDLinear` (alias `AdaptiveSVDLinear`)

---

## 1. Source Files Inventory
- **LaTeX Paper Root:** `/Users/tahamajs/Documents/uni/Project/paper/PerModulePaper/adaptive_compressor/main.tex`
- **Synchronized Monolithic Paper:** `/Users/tahamajs/Documents/uni/Project/paper/PerModulePaper/adaptive_compressor/paper_full.tex`
- **Compiled PDF:** `/Users/tahamajs/Documents/uni/Project/paper/PerModulePaper/adaptive_compressor/main.pdf`
- **Canonical Python Code:** `/Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adaptive_ranks.py`
- **Wrapper/Alias Code:** `/Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/adaptive_compressor.py`
- **Multi-Model Benchmark Runner:** `/Users/tahamajs/Documents/uni/Project/scripts/run_all_4_adaptive_compressor.py`

---

## 2. Empirical Benchmark Data Sources
- `results/unified_adaptive_compressor/distilbert/summary.csv`:
  - Variant: `adaptive_compressor` (PASS_NO_COMPRESSION)
  - Accuracy: 90.71%
  - F1 Score: 0.9099
  - Parameters: 66,955,010 (1.00x)
  - Peak VRAM: 183.07 MB
  - Throughput: 49,536.87 tok/s
- `results/unified_adaptive_compressor/gpt2/summary.csv`:
  - Variant: `adaptive_compressor` (PASS)
  - Accuracy: 76.95%
  - F1 Score: 0.8016
  - Parameters: 124,177,153 (1.0021x reduction)
  - Peak VRAM: 638.52 MB
  - Throughput: 22,313.99 tok/s
  - Reconstruction Error: 0.1982
- `results/unified_adaptive_compressor/roberta/summary.csv`:
  - Variant: `adaptive_compressor` (PASS_NO_COMPRESSION)
  - Accuracy: 85.32%
  - F1 Score: 0.8447
  - Parameters: 127,325,954 (1.00x)
  - Peak VRAM: 326.48 MB
  - Throughput: 11,130.61 tok/s
- `results/unified_sweep/distilbert/summary.csv` (`adaptive` sweep):
  - Accuracy: 87.50%
  - Compression Rate: 1.0437x
  - Parameters After: 64,153,346

---

## 3. Mathematical Specifications
- **Energy-thresholded rank criterion:**
  $$\mathcal{E}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^\rho \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
- **Minimum rank floor:**
  $$r_{\min} = \max(16, \lfloor 0.15 \cdot \min(d_{\text{in}}, d_{\text{out}}) \rfloor, \lfloor 0.10 \cdot d_{\text{out}} \rfloor)$$
- **Activation covariance whitening:**
  $$\Sigma_X = \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{t=1}^N x_{b, t}^\top x_{b, t}, \quad M = W \cdot \Sigma_X^{1/2}$$
- **Symmetric square-root singular distribution:**
  $$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = (\sqrt{\Sigma_r} V^\top_{:r, :}) \Sigma_X^{-1/2}$$
- **Condition number preservation:**
  $$\kappa(W_r) = \frac{\sigma_1}{\sigma_r} \le \frac{\sigma_1}{\sigma_\rho} = \kappa(W)$$
- **Dynamical cascade stability:**
  $$\|X_L\|_F \le \|X_0\|_F \exp\left(\sqrt{N} \sum_{l=1}^L \|W_{V, r}^{(l)} W_{O, r}^{(l)}\|_2\right) < \infty$$

---

## 4. Citation and Reference Coverage
- **Total Bibliographic Entries:** 26 verified references.
- **Coverage:** Low-Rank Factorization (Eckart-Young, Halko, SVD-LLM, ASVD, FWSVD, DRONE, ALBERT, Tensor-Train), Quantization (GPTQ, AWQ, SmoothQuant), Pruning & Sparsity (Wanda), Saliency & Continual Learning (Kirkpatrick, Molchanov), Efficient Attention (Vaswani, Katharopoulos, Performer, Nyströmformer), Foundation Architectures (DistilBERT, RoBERTa, GPT-2, LLaMA, LoRA), Systems & Energy (PagedAttention, Horowitz, Strubell).
- **Undefined Citations:** 0
- **Undefined References:** 0
