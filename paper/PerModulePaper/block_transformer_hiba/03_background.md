# 3. Background

To formalize the HIBA architecture, we first establish the mathematical foundations of the components it modifies and integrates.

## 3.1 Multi-Head Self-Attention (MHSA)

For an input sequence $X \in \mathbb{R}^{B \times L \times d}$, standard Multi-Head Self-Attention computes:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \tag{3.1}$$

$$\text{head}_h = \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}} + M\right) V_h \tag{3.2}$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h H}$, $d_h$ is the head dimension, and $M$ is an optional causal or padding mask. The intermediate attention matrix $Q_h K_h^\top \in \mathbb{R}^{L \times L}$ dictates the $\mathcal{O}(L^2)$ memory bottleneck during long-context inference.

## 3.2 Grouped-Query Attention (GQA)

Modern large language models (e.g., Qwen, LLaMA) utilize Grouped-Query Attention to reduce the KV-cache footprint. In GQA, the number of key/value heads $H_{kv}$ is strictly less than the number of query heads $H$:

$$K, V \in \mathbb{R}^{B \times L \times H_{kv} d_h} \tag{3.3}$$

During attention computation, the KV heads are broadcasted (repeated) by a factor of $H / H_{kv}$ to match the query heads. HIBA natively supports GQA by preserving this dimensionality constraint and dynamically repeating keys and values prior to the block-sparse routing.

## 3.3 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) encodes absolute positional information directly into queries and keys via a rotation matrix. Let $x_{m}$ be a vector at position $m$. RoPE applies a rotation parameterized by frequency $\theta$:

$$\tilde{x}_{m} = x_{m} \cos(m\theta) + \mathrm{rot}(x_{m}) \sin(m\theta) \tag{3.4}$$

Crucially, in models like Qwen3.5, RoPE is applied only to a partial subspace of the head dimension (e.g., `rotary_dim = 64` out of `head_dim = 256`). HIBA explicitly models this partial-RoPE dimension slicing, rotating only the specified coordinate subset and concatenating the unrotated pass-through coordinates to ensure exact numerical parity with the base model.

## 3.4 Low-Rank Adaptation (LoRA)

To recover performance after replacing the attention mechanism, we employ LoRA (Hu et al., 2021). The effective weight matrix is computed as:

$$W_{\text{eff}} = W_{\text{base}} + \frac{\alpha}{r} (B A) \tag{3.5}$$

where $W_{\text{base}}$ is frozen, $A \in \mathbb{R}^{r \times k}$ is initialized from a normal distribution $\mathcal{N}(0, \sigma^2)$, and $B \in \mathbb{R}^{d \times r}$ is initialized to zero. This ensures that at step zero ($\Delta W = 0$), the model behavior is completely dictated by the pretrained weights, allowing for stable recovery fine-tuning.


---