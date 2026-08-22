# Bidirectional Selective State Space Model (Bi-Mamba) Attention

## 1. Executive Abstract & Architectural Motivation
Traditional Transformer Multi-Head Self-Attention scales quadratically $\mathcal{O}(L^2)$ with sequence length $L$, creating severe computational and memory bottlenecks on long sequences. While State Space Models (SSMs) like Mamba achieve strictly linear $\mathcal{O}(L)$ scaling via selective discretization and causal state recurrence, applying causal unidirectional SSMs directly to pretrained bidirectional masked language models (like DistilBERT on SST-2) causes severe representational mismatch (stalling at 50.92% random guessing baseline).

**Bidirectional Mamba (Bi-Mamba)** repairs this fundamental mismatch:
1. **Bidirectional Selective Scanning**: Computes forward ($\vec{h}_t$) and backward ($\overleftarrow{h}_t$) hidden state-space recurrences across the sequence, allowing every token to attend symmetrically to past and future context.
2. **Pretrained Attention Weight Transfer**: Transfers pretrained Multi-Head Attention query, key, and value projection matrices into the content input branch $U$ and multiplicative gating branch $Z$.
3. **Symmetric 1D Depthwise Convolution**: Applies non-causal symmetric 1D depthwise convolution across token representations to preserve local phrase n-gram correlations before SSM discretization.
4. **Contractive Lyapunov Stability**: Enforces diagonal state transition matrix $\bar{A} = \exp(-\exp(A_{\text{log}}) \odot \Delta t) \in (0, 1)$ to guarantee strict contractivity and prevent gradient explosion.

---

## 2. Mathematical Formulations & Derivations

### 2.1 Input Projection & Gating
Given token activations $X \in \mathbb{R}^{B \times L \times D}$:
$$[U, Z] = X W_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}}$$
where $U$ is the content branch and $Z$ is the gating branch.

### 2.2 Symmetric 1D Convolution
$$\widetilde{U} = \text{SiLU}(\text{Conv1D}(U^\top)^\top) \in \mathbb{R}^{B \times L \times D_{\text{inner}}}$$
with kernel size $d_{\text{conv}} = 4$ and symmetric boundary padding.

### 2.3 Input-Dependent Selective Discretization
$$\Delta_t = \text{softplus}(\widetilde{U}_t W_\Delta + b_\Delta) \in \mathbb{R}^{B \times D_{\text{inner}}}$$
$$B_t = \widetilde{U}_t W_B \in \mathbb{R}^{B \times d_{\text{state}}}, \quad C_t = \widetilde{U}_t W_C \in \mathbb{R}^{B \times d_{\text{state}}}$$
$$\bar{A}_t = \exp(-\exp(A_{\text{log}}) \odot \Delta_t) \in (0, 1)^{B \times D_{\text{inner}} \times d_{\text{state}}}$$

### 2.4 Bidirectional Recurrent Scanning
$$\vec{h}_t = \bar{A}_t \odot \vec{h}_{t-1} + (\Delta_t \odot \widetilde{U}_t) \otimes B_t$$
$$\overleftarrow{h}_t = \bar{A}_t \odot \overleftarrow{h}_{t+1} + (\Delta_t \odot \widetilde{U}_t) \otimes B_t$$

$$\vec{y}_t = \sum_{s=1}^{d_{\text{state}}} \vec{h}_{t, :, s} \cdot C_{t, s}, \quad \overleftarrow{y}_t = \sum_{s=1}^{d_{\text{state}}} \overleftarrow{h}_{t, :, s} \cdot C_{t, s}$$
$$Y_t = \frac{1}{2}(\vec{y}_t + \overleftarrow{y}_t) + D \odot \widetilde{U}_t$$

### 2.5 SiLU Gating & Output Projection
$$\text{Output} = (Y \odot \text{SiLU}(Z)) W_{\text{out}} + b_{\text{out}}$$

---

## 3. Computational Complexity & Memory Scaling

| Metric | Standard Multi-Head Attention | Bidirectional Mamba |
| :--- | :--- | :--- |
| **Time Complexity** | $\mathcal{O}(B \cdot L^2 \cdot D)$ | $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ |
| **Memory Footprint** | $\mathcal{O}(B \cdot H \cdot L^2)$ | $\mathcal{O}(B \cdot D \cdot d_{\text{state}})$ |
| **Context Horizon Scaling** | Quadratic $\mathcal{O}(L^2)$ | Strictly Linear $\mathcal{O}(L)$ |
| **Inference Step Memory** | $\mathcal{O}(L \cdot D)$ (KV-Cache) | $\mathcal{O}(D \cdot d_{\text{state}})$ (Fixed-size state) |

---

## 4. Empirical Analysis of SST-2 Results

### Why Unidirectional Models Stalled at 50.92%
1. **Representational Mismatch**: DistilBERT's classification head relies on bidirectional contextualization at index 0 (`[CLS]`). In a causal unidirectional scan ($t=1 \to L$), token `[CLS]` at $t=0$ only sees itself and has zero representation of the downstream sentence, yielding 50.92% random guessing.
2. **Missing Pretrained Knowledge**: Pure random SSM initialization destroys pretrained linguistic features.
3. **Unbounded Recurrence**: Uncontrolled $\bar{A} \to 1.0$ causes state saturation over sequences.

### The Repair Impact
By adding **bidirectional scanning** (forward + backward passes), **pretrained projection weight transfer** ($W_v \to U, W_q \to Z, W_o \to W_{\text{out}}$), and **bounded log-space decay**, Bi-Mamba provides full contextual coverage and stable gradients across all classification tokens.
