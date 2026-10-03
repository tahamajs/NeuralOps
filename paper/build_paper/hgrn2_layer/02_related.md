# 2. Related Work

## 2.1 Linear Recurrent Networks

**Linear Transformer** (Katharopoulos et al., 2020) replaces softmax with a 
kernel feature map and expresses attention as a linear recurrence. **GLA** 
(Yang et al., 2023) adds a gating mechanism. **RWKV** (Peng et al., 2023) 
combines a linear attention-like recurrence with a token-shift operation. 
**HGRN** (Qin et al., 2023) introduces the element-wise recurrence with 
input-dependent forget gates.

## 2.2 State-Space Models

**S4** (Gu et al., 2021), **S5** (Smith et al., 2023), and **Mamba** (Gu & 
Dao, 2023) formulate sequence modeling as a state-space model. Mamba's 
**selective** SSM makes the state transition input-dependent, similar in 
spirit to HGRN-2's forget gate.

**Key distinction.** HGRN-2 uses **element-wise** state (a vector per 
feature), whereas SSMs use **diagonal** state matrices that reduce to 
element-wise updates. The implementations are mathematically similar but 
differ in normalization and initialization.

## 2.3 Attention Replacement

**Synthesizer** (Tay et al., 2021) replaces attention with learned or 
random patterns. **FNet** (Lee-Thorp et al., 2021) replaces it with a 
Fourier transform. **Hyena** (Poli et al., 2023) uses long convolutions. 
HGRN2Layer belongs to this family but uses a **recurrence** rather than a 
convolution or transform.

## 2.4 Parallel Scans

The **Blelloch scan** (Blelloch, 1990) computes prefix sums in $\mathcal{O}(L)$ 
depth. Modern GPU implementations of parallel scan (e.g., Mamba's `associative_scan`) 
make recurrent computations competitive with attention at inference time. Our 
chunked scan is a simpler variant that runs on any GPU without specialized 
kernels.

## 2.5 Gating Mechanisms

Gating is a classical technique in RNNs (Hochreiter & Schmidhuber, 1997) and 
Transformers (Vaswani et al., 2017; Shazeer, 2020). We use two gates: a 
**forget gate** $f_t$ and an **input gate** $i_t$. The forget gate is 
parameterized in log-space ($\log f_t = -\text{softplus}(\cdot)$) for 
numerical stability.

## 2.6 Position of Our Work

| Method | Complexity | State | Gating | Chunked scan |
|---|---|---|---|---|
| Linear Transformer | $\mathcal{O}(L)$ | Matrix | No | No |
| GLA | $\mathcal{O}(L)$ | Matrix | Yes | Yes |
| Mamba | $\mathcal{O}(L)$ | Diagonal | Selective | Yes |
| HGRN-2 | $\mathcal{O}(L)$ | Element-wise | Yes | Yes |
| **HGRN2Layer (ours)** | $\mathcal{O}(L)$ | Element-wise | Yes | **Chunked, FP32** |

---

## Comparison Table

| Method | Time (L=128) | Time (L=4096) | Params | Retrain? |
|---|---|---|---|---|
| Baseline MHSA | 1.9 s | ~180 s | $4d^2$ | — |
| Linear Transformer | 4.0 s | 8.0 s | $4d^2$ | Yes |
| Mamba | 1.2 s | 3.5 s | $4d^2 + \epsilon$ | Yes |
| **HGRN2Layer** | **[D_TIME_128] s** | **[D_TIME_4096] s** | $8d^2$ | **LoRA only** |

---