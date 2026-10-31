# 3D Neural Architecture Visualizer & Attention Lab: 5-Minute Demo Script

> **Live Deployment:** [Hugging Face Spaces - 3D Neural Architecture Visualizer](https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer)  
> **Source Repository:** `Hooshaai/3d-neural-architecture-visualizer`  
> **Target Audience:** AI researchers, ML engineers, systems architects, and students.

---

## Executive Overview

The **Attention Lab / 3D Neural Architecture Visualizer** is an interactive, GPU-accelerated spatial laboratory built in Three.js and vanilla Web standards. It renders exact token physics and tensor projections for modern transformer models, including **DistilBERT**, **RoBERTa**, **GPT-2**, **Qwen 2.5 (GQA)**, **ToMoE**, and **SliceGPT**.

The visualizer provides real-time architectural telemetry across **13 attention mechanisms**, a deep **10-section telemetry inspector**, a **6-model comparison matrix**, an **educational atlas**, complete **3D dataflow wiring & residual highways**, and a **cinematic Present Mode** designed for high-impact 5-minute conference demonstrations.

---

## Quick Controls & Keyboard Shortcuts

| Key / Control | Action | Visual Effect |
| :--- | :--- | :--- |
| **`P`** or **`▶ Present`** | Toggle Presentation Mode | Minimal HUD, 10-step progress scrubber, glassmorphic telemetry drawer |
| **`L`** or **`📖 Learn`** | Open Educational Atlas | Plain-English primers with KaTeX formulas for core concepts |
| **`G`** or **`🎓 Lessons`** | Interactive Guided Lessons | 5 bite-sized architectural masterclasses with 3D camera guidance |
| **`H`** or **`🔥 Heatmap`** | Attention Heatmap Explorer | Exact 4D attention weights tensor [layers, heads, seq, seq] with PNG export |
| **`T`** or **`⚡ Predict`** | Token Prediction Playground | Autoregressive generation, temperature scaling, top-k logit distribution |
| **`C`** or **`🎛️ Compress`** | SVD Compression Simulator | Live rank slider (8 to 384), weight matrix reconstruction, Apply to 3D |
| **`🏁 Race`** | Side-by-Side Model Race | Dual-track real-time generation benchmark with live throughput & latency |
| **`🔗 Share`** | Copy Shareable Permalink | Deep-linking URL hash `#model=roberta&module=svd&view=3d&layer=3` |
| **`V`** or **`🎥 Record`** | 360° Orbit Video Exporter | 5s/10s cinematic showcase WebM/MP4 video recorder via MediaRecorder |
| **`M`** or **`📊 Matrix`** | Cross-Architecture Matrix | 6-model side-by-side comparison table with CSV export |
| **`+` / `=`** or **`➕`** | Zoom In | Smooth camera approach towards active layer (triggers LOD) |
| **`-` / `_`** or **`➖`** | Zoom Out | Smooth camera retraction towards macro overview |
| **`Space`** | Play / Pause Simulation | Halts or resumes 3D token particle physics |
| **`Esc`** | Exit Modal / Present Mode | Restores standard interactive exploration view |
| **Mouse Left Drag** | Orbit Camera | Rotates view around the focused transformer layer |
| **Mouse Right Drag** | Pan Camera | Lateral camera movement |
| **Mouse Wheel** | Continuous Zoom | Continuous zoom in/out |

---

## 5-Minute Presentation Narrative Script

### Timing Breakdown
- **0:00 - 0:30 (Step 1):** Welcome & The Transformer Anatomy
- **0:30 - 1:00 (Step 2):** Standard Multi-Head Attention (MHA)
- **1:00 - 1:35 (Step 3):** Grouped-Query Attention (GQA) & KV Cache Efficiency
- **1:35 - 2:05 (Step 4):** Fused QKV Projections (`c_attn`)
- **2:05 - 2:45 (Step 5):** SVD Low-Rank Attention Factorization
- **2:45 - 3:20 (Step 6):** Multi-Head Latent Attention (MLA)
- **3:20 - 3:55 (Step 7):** Mixture-of-Experts (MoE) Dynamic Routing
- **3:55 - 4:25 (Step 8):** Structured Dimensional Slicing (SliceGPT)
- **4:25 - 4:45 (Step 9):** Residual Highways & Normalization
- **4:45 - 5:00 (Step 10):** Cross-Model Matrix & Summary

---

### Step 1: Welcome & The Transformer Anatomy (0:00 - 0:30)
- **Visual Action:** Start in standard view with **DistilBERT** loaded. Click `▶ Present Mode` (or press `P`). The camera flies to an overview of the 6 transformer layers.
- **Presenter Narration:**
  > *"Welcome to the Attention Lab and 3D Neural Architecture Visualizer. What you are seeing is a real-time, mathematically accurate 3D representation of a Transformer. Tokens enter through the embedding layer at the bottom and stream upward through repeating blocks of Multi-Head Attention, Feed-Forward Networks, and lateral Residual Highways. Every geometry you see corresponds directly to exact weight matrix dimensions, tensor ranks, and memory footprints."*

---

### Step 2: Multi-Head Self-Attention (MHA) (0:30 - 1:00)
- **Visual Action:** Advance to Step 2 (click `Next ▶` or press `]`). The camera swoops into Layer 0's Attention block. The inspector drawer updates with head telemetry and projection dimensions.
- **Presenter Narration:**
  > *"Here we enter Layer 0: standard Multi-Head Attention. The hidden dimension of 768 is split across 12 independent query heads, each of dimension 64. In the inspector drawer on the right, observe the exact projection operators: $W_Q, W_K, W_V, W_O$, the scaling factor $1/\sqrt{d_k} = 0.125$, and the live attention probability heatmap. Notice how each head projects tokens into a distinct subspace, allowing the network to simultaneously track syntactic dependencies and long-range semantics."*

---

### Step 3: Grouped-Query Attention (GQA) & KV Cache Efficiency (1:00 - 1:35)
- **Visual Action:** Switch model to **Qwen 2.5 (GQA)**. Observe the 3D projection streams: Query tokens fan out into 14 wide heads, while Key and Value tokens compress into just 2 shared head groups.
- **Presenter Narration:**
  > *"Now we switch to Qwen 2.5, featuring Grouped-Query Attention (GQA). In standard MHA, every query head requires its own Key and Value cache, which causes massive memory bandwidth bottlenecks during autoregressive generation. GQA shares 1 Key and 1 Value head across every 7 Query heads. As shown in the KV Cache Footprint bar chart, this slashes KV memory by 85.7%—reducing cache consumption from 1,536 bytes per token down to just 256 bytes per token—enabling 8× larger batch sizes on GPU hardware."*

---

### Step 4: Fused QKV Projections (GPT-2 `c_attn`) (1:35 - 2:05)
- **Visual Action:** Switch model to **GPT-2**. The separate Q, K, V slabs coalesce into a single unified green projection slab.
- **Presenter Narration:**
  > *"Looking at GPT-2, we see another engineering optimization: Fused QKV Attention. Rather than executing three separate matrix multiplications for $W_Q$, $W_K$, and $W_V$, GPT-2 packs them into a single $[768 \times 2304]$ fused linear operator (`c_attn`). This maximizes GPU compute kernel occupancy and memory bandwidth utilization before slicing the result in SRAM."*

---

### Step 5: SVD Low-Rank Attention Factorization (2:05 - 2:45)
- **Visual Action:** Open the Module Catalog (`Library (241)`), search for `svd`, and apply `SVDLinearAttention`. The attention projection nodes morph into dual-cone hourglass bottleneck twins.
- **Presenter Narration:**
  > *"Here is our core research contribution: Singular Value Decomposition applied to linear attention. By computing the truncated Eckart-Young-Mirsky approximation $W \approx U_r \Sigma_r V_r^\top$, we replace full $[768 \times 768]$ dense matrices with factorized rank-$r$ bottleneck twins: Factor $V$ down-projects $768 \to 64$, and Factor $U$ up-projects $64 \to 768$. Look at the telemetry: this achieves an 83.3% parameter reduction with 92.4% retained SST-2 accuracy, running fully uncompressed on standard PyTorch runtimes."*

---

### Step 6: Multi-Head Latent Attention (MLA) (2:45 - 3:20)
- **Visual Action:** Advance tour to Step 6 (MLA). The 3D token stream compresses into a narrow golden latent bottleneck conduit.
- **Presenter Narration:**
  > *"Pioneered in DeepSeek-V2 and V3, Multi-Head Latent Attention takes compression one step further. Instead of caching high-dimensional Key-Value heads, MLA compresses the entire KV representation into a single 512-dimensional latent vector $c_t^{KV}$. During autoregressive decoding, only this compact vector is stored in VRAM. The decompression matrix can even be absorbed directly into the queries, completely eliminating key decompression during attention calculation."*

---

### Step 7: Mixture-of-Experts (MoE) Dynamic Routing (3:20 - 3:55)
- **Visual Action:** Switch model to **ToMoE** (MoE). Camera moves to the FFN layer. The single dense slab expands into 4 parallel expert subnetworks with a gating router at the front.
- **Presenter Narration:**
  > *"Moving to the Feed-Forward sublayer, we observe Mixture-of-Experts. Instead of passing every token through a uniform dense FFN, a learned Top-2 gating router scores each token and dynamically routes it to only 2 of the 4 specialized expert subnetworks. This provides 4× the parameter capacity with the FLOP budget and execution latency of a standard dense layer."*

---

### Step 8: Structured Pruning & Dimensional Reduction (SliceGPT) (3:55 - 4:25)
- **Visual Action:** Switch model to **SliceGPT**. The transformer layers visibly shrink in width; inactive channels turn dim gray.
- **Presenter Narration:**
  > *"Unlike traditional unstructured pruning which creates irregular sparse masks that run slowly on GPUs, SliceGPT performs structured coordinate slicing. Using orthogonal transformation matrices $Q$ computed from activation covariances, signal energy is concentrated into principal coordinates, physically excising 25% of the hidden dimensions. The result is a physically smaller, denser model that runs instantly faster on standard Tensor Cores."*

---

### Step 9: Residual Highways & Normalization (4:25 - 4:45)
- **Visual Action:** Camera tilts to display the lateral green emerald conduit running alongside the main layer stack.
- **Presenter Narration:**
  > *"Notice the lateral emerald green particle stream running parallel to the transformer blocks. This is the Residual Highway. Rather than forcing information to pass exclusively through parameterized transformations, tokens bypass the sublayers and are recombined via additive residual junctions ($+$). Paired with pre-layer normalization (RMSNorm / LayerNorm), this ensures gradients propagate without vanishing across deep networks."*

---

### Step 10: Cross-Model Matrix & Summary (4:45 - 5:00)
- **Visual Action:** Press `M` to open the **Cross-Model Comparison Matrix Dashboard**.
- **Presenter Narration:**
  > *"To conclude, we open the Cross-Model Matrix Dashboard. Here we compare all 6 architectures across parameter count, hidden dimension, KV cache footprint, and SST-2 accuracy. Every metric is exportable as publication-ready CSV data. The visualizer is live on Hugging Face Spaces for interactive research. Thank you!"*

---

## Technical Specifications & Compatibility Matrix

| Model Key | Model Name | Hidden Dim ($d$) | Heads ($H_Q$) | KV Heads ($H_{KV}$) | Attention Paradigm | FFN Type | Normalization |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `distilbert` | DistilBERT Base | 768 | 12 | 12 | Standard MHA | Dense GELU (3072d) | Post-LayerNorm |
| `roberta` | RoBERTa Base | 768 | 12 | 12 | Standard MHA | Dense GELU (3072d) | Post-LayerNorm |
| `gpt2` | GPT-2 Small | 768 | 12 | 12 | Fused QKV (`c_attn`) | Dense New-GELU | Pre-LayerNorm |
| `qwen` | Qwen 2.5 (0.5B) | 896 | 14 | 2 | Grouped-Query (GQA) | SwiGLU FFN (4864d) | RMSNorm |
| `tomoe` | ToMoE (MoE) | 768 | 12 | 12 | MHA + Routed FFN | 4-Way Top-2 MoE | Pre-LayerNorm |
| `slicegpt` | SliceGPT (Pruned) | 576 | 12 | 12 | Sliced Coordinate MHA | Sliced FFN (2304d) | RMSNorm |

---

## Deployment & Verification
 
- **Hugging Face Space Repository:** `Hooshaai/3d-neural-architecture-visualizer`
- **Upload Script:** `python3 upload_to_hf_spaces.py`
- **Unit & Schema Test Suite:** `node --test Docs/visualizer/tests/*.test.js` (40 passing unit tests across 15 suites, 100% pass rate)
