<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention Architectures

### 2.1.1 Sparse and Sliding-Window Attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020). OHA 
uses a sliding window for the local branch, same as these methods.

### 2.1.2 Low-Rank Attention

Linformer (Wang et al., 2020) projects K, V to a fixed low rank. 
Nyströmformer (Xiong et al., 2021) uses Nyström approximation. OHA 
compresses the **sequence** of K, V, not their feature dimension.

### 2.1.3 Kernel Attention

Performer (Choromanski et al., 2021), Linear Transformer (Katharopoulos 
et al., 2020). OHA uses softmax attention on both branches.

### 2.1.4 Hybrid Attention

NHA (companion module) uses RFF. Infini-Attention uses a learned memory 
matrix. OHA uses **soft-attention pooling** for sequence compression.

## 2.2 Sequence Compression

### 2.2.1 Pooling-Based

Set Transformer (Lee et al., 2019) uses induced set attention with 
learnable "inducing points". OHA's compression is similar but simpler: 
a single softmax-attention layer.

### 2.2.2 Soft-Attention Pooling

Hierarchical Attention Networks (Yang et al., 2016) pool sequences via 
attention. OHA extends this to K, V compression.

### 2.2.3 Learned Memory

Compressive Transformer (Rae et al., 2020), Recurrent Memory Transformer 
(Bulatov et al., 2022). These maintain a memory across segments. OHA's 
compression is per-sequence, not across segments.

## 2.3 Gating

### 2.3.1 Gated Attention

GTrXL (Parisotto et al., 2020) uses gates for RL. FLASH (Hua et al., 2022) 
gates between local and global. OHA uses per-head gating initialized to 
favor local.

### 2.3.2 Adaptive Computation

Adaptive Computation Time (Graves, 2016), PonderNet (Banino et al., 2021). 
These adapt compute per token. OHA's gate is per-head per-token.

## 2.4 Position of Our Work

| Axis | Prior work | OHA |
|---|---|---|
| **Local branch** | Sliding window | Sliding window |
| **Global branch** | Fixed tokens / learned memory | **Soft-attention compressed summary** |
| **Compression** | Learned matrix | **Data-dependent** |
| **Gating** | Fixed / global | **Per-head, per-token** |
| **Multi-architecture** | Usually one family | Four families |
| **Parameter effect** | Varies | **Increases** |

**Key distinction.** Prior hybrid attention methods use either fixed global 
tokens (Longformer) or a fixed learned memory (Infini-Attention). OHA 
computes the compressed summary **on-the-fly** from the input, making it 
input-adaptive.

## 2.5 Comparison to Companion Modules

| Module | Global mechanism | Compression | Params |
|---|---|---|---|
| NHA | RFF kernel | N/A | +LoRA |
| Infini-Attention | Learned memory | N/A | +memory+LoRA |
| EAT | Token pruning | Pruning | +LoRA |
| **OHA** | **Soft-attn pooling** | **Sequence** | **+compress+gate+LoRA** |

---
