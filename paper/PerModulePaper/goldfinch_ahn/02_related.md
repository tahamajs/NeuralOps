<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Hybrid Attention

**Longformer** (Beltagy et al., 2020). Sliding-window + global tokens. 
Longformer's window is fixed; ours is token-adaptive via a gate.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global. 
Theoretically universal.

**Hybrid Transformer** (Zhu et al., 2021). Local attention + low-rank 
global. Similar in spirit.

**Our approach.** Sliding-window + linear projection + preservation gate.

## 2.2 Sliding-Window Attention

**Longformer** (Beltagy et al., 2020). The canonical sliding-window 
implementation.

**Mistral** (Jiang et al., 2023). Uses sliding window for efficiency.

**Our approach.** Sliding-window with an input-dependent gate.

## 2.3 Residual and Preservation

**LayerScale** (Touvron et al., 2021). Multiplies residual branch by small 
learnable scalar.

**ReZero** (Bachlechner et al., 2021). Zero-initialized residual branch 
with learnable scalar. Ensures identity at initialization.

**Our approach.** Preservation gate blends output with input via learnable 
$\alpha = \sigma(2.0) \approx 0.88$. Similar to ReZero but with a 
non-zero initialization.

## 2.4 Linear Attention and State-Space

**Linear Transformer** (Katharopoulos et al., 2020). Linear attention.

**Gated DeltaNet** (Yang et al., 2024). Delta-rule recurrent.

**Note.** Our long-memory branch is a **linear projection**, not a recurrent 
mechanism. See Section 1.3.

## 2.5 Position of Our Work

| Axis | Prior work | Ours |
|---|---|---|
| **Local attention** | Sliding window | Sliding window |
| **Long memory** | Recurrent | **Linear projection** |
| **Mixing** | Fixed or gate | Input-dependent gate |
| **Preservation** | LayerScale, ReZero | Preservation gate |

**Key distinction.** GoldFinch-AHN is **conservative**: the preservation 
gate keeps the block close to identity. This reduces risk but also reduces 
potential gains.

---

## Comparison Table

| Method | Local | Long | Gate | Preserve |
|---|---|---|---|---|
| Longformer | Window | Global tokens | Fixed | No |
| BigBird | Window | Random + global | Fixed | No |
| ReZero | N/A | N/A | N/A | Yes |
| LayerScale | N/A | N/A | N/A | Yes |
| **GoldFinch-AHN** | **Window** | **Linear** | **Input** | **Yes** |

---