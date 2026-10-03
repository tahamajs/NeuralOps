<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Attention Architectures

### 2.1.1 Sparse and Sliding-Window Attention

Longformer (Beltagy et al., 2020) combines a sliding window with a small 
set of global tokens. BigBird (Zaheer et al., 2020) adds random attention. 
Infini-Attention differs by using a **learned** memory instead of fixed 
global tokens.

### 2.1.2 Compressive Memory

Compressive Transformer (Rae et al., 2020) compresses past activations into 
a small buffer. Transformer-XL (Dai et al., 2019) caches the previous 
segment's hidden states. Our memory is a **learned matrix** that persists 
across all tokens, not a cache of past activations.

### 2.1.3 Linear Attention

Performer (Choromanski et al., 2021) and Linear Transformer (Katharopoulos 
et al., 2020) replace softmax with kernel-based approximations, achieving 
$\mathcal{O}(L)$ complexity. Infini-Attention retains softmax but on a 
sliding window plus a fixed memory.

### 2.1.4 Recurrent Memory

Recurrent Memory Transformer (Bulatov et al., 2022) uses special memory 
tokens. Infini-Attention's memory is a matrix, not tokens, and applies to 
every position.

## 2.2 Memorizing Transformers

Wu et al. (2022) add a kNN lookup to a stored database of past keys and 
values. Our memory is **learned** and **fixed-size**, not a database.

## 2.3 Gated Attention Fusion

Gating mechanisms are common in transformers (GTrXL, Parisotto et al., 2020). 
Our gate $\sigma(\beta)$ is a scalar per attention head, initialized to make 
the memory nearly silent at start.

## 2.4 Parameter-Efficient Fine-Tuning

LoRA (Hu et al., 2021) is the standard for freezing base weights and adding 
low-rank adapters. Infini-Attention uses LoRA for recovery.

## 2.5 Position of Our Work

| Axis | Prior work | Infini-Attention |
|---|---|---|
| **Local attention** | Sliding window | Sliding window |
| **Global context** | Fixed global tokens or kNN | **Learned memory matrix** |
| **Fusion** | Concatenation | **Gated scalar** |
| **Multi-architecture** | Usually one family | **Four families** |
| **Parameter effect** | Varies | **Increases** (memory + LoRA) |
| **Gate init** | N/A | **β = -5 (memory silent)** |

**Key distinction.** Prior compressive memory methods either (a) compress 
past activations (Compressive Transformer), or (b) query a database (kNN). 
Infini-Attention learns a **fixed-size parametric memory** per head, 
trained jointly with LoRA.

## 2.6 Comparison to Companion Modules

| Module | Global mechanism | Params | Multi-arch |
|---|---|---|---|
| NHA | RFF global | +LoRA | Yes |
| EAT | Token pruning | +LoRA | Yes |
| A3 | Low-rank factorization | **−50%** | Yes |
| **Infini-Attention** | **Learned memory** | **+memory+LoRA** | **Yes** |

---