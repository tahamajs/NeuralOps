# 2. Related Work

## 2.1 Information Bottleneck

**Tishby et al. (1999)** introduced the Information Bottleneck principle 
for representation learning. **Alemi et al. (2017)** formulated the VIB 
(Variational IB) for deep networks. **Shwartz-Ziv & Tishby (2017)** 
analyzed IB in training dynamics.

Our work is the first (to our knowledge) to apply IB explicitly to the SSM 
branch of a hybrid architecture.

## 2.2 Hybrid Attention-SSM

**Jamba** (Lieber et al., 2024) interleaves transformer and Mamba layers. 
**Zamba** (Glorioso et al., 2024) shares attention blocks. **Hymba** (Dong 
et al., 2024) fuses attention and SSM in a single block. **Samba** (Ren et 
al., 2024) uses a similar design.

**Our distinction.** We add an **explicit bottleneck** between the input 
and the SSM branch, forcing the SSM to operate on compressed concepts.

## 2.3 Differential Attention

**Differential Transformer** (Ye et al., 2024) introduces two-Q-K 
subtraction for noise reduction. We use this as our attention branch with 
a softened damping ($\lambda = 0.2$ vs Ye's $0.8$).

## 2.4 State Space Models

**S4** (Gu et al., 2021), **Mamba** (Gu & Dao, 2023), **HGRN-2** (Qin et 
al., 2024) are linear-time recurrent models. Our SSM branch uses 
**SimpleSSM**, a lightweight variant.

## 2.5 Concept Bottlenecks

**Concept Bottleneck Models** (Koh et al., 2020) enforce human-interpretable 
intermediate concepts. Our bottleneck is not human-interpretable but 
serves a similar compression purpose.

**Autoencoders** use bottlenecks for dimensionality reduction. We use a 
bottleneck as an **information constraint** on the SSM input.

## 2.6 Position of Our Work

| Method | Attention | SSM | Bottleneck | IB-motivated |
|---|---|---|---|---|
| Standard MHSA | Single | No | No | No |
| Jamba | Single | Yes | No | No |
| Hymba | Single | Yes | No | No |
| Diff Transformer | Differential | No | No | No |
| **JambaInfo** | **Differential** | **Yes** | **Yes** | **Yes** |

**Key distinction.** JambaInfo is the first to combine differential 
attention, SSM, and an explicit information bottleneck.

---

## Comparison Table

| Method | Complexity | Bottleneck | Noise cancel | Params |
|---|---|---|---|---|
| MHSA | $O(L^2)$ | No | No | $4d^2$ |
| Jamba | $O(L^2 + L)$ | No | No | $7d^2$ |
| Hymba | $O(L^2 + L)$ | No | No | $7d^2$ |
| **JambaInfo** | $O(L^2 + L)$ | **Yes** | **Yes** | **$7d^2 + 2bd$** |

---