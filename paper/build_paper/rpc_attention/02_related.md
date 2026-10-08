<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Random Projections

### 2.1.1 Johnson–Lindenstrauss Lemma

Johnson & Lindenstrauss (1984) proved that $N$ points in $\mathbb{R}^d$ can 
be embedded into $\mathbb{R}^{\mathcal{O}(\log N / \epsilon^2)}$ with 
pairwise distances preserved up to factor $(1 + \epsilon)$. The embedding 
can be a random Gaussian matrix.

### 2.1.2 Applications

- **Nearest neighbor search:** Indyk & Motwani (1998) use JL for 
  locality-sensitive hashing.
- **Sketching:** Woodruff (2014) surveys sketching for numerical linear 
  algebra.
- **Kernel approximation:** Rahimi & Recht (2007) use random features 
  for kernel machines.

### 2.1.3 Sparse Random Projections

Achlioptas (2003) shows that sparse random matrices (entries in 
$\{-1, 0, +1\}$) also preserve distances with similar guarantees, 
often faster to apply. We use dense Gaussian in RPC.

## 2.2 Efficient Attention

### 2.2.1 Sparse Attention

Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020). These 
restrict attention patterns; RPC keeps dense attention but in a lower 
dimension.

### 2.2.2 Linear Attention

Performer (Choromanski et al., 2021) uses random features to approximate 
softmax. Linear Transformer (Katharopoulos et al., 2020) uses $\phi(x) = \text{elu}(x) + 1$. 
RPC differs: it sketches Q and K but keeps softmax on the sketched scores.

### 2.2.3 Low-Rank Attention

Linformer (Wang et al., 2020) projects K, V along the sequence dimension. 
Nyströmformer (Xiong et al., 2021) uses Nyström landmarks. RPC projects 
Q, K along the head dimension.

### 2.2.4 FlashAttention

Dao et al. (2022) avoids materializing the attention matrix but does not 
reduce FLOPs. RPC reduces FLOPs directly.

## 2.3 Distinction from Companion Modules

| Module | Approach | Params | Memory |
|---|---|---|---|
| NHA | Hybrid local + RFF global | +LoRA | Same |
| AASVD | Low-rank factorization | **−50%** | Less |
| EAT | Token pruning | +LoRA | Less |
| OHA | Compressed global branch | +compress+gate+LoRA | Same |
| **RPC** | **Q/K sketching** | **+LoRA** | **Same** |

**Key distinction.** RPC is the only module that reduces attention FLOPs 
**without reducing the score matrix size**. It trades inner-product 
precision for compute.

## 2.4 Position of Our Work

| Axis | Prior work | RPC |
|---|---|---|
| **Score matrix** | Full $L \times L$ | Full $L \times L$ |
| **Score FLOPs** | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L d_h r + L^2 r)$ |
| **Projection** | Fixed or learned | **Fixed random** |
| **Memory** | Same as baseline | Same as baseline |
| **Multi-architecture** | Usually one family | Four families |

---