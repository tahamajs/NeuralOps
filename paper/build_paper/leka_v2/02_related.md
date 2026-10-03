# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) first replaced softmax 
with a kernel feature map. **Performer** (Choromanski et al., 2021) uses 
positive orthogonal random features. **cosFormer** (Qin et al., 2022) uses 
ReLU with cosine reweighting. **Nyströmformer** (Xiong et al., 2021) uses 
Nyström approximation with landmarks.

## 2.2 Landmark Methods

**Nyströmformer** uses Nyström landmarks selected via a downsampling rule. 
**Set Transformer** (Lee et al., 2019) uses inducing points. **Perceiver** 
(Jaegle et al., 2021) uses a learned latent array.

**Our distinction.** We select landmarks **per-example** via a learned 
scorer that operates on the RFF features. The selection is content-dependent 
and trained end-to-end.

## 2.3 Local-Global Hybrid

**Longformer** (Beltagy et al., 2020) combines local sliding-window 
attention with global tokens. **BigBird** (Zaheer et al., 2020) adds 
random attention. **Hymba** (Dong et al., 2024) and **Jamba** (Lieber 
et al., 2024) fuse attention and SSM.

**Our distinction.** We fuse **local sliding window** with **landmark 
linear attention**, both derived from the same Q/K/V projections.

## 2.4 Softmax Variants

**Softpick** (Gu & Dao, 2024) uses ReLU-damped softmax for stable 
masked attention. **Sparsemax** (Martins & Astudillo, 2016) projects 
onto the simplex. **Entmax** (Peters et al., 2019) uses $\alpha$-entmax.

**Our use.** We use softpick to handle the case where all rows of the 
attention matrix are masked (which causes standard softmax to give 
uniform attention).

## 2.5 Random Fourier Features

**Rahimi & Recht (2007)** introduced RFF for kernel approximation. 
**Yu et al. (2016)** proposed orthogonal random features with lower 
variance. We extend RFF with **dual-scale** features.

## 2.6 Position of Our Work

| Method | Linear | Landmark | Local | Softmax |
|---|---|---|---|---|
| Linear Transformer | Yes | No | No | — |
| Performer | Yes | No | No | RFF |
| Nyströmformer | Yes | Fixed | No | — |
| Longformer | No | No | Yes | Softmax |
| **LEKAv2** | **Yes** | **Learned** | **Yes** | **Softpick** |

---

## Comparison Table

| Method | Complexity | Landmark | Local | Kernel |
|---|---|---|---|---|
| MHSA | $O(L^2)$ | — | — | softmax |
| Linear Transformer | $O(L F)$ | No | No | elu+1 |
| Performer | $O(L F)$ | No | No | RFF |
| Nyströmformer | $O(L L_m)$ | Fixed | No | Nyström |
| Longformer | $O(L W)$ | No | Yes | softmax |
| **LEKAv2** | $O(L F + L W)$ | **Learned** | **Yes** | **Dual RFF + Softpick** |

---