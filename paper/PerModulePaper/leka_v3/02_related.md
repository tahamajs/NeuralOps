# 2. Related Work

## 2.1 Kernel and Linear Attention

**Performer** (Choromanski et al., 2021) uses positive orthogonal random 
features (PORF) to approximate the softmax kernel. **Linear Transformer** 
(Katharopoulos et al., 2020) uses $\phi(x) = \text{elu}(x) + 1$. 
**Random Feature Attention** (Peng et al., 2021) uses Gaussian RFF.

LEKAv3 differs by using **multi-scale cos/sin features** and by applying 
the kernel to a **selected set of landmarks** rather than all keys.

## 2.2 Landmark-Based Attention

**Nyströmformer** (Xiong et al., 2021) selects landmarks by Nyström 
approximation, using fixed sampling. **Luna** (Ma et al., 2021) uses 
learned landmarks with a query-averaging heuristic.

LEKAv3's landmark scorer is **learned** and operates on **kernel features** 
(not raw keys), giving $\mathcal{O}(L d_\text{feat})$ cost.

## 2.3 Sparse and Local Attention

**Longformer** (Beltagy et al., 2020) combines a sliding window with global 
tokens. **BigBird** (Zaheer et al., 2020) adds random attention. 
**Star-Transformer** (Guo et al., 2019) uses a star topology.

LEKAv3's local branch uses a **causal-aware sliding window** with a learnable 
fusion gate.

## 2.4 Hybrid Attention

**H-Transformer-1D** (Zhu & Soricut, 2021) uses hierarchical attention. 
**Reformer** (Kitaev et al., 2020) uses LSH-bucketed attention. LEKAv3's 
landmark selection is a form of content-based routing.

## 2.5 Position of Our Work

| Method | Complexity | Learnable selector? | Local branch? | Causal? |
|---|---|---|---|---|
| Performer | $\mathcal{O}(L)$ | No (fixed features) | No | Partial |
| Nyströmformer | $\mathcal{O}(L)$ | No | No | Partial |
| Longformer | $\mathcal{O}(L W)$ | No | Yes | Yes |
| BigBird | $\mathcal{O}(L W)$ | No | Yes | Yes |
| **LEKAv3** | $\mathcal{O}(L \cdot L_\text{land} + L W)$ | **Yes** | **Yes** | **Yes** |

**Key distinction.** LEKAv3 is the first landmark-based attention where the 
landmark selection is **trained jointly** with the downstream model, rather 
than fixed at initialization.

---

## Comparison Table

| Method | Global | Local | Selector | Complexity |
|---|---|---|---|---|
| Standard | ✓ | ✓ | — | $\mathcal{O}(L^2)$ |
| Performer | ✓ | ✗ | fixed | $\mathcal{O}(L)$ |
| Nyströmformer | ✓ | ✗ | fixed | $\mathcal{O}(L)$ |
| Longformer | ✓ | ✓ | fixed | $\mathcal{O}(L W)$ |
| NHA | ✓ | ✓ | fixed | $\mathcal{O}(L^2)$ |
| **LEKAv3** | ✓ | ✓ | **learned** | $\mathcal{O}(L L_\text{land} + L W)$ |

---