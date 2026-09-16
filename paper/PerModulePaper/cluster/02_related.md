# 2. Related Work

## 2.1 Sparse Attention

**Longformer.** Beltagy et al. (2020) combine a sliding window of radius 
$W$ with a small set of global tokens. Complexity: $\mathcal{O}(L W)$. 
ClusterFormer captures more global context at similar cost.

**BigBird.** Zaheer et al. (2020) add random attention to sliding-window. 
Theoretically Turing-complete; empirically similar to Longformer.

**Sparse Transformer.** Child et al. (2019) use strided and fixed patterns. 
Orthogonal to clustering.

## 2.2 Kernel and Linear Attention

**Linear Transformer.** Katharopoulos et al. (2020) replace softmax with 
$\phi(x) = \text{elu}(x) + 1$, achieving $\mathcal{O}(L d^2)$ via 
associative reordering.

**Performer.** Choromanski et al. (2021) approximate softmax via positive 
orthogonal random features (PORF). Complexity: $\mathcal{O}(L m d)$ with 
$m$ features.

**Nyströmformer.** Xiong et al. (2021) approximate the attention matrix 
via Nyström landmarks, achieving $\mathcal{O}(L m d)$.

## 2.3 Cluster-Based Attention

**Routing Transformer.** Roy et al. (2021) cluster tokens into $k$ groups 
and restrict attention within groups. Complexity: $\mathcal{O}(L \cdot L/k)$. 
ClusterFormer's centroid-to-key cross-attention is fundamentally different.

**Reformer.** Kitaev et al. (2020) use LSH to approximate attention. Its 
guarantee is probabilistic, not exact.

**Cluster-Former.** Wang et al. (2021) cluster at sentence granularity for 
document tasks. Not applicable to token-level attention.

**Pixelfly / Cluster Attention.** Various works explore clustering in vision 
transformers. ClusterFormer targets language.

## 2.4 MoE and Routing

**Switch Transformer.** Fedus et al. (2022) route tokens to experts. Similar 
routing mechanism but different purpose.

**Routing Attention.** Our framework's `routing` module clusters and routes 
sequentially. ClusterFormer is vectorized and differentiable.

## 2.5 Position of Our Work

| Method | Complexity | Global? | Local? | Differentiable? |
|---|---|---|---|---|
| Softmax | $\mathcal{O}(L^2)$ | Yes | Yes | Yes |
| Longformer | $\mathcal{O}(LW)$ | Partial | Yes | Yes |
| Performer | $\mathcal{O}(Lm)$ | Yes | No | Yes |
| Reformer | $\mathcal{O}(L\log L)$ | Yes | No | Approx. |
| Routing | $\mathcal{O}(L^2/k)$ | Partial | No | No (argmin) |
| **ClusterFormer** | $\mathcal{O}(KL)$ | **Yes** | **Yes** | **Yes** |

ClusterFormer is unique in combining:
1. **Exact** centroid-to-key attention (not approximate).
2. **Global + local** blending.
3. **Fully differentiable** clustering (softmax assignment).
4. **Multi-architecture** support including GQA and RoPE.

## 2.6 Comparison Table

| Feature | Longformer | Performer | Reformer | ClusterFormer |
|---|---|---|---|---|
| Complexity | $O(LW)$ | $O(Lm)$ | $O(L\log L)$ | $O(KL)$ |
| Global context | Partial | Yes | Yes | Yes |
| Local context | Yes | No | No | Yes |
| Differentiable | Yes | Yes | Approx. | Yes |
| Learnable hyperparams | 0 | 0 | 2 | **3** (τ, α, K) |

---