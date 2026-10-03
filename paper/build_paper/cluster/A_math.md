# Appendix A. Mathematical Derivations

## A.1 Complexity Derivation

**Claim.** ClusterFormer's attention cost is $\mathcal{O}(K L d_h)$.

**Proof.** Three components:

1. **K-means.** Each iteration costs:
   - Assignment: `cdist` between $Q \in \mathbb{R}^{BHLd}$ and $C \in \mathbb{R}^{BH K d}$, cost $\mathcal{O}(BHLKd)$.
   - Update: gather $Q$ by cluster id, cost $\mathcal{O}(BHLd)$.
   - Three iterations: $\mathcal{O}(3 BHLKd)$.

2. **Centroid-to-key.** Product $(BH K d) \times (BH d L)$ = $\mathcal{O}(BHKLd)$.

3. **Soft assignment.** Product $(BH L d) \times (BH d K)$ = $\mathcal{O}(BH LKd)$.

Sum is $\mathcal{O}(BHLKd)$. $\square$

## A.2 K-Means Convergence

**Claim.** Lloyd's algorithm decreases the objective monotonically.

**Proof.** Each iteration:

1. **Assignment step:** For fixed centroids, assigning each point to 
   nearest centroid minimizes the objective given centroids.
2. **Update step:** For fixed assignments, the mean minimizes the sum of 
   squared distances.

Both steps decrease the objective or keep it unchanged. Since the objective 
is bounded below by 0, the algorithm converges. $\square$

## A.3 Soft Assignment Approximation

The soft assignment with temperature $\tau$:

$$
\text{assign}_{i,k} = \frac{\exp(Q_i \cdot C_k / (\sqrt{d_h} \tau))}{\sum_{k'} \exp(Q_i \cdot C_{k'} / (\sqrt{d_h} \tau))}
\tag{A.1}
$$

**Limit.** As $\tau \to 0$, this becomes a hard assignment (one-hot). As 
$\tau \to \infty$, it becomes uniform.

## A.4 Approximation Error Bound

**Statement.** The approximation error of ClusterFormer satisfies:

$$
\|\text{Output}_\text{CF} - \text{Output}_\text{exact}\|_F \leq f(\text{K-means distortion})
\tag{A.2}
$$

where $f$ is monotone increasing. We do not provide a tight bound; see 
empirical analysis in Section 8.

## A.5 Causality Preservation

**Claim.** For causal decoders, ClusterFormer preserves causality.

**Proof.** We use exact causal attention for decoders (Section 5.4):

$$
\text{Attn}_i = \sum_{j \leq i} \alpha_{ij} V_j
\tag{A.3}
$$

Only keys with $j \leq i$ contribute. Causality is preserved. $\square$

## A.6 LoRA Parameter Count

For each layer, LoRA adds:
- $r_L \times d$ for $A_Q, A_K, A_V, A_O$
- $d \times r_L$ for $B_Q, B_K, B_V, B_O$

Total per layer: $8 r_L d$.

Across $N_l$ layers: $8 r_L d N_l$.

For DistilBERT ($d=768$, $r_L=16$, $N_l=6$): $8 \times 16 \times 768 \times 6 = 589{,}824$.

---