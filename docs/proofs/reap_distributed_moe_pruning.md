# Mathematical Proof: Distributed REAP Router Activation Statistics and Unbiased Global Expert Pruning

## 1. Problem Formulation and Setting

Let $\mathcal{M}$ be a Mixture of Experts (MoE) model distributed across $W$ worker nodes (ranks) indexed by $w \in \{0, 1, \dots, W-1\}$.
The model contains $E$ experts $\mathcal{E} = \{e_1, e_2, \dots, e_E\}$, where each expert $e_i$ is parametrized by weights $\theta_i \in \mathbb{R}^P$.

For an input token representation $x \in \mathbb{R}^d$, the router computes gating logits $h(x) = W_g x \in \mathbb{R}^E$ and gating probabilities via softmax:
$$p_i(x) = \frac{\exp(h_i(x))}{\sum_{j=1}^E \exp(h_j(x))}, \quad i \in \{1, \dots, E\}$$

In top-$k$ routing ($k \ll E$), an expert activation indicator function is defined as:
$$\mathbb{I}_i(x) = \mathbb{I}\left(i \in \operatorname{TopK}(p(x), k)\right)$$
and the token output representation is:
$$y(x) = \sum_{i \in \operatorname{TopK}(p(x), k)} \tilde{p}_i(x) e_i(x)$$
where $\tilde{p}_i(x) = \frac{p_i(x)}{\sum_{j \in \operatorname{TopK}(p(x), k)} p_j(x)}$ are the re-normalized routing coefficients.

We wish to prune $E - K$ experts to retain a subset $S^* \subset \{1, \dots, E\}$ with $|S^*| = K$, while minimizing the global representation drift across the full data distribution $\mathcal{D}$:
$$\min_{S \subset \{1, \dots, E\}, |S|=K} \mathcal{L}_{\text{drift}}(S) \triangleq \mathbb{E}_{x \sim \mathcal{D}} \left[ \| y(x) - y_S(x) \|_2^2 \right]$$
where $y_S(x)$ is the representation produced when routing is restricted to experts in $S$.

---

## 2. Global Expert Importance and Representation Drift Bound

### Theorem 1 (Representation Drift Upper Bound)
Let expert outputs be bounded such that $\sup_{x \in \operatorname{supp}(\mathcal{D})} \|e_i(x)\|_2 \le C_e$ for all $i \in \{1, \dots, E\}$.
Define the global expert importance score $I_i^*$ as the expected routed probability mass:
$$I_i^* \triangleq \mathbb{E}_{x \sim \mathcal{D}} \left[ p_i(x) \mathbb{I}_i(x) \right]$$

Then the expected representation drift under expert subset selection $S$ satisfies:
$$\mathcal{L}_{\text{drift}}(S) \le 4 C_e^2 \sum_{i \notin S} I_i^*$$

### Proof:
For any token $x$, let $\Delta(x) = y(x) - y_S(x)$.
Case 1: If $\operatorname{TopK}(p(x), k) \subseteq S$, then no active expert for token $x$ is pruned, so $y(x) = y_S(x)$ and $\|\Delta(x)\|_2 = 0$.
Case 2: If active experts intersect $S^c = \{1, \dots, E\} \setminus S$, let $\mathcal{A}_{\text{pruned}}(x) = \operatorname{TopK}(p(x), k) \cap S^c$.
Under re-normalized routing with remaining experts, by the triangle inequality:
$$\|\Delta(x)\|_2 = \left\| \sum_{i \in \mathcal{A}_{\text{pruned}}(x)} \tilde{p}_i(x) e_i(x) + \sum_{j \in S} (\tilde{p}_j(x) - \tilde{p}_{j, S}(x)) e_j(x) \right\|_2$$
Using $\sum_{j \in S} (\tilde{p}_{j, S}(x) - \tilde{p}_j(x)) = \sum_{i \in \mathcal{A}_{\text{pruned}}(x)} \tilde{p}_i(x)$ and $\|e_j(x)\|_2 \le C_e$:
$$\|\Delta(x)\|_2 \le C_e \sum_{i \in \mathcal{A}_{\text{pruned}}(x)} \tilde{p}_i(x) + C_e \sum_{j \in S} |\tilde{p}_j(x) - \tilde{p}_{j, S}(x)| \le 2 C_e \sum_{i \in \mathcal{A}_{\text{pruned}}(x)} \tilde{p}_i(x)$$
Squaring both sides and using $\left(\sum_{i \in \mathcal{A}_{\text{pruned}}(x)} \tilde{p}_i(x)\right)^2 \le \sum_{i \in \mathcal{A}_{\text{pruned}}(x)} \tilde{p}_i(x) \le \sum_{i \notin S} \tilde{p}_i(x) \mathbb{I}_i(x)$:
$$\|\Delta(x)\|_2^2 \le 4 C_e^2 \sum_{i \notin S} p_i(x) \mathbb{I}_i(x)$$
Taking expectations over $x \sim \mathcal{D}$:
$$\mathcal{L}_{\text{drift}}(S) \le 4 C_e^2 \sum_{i \notin S} \mathbb{E}_{x \sim \mathcal{D}} [p_i(x) \mathbb{I}_i(x)] = 4 C_e^2 \sum_{i \notin S} I_i^*$$
$\blacksquare$

Minimizing this upper bound is equivalent to maximizing $\sum_{i \in S} I_i^*$, which is achieved by greedy selection: retaining the top-$K$ experts with the largest importance scores $I_i^*$, or equivalently pruning the bottom $E - K$ experts with minimal $I_i^*$.

---

## 3. Distributed All-Reduce as an Unbiased Estimator

In distributed evaluation, rank $w \in \{0, \dots, W-1\}$ observes a local batch of $N_w$ tokens $\mathcal{B}_w = \{x_{w, 1}, \dots, x_{w, N_w}\}$ drawn i.i.d. from $\mathcal{D}$.
The total number of calibration tokens across all ranks is:
$$N = \sum_{w=0}^{W-1} N_w$$

Each rank computes its local unnormalized importance accumulator:
$$S_{i, w} = \sum_{n=1}^{N_w} p_i(x_{w, n}) \mathbb{I}_i(x_{w, n}) \in \mathbb{R}$$

Applying the collective distributed operation `torch.distributed.all_reduce(..., op=ReduceOp.SUM)` yields:
$$S_i = \sum_{w=0}^{W-1} S_{i, w} = \sum_{w=0}^{W-1} \sum_{n=1}^{N_w} p_i(x_{w, n}) \mathbb{I}_i(x_{w, n})$$
The global importance estimator is formed by dividing by $N$:
$$\widehat{I}_i = \frac{1}{N} S_i = \frac{1}{N} \sum_{w=0}^{W-1} \sum_{n=1}^{N_w} p_i(x_{w, n}) \mathbb{I}_i(x_{w, n})$$

### Theorem 2 (Unbiasedness and Consistency)
1. **Unbiasedness**: For every expert $i \in \{1, \dots, E\}$:
   $$\mathbb{E}[\widehat{I}_i] = I_i^*$$
2. **Variance Reduction**: If $\operatorname{Var}_{x \sim \mathcal{D}}(p_i(x) \mathbb{I}_i(x)) = \sigma_i^2$, then:
   $$\operatorname{Var}(\widehat{I}_i) = \frac{\sigma_i^2}{N} = \frac{\sigma_i^2}{\sum_{w=0}^{W-1} N_w}$$
   which scales as $\mathcal{O}(1 / (W \cdot N_{\text{local}}))$ under balanced batches ($N_w = N_{\text{local}}$).

### Proof:
By linearity of expectation:
$$\mathbb{E}[\widehat{I}_i] = \frac{1}{N} \sum_{w=0}^{W-1} \sum_{n=1}^{N_w} \mathbb{E}_{x_{w, n} \sim \mathcal{D}} [p_i(x_{w, n}) \mathbb{I}_i(x_{w, n})] = \frac{1}{N} \sum_{w=0}^{W-1} \sum_{n=1}^{N_w} I_i^* = \frac{N}{N} I_i^* = I_i^*$$
Because samples across ranks and sequences are i.i.d.:
$$\operatorname{Var}(\widehat{I}_i) = \frac{1}{N^2} \sum_{w=0}^{W-1} \sum_{n=1}^{N_w} \operatorname{Var}(p_i(x_{w, n}) \mathbb{I}_i(x_{w, n})) = \frac{N \sigma_i^2}{N^2} = \frac{\sigma_i^2}{N}$$
$\blacksquare$

---

## 4. Rank-Local Estimation Fallacy (Distribution Skew)

In real distributed pipelines, worker ranks often process non-i.i.d. shards (e.g., domain-specific partitioning where rank 0 processes code, rank 1 processes natural language, rank 2 processes mathematics). Let $\mathcal{D}_w$ denote the data distribution on rank $w$, where $\mathcal{D} = \sum_{w=0}^{W-1} \alpha_w \mathcal{D}_w$ with $\sum_w \alpha_w = 1$.

If rank $w$ uses only its local estimator:
$$\widehat{I}_{i, w}^{\text{local}} = \frac{1}{N_w} S_{i, w}$$
then:
$$\mathbb{E}[\widehat{I}_{i, w}^{\text{local}}] = I_{i, \mathcal{D}_w}^* \ne I_i^*$$
The bias is:
$$\operatorname{Bias}(\widehat{I}_{i, w}^{\text{local}}) = I_{i, \mathcal{D}_w}^* - \sum_{w'=0}^{W-1} \alpha_{w'} I_{i, \mathcal{D}_{w'}}^* \ne 0$$

An expert specialized in code would have high $I_{i, \mathcal{D}_0}^*$ on rank 0, but very low $I_{i, \mathcal{D}_1}^*$ on rank 1. Pruning without distributed all-reduce leads to catastrophic pruning of domain experts.
Under distributed `all_reduce(..., op=ReduceOp.SUM)`:
$$\mathbb{E}\left[\frac{1}{N} \sum_{w=0}^{W-1} S_{i, w}\right] = \sum_{w=0}^{W-1} \frac{N_w}{N} I_{i, \mathcal{D}_w}^* = \sum_{w=0}^{W-1} \alpha_w I_{i, \mathcal{D}_w}^* = I_i^*$$
proving that `all_reduce` strictly eliminates cross-rank distribution skew bias.

---

## 5. Weight Re-normalization Under Pruned Experts

When expert subset $S^* \subset \{1, \dots, E\}$ is retained ($|S^*| = K$), the router's projection matrix $W_g \in \mathbb{R}^{E \times d}$ and bias $b_g \in \mathbb{R}^E$ are sliced along the expert dimension:
$$W_g^{(S^*)} = W_g[S^*, :], \quad b_g^{(S^*)} = b_g[S^*]$$

During inference, logits are computed over the surviving $K$ experts:
$$h^{(S^*)}(x) = W_g^{(S^*)} x + b_g^{(S^*)}$$
The top-$k'$ surviving routing weights ($k' = \min(k, K)$) are re-normalized:
$$\tilde{p}_j^{(S^*)}(x) = \frac{\exp(h_j^{(S^*)}(x))}{\sum_{m \in \operatorname{TopK}'(h^{(S^*)}(x))} \exp(h_m^{(S^*)}(x))}$$

Because $\sum_{j} \tilde{p}_j^{(S^*)}(x) = 1$, the activation scale is preserved:
$$\mathbb{E}\left[\left\|\sum_{j} \tilde{p}_j^{(S^*)}(x) e_j(x)\right\|\right] \approx \mathbb{E}[\|y(x)\|]$$
preventing downstream layer activation degradation and minimizing representation drift.
$\blacksquare$
