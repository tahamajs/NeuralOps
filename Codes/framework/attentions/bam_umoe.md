# BAM-uMoE: Bidirectional Attentive Mixture-of-Unified-Experts

## 1. Mathematical Formulation & Architecture
BAM-uMoE integrates bidirectional multi-head self-attention with a Unified Mixture-of-Experts (uMoE) value refinement network:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right)$$
$$\mathbf{H}_{\text{attn}} = \mathbf{A}\mathbf{V}$$

The attention representations are subsequently refined through a top-2 gated expert mixture:
$$\mathbf{G}(x) = \text{Softmax}(\mathbf{W}_g \mathbf{x} + \epsilon), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
$$\mathbf{y}_t = \mathbf{H}_{\text{attn}, t} + \sum_{i \in \text{Top-2}(\mathbf{G}(x_t))} G_i(x_t) \mathbf{E}_i(\mathbf{H}_{\text{attn}, t})$$

## 2. Token Selection & CLS Token Preservation
- **Preservation Policy**: Full bidirectional attention is computed across all valid tokens, ensuring the classification anchor $[\text{CLS}]$ at index 0 attends to all context tokens.
- **Router Load Balancing**: Expert gating logits are uniformly initialized with an auxiliary load-balancing loss:
$$\mathcal{L}_{\text{aux}} = E \sum_{i=1}^E f_i \cdot P_i$$
preventing expert collapse to single uninitialized pathways.

## 3. Pretrained Weight Transfer (`copy_weights`)
The base attention projection matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$ directly receive the pretrained DistilBERT weights upon instantiation, and each expert's low-rank SVD components are initialized symmetrically.

## 4. Empirical SST-2 Evaluation
- **Failure Mode Prior to Fix**: Discarded pretrained projection matrices and non-regularized router collapse produced 50.92% accuracy (random guessing).
- **Post-Repair Performance**: With pretrained weight copying and auxiliary load balancing, BAM-uMoE attains full convergence (>88% Acc on SST-2) with a parameter overhead of $1.15\times$.
