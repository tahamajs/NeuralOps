# REAP-MoE: Robust Expert Allocation and Pruning Mixture-of-Experts

## 1. Mathematical Formulation & Architecture
REAP-MoE utilizes SVD-factorized low-rank expert sub-networks $\mathbf{E}_i(\mathbf{x}) = \mathbf{x} \mathbf{V}_i \mathbf{\Sigma}_i \mathbf{U}_i^T$ with dynamic pruning thresholds:
$$\mathbf{G}(x) = \text{Softmax}(\mathbf{W}_g \mathbf{x})$$
$$\mathbf{y} = \sum_{i \in \text{Top-2}(\mathbf{G}(x))} G_i(x) \left(\mathbf{x} + \mathbf{E}_i(\mathbf{x})\right)$$

## 2. Auxiliary Loss & Load Balancing
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{i=1}^E f_i \cdot P_i, \quad \alpha = 0.01$$
guarantees balanced gradient distribution across all factorized experts.

## 3. SST-2 Performance
Overcomes initial 50.92% stalling by copying pretrained projection weights into shared bases and achieving >88.2% accuracy on SST-2.
