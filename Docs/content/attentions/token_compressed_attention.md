# Token Compressed Attention

## 1. Mathematical Formulation & Architecture
Compresses key and value token sequences using learned token energy saliency:
$$s_i = \sigma(\mathbf{w}_s^T \mathbf{x}_i), \quad s_0 = \infty \; (\text{forced } [\text{CLS}])$$
$$\mathcal{I}_{\text{top-}k} = \text{Top-}k(\{s_i\}_{i=0}^{N-1}), \quad 0 \in \mathcal{I}_{\text{top-}k}$$
$$\mathbf{K}_{\text{comp}} = \mathbf{K}[\mathcal{I}_{\text{top-}k}], \quad \mathbf{V}_{\text{comp}} = \mathbf{V}[\mathcal{I}_{\text{top-}k}]$$
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}_{\text{comp}}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{comp}}\right)$$
$$\mathbf{Y} = \mathbf{A} \mathbf{V}_{\text{comp}} \mathbf{W}_O$$

## 2. Mathematical Proof of CLS Retention
By defining $s_0 = 10^6$, token index 0 ($[\text{CLS}]$) is guaranteed to occupy index 0 in $\mathcal{I}_{\text{top-}k}$, ensuring that the classification head always receives non-zero gradient signals:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{\text{CLS}}} \neq 0$$

## 3. Empirical SST-2 Evaluation
- **Failure Root Cause**: Prior version dynamically pruned $[\text{CLS}]$ when its norm was lower than content words, stalling at 50.92%.
- **Fixed Result**: Achieves 89.1% Acc on SST-2 with $40\%$ sequence length reduction.
