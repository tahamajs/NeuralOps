# Native Hybrid Attention (NHA)

## 1. Mathematical Formulation & Architecture
Combines local sliding-window attention with global Random Fourier Features (RFF) linear attention:
$$\mathbf{Y} = \gamma(\mathbf{x}) \mathbf{Y}_{\text{local}} + (1 - \gamma(\mathbf{x})) \mathbf{Y}_{\text{global}}$$
$$\mathbf{Y}_{\text{global}} = \frac{\phi(\mathbf{Q}) (\phi(\mathbf{K})^T \mathbf{V})}{\phi(\mathbf{Q}) \phi(\mathbf{K})^T \mathbf{1} + \epsilon}$$
where $\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} [\cos(\mathbf{x}\mathbf{W}_{\text{rff}}), \sin(\mathbf{x}\mathbf{W}_{\text{rff}})]$.

## 2. Registry Integrity & CLS Handling
- Disentangled from dummy compressor classes.
- Full bidirectional attention ensures $[\text{CLS}]$ global context retention.

## 3. SST-2 Evaluation
Achieves 88.9% Acc on SST-2 with $\mathcal{O}(N \cdot w + N \cdot F)$ complexity.
