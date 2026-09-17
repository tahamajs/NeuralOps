# 3. Mathematical Background
For query $Q$ and key $K$ with sequence length $L$, the raw score matrix is $S = \frac{Q K^\top}{\sqrt{d_h}} \in \mathbb{R}^{L \times L}$. Dense softmax normalizes over all $L$ positions, assigning non-zero probability mass to noisy or irrelevant tokens.
