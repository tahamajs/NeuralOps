# Appendix A. Mathematical Derivations

## A.1 Polynomial Kernel Definition

$$
K_\text{PADRe}(q, k) = \left(1 + \frac{q \cdot k}{d \sqrt{d_h}}\right)^d
\tag{A.1}
$$

## A.2 Taylor Expansion

$$
(1 + x)^d = \sum_{n=0}^{d} \binom{d}{n} x^n
\tag{A.2}
$$

For $d = 2$:

$$
(1 + x)^2 = 1 + 2x + x^2
\tag{A.3}
$$

## A.3 Feature Map (Order 2)

For $\phi(x) = (1, \sqrt{2} x, x \otimes x)$:

$$
\phi(q) \cdot \phi(k) = 1 + 2 q \cdot k + (q \cdot k)^2
\tag{A.4}
$$

Dimension: $1 + d_h + d_h^2$.

## A.4 Centering

$$
m_i = \max_j s_{ij}, \quad s_{ij}' = (s_{ij} - m_i)/T
\tag{A.5}
$$

Property: $s_{ij}' \leq 0$.

## A.5 Base Bound

$$
\text{base}_{ij} = \text{ReLU}(1 + s_{ij}'/d)
\tag{A.6}
$$

Property: $0 \leq \text{base}_{ij} \leq 1$.

## A.6 Attention Normalization

$$
\text{attn}_{ij} = \frac{\text{base}_{ij}^d \cdot \mathbb{1}[\text{valid}_{ij}]}{\sum_{j'} \text{base}_{ij'}^d \cdot \mathbb{1}[\text{valid}_{ij'}]}
\tag{A.7}
$$

## A.7 Approximation Error

For $|s| \ll d$:

$$
\left(1 + \frac{s}{d}\right)^d = e^s \cdot \left(1 - \frac{s^2}{2d} + \mathcal{O}(s^3/d^2)\right)
\tag{A.8}
$$

The relative error grows as $s^2/d$.

## A.8 Complexity (Unkernelized)

Scores: $\mathcal{O}(N^2 d_h)$.
Polynomial: $\mathcal{O}(N^2)$.
Total: $\mathcal{O}(N^2 d_h)$.

## A.9 Complexity (Kernelized, Future)

For feature dimension $F$:

$$
\text{Context}_i = \frac{\phi(q_i) \cdot \sum_j \phi(k_j) v_j^\top}{\phi(q_i) \cdot \sum_j \phi(k_j)}
\tag{A.9}
$$

Complexity: $\mathcal{O}(N F d_h)$.

---