# Appendix A. Linear Kernel Formulation
Kernel feature map $\phi(x) = \operatorname{elu}(x) + 1$ guarantees non-negative attention weights.



## 3. Rigorous Theoretical Proofs & Stability Analysis

### 3.1 Proof of Strict Positivity and Non-Zero Denominators in Global Feature Maps

**Theorem 1 (Strict Positivity and Invertibility of RFF Normalizer)**:
*Let $\phi_Q(x) = \frac{1}{\sqrt{F}} \exp(\operatorname{clamp}(x \Omega, -\infty, c_{\max}))$ and $\phi_K(x) = \frac{1}{\sqrt{F}} \exp(\operatorname{clamp}(x \Omega, -\infty, c_{\max}))$ with $F \in \mathbb{N}^+$, $c_{\max} < \infty$, and normalizer epsilon $\epsilon_{\text{norm}} > 0$. Then for all inputs $Q, K$, the attention denominator satisfies $D_i \ge \epsilon_{\text{norm}} > 0$, strictly preventing division-by-zero singularities and negative normalizer values.*

*Proof*:
1. For any scalar $u \in \mathbb{R}$, the exponential function satisfies $\exp(u) > 0$.
2. Clamping ensures $-\infty < \operatorname{clamp}(u, -\infty, c_{\max}) \le c_{\max} < \infty$. Hence:
   $$\forall (b, h, i, f), \quad \phi_Q(Q)_{b, h, i, f} \ge \frac{1}{\sqrt{F}} \exp(-\infty) = 0 \quad \text{and} \quad \phi_Q(Q)_{b, h, i, f} \le \frac{\exp(c_{\max})}{\sqrt{F}} < \infty$$
   For finite input values $Q_{b, h, i}$, $\phi_Q(Q)_{b, h, i, f} \ge \frac{\exp(-M)}{\sqrt{F}} > 0$ where $M = \|\widehat{Q}_g\|_2 \|\Omega\|_F < \infty$.
3. For all valid (unmasked) key positions $j \in \{1, \dots, L\}$, $\phi_K(K)_{b, h, j, f} > 0$.
4. The key accumulation vector entries satisfy:
   $$(Z_{\text{sum}})_{b, h, f} = \sum_{j=1}^L \phi_K(K)_{b, h, j, f} \ge L_{\text{valid}} \cdot \min_{j} \phi_K(K)_{b, h, j, f} \ge 0$$
   with $(Z_{\text{sum}})_{b, h, f} > 0$ whenever at least one non-padding token exists ($L_{\text{valid}} \ge 1$).
5. The denominator evaluates to the Euclidean inner product:
   $$D_{b, h, i} = \sum_{f=1}^F \phi_Q(Q)_{b, h, i, f} (Z_{\text{sum}})_{b, h, f} + \epsilon_{\text{norm}}$$
   Since each summand is strictly non-negative ($\phi_Q > 0$ and $Z_{\text{sum}} \ge 0$):
   $$D_{b, h, i} \ge 0 + \epsilon_{\text{norm}} = \epsilon_{\text{norm}} > 0$$
6. Therefore, $D_{b, h, i}^{-1} \le \epsilon_{\text{norm}}^{-1} < \infty$, establishing strict numerical stability, non-singularity, and bounded output magnitude for arbitrary sequence lengths $L$. $\blacksquare$

### 3.2 Lyapunov Boundedness of Associative Matrix Memory Dynamics

**Theorem 2 (Lyapunov Bounded Memory Accumulation)**:
*Consider the sequential accumulation of the key-value associative state $S_t = \sum_{j=1}^t \phi_K(k_j)^\top v_j$. Under $L_2$ bounded inputs $\|v_j\|_2 \le M_v$ and clamped feature maps $\|\phi_K(k_j)\|_2 \le M_\phi$, the normalized readout $y_t = \frac{\phi_Q(q_t) S_t}{\phi_Q(q_t) \sum_{j=1}^t \phi_K(k_j)^\top + \epsilon}$ remains strictly bounded within the convex hull of the value sequence: $\|y_t\|_2 \le M_v$ for all sequence lengths $L = 1024$.*

*Proof*:
1. The unnormalized readout expands to:
   $$\phi_Q(q_t) S_t = \sum_{j=1}^t \left( \phi_Q(q_t) \phi_K(k_j)^\top \right) v_j = \sum_{j=1}^t \alpha_{t, j} v_j$$
   where $\alpha_{t, j} = \langle \phi_Q(q_t), \phi_K(k_j) \rangle = \sum_{f=1}^F \phi_Q(q_t)_f \phi_K(k_j)_f > 0$.
2. The denominator expands identically:
   $$D_t = \phi_Q(q_t) \sum_{j=1}^t \phi_K(k_j)^\top + \epsilon = \sum_{j=1}^t \alpha_{t, j} + \epsilon$$
3. Taking the normalized output norm:
   $$\|y_t\|_2 = \left\| \frac{\sum_{j=1}^t \alpha_{t, j} v_j}{\sum_{j=1}^t \alpha_{t, j} + \epsilon} \right\|_2 \le \frac{\sum_{j=1}^t \alpha_{t, j} \|v_j\|_2}{\sum_{j=1}^t \alpha_{t, j} + \epsilon} \le M_v \cdot \frac{\sum_{j=1}^t \alpha_{t, j}}{\sum_{j=1}^t \alpha_{t, j} + \epsilon} < M_v$$
4. Define the quadratic Lyapunov function $V(y_t) = \|y_t\|_2^2$. The Lyapunov function satisfies:
   $$V(y_t) < M_v^2 < \infty \quad \forall t \in [1, L]$$
   guaranteeing uniform stability and preventing state divergence across arbitrary context horizons $L \le 1024$. $\blacksquare$

---