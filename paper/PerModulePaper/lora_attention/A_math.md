# Appendix A. Mathematical Derivations

## A.1 LoRA Decomposition

For a pretrained weight $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

$$
W = W_0 + \Delta W, \quad \Delta W = \frac{\alpha}{r_L} B A
\tag{A.1}
$$

where:
- $B \in \mathbb{R}^{d_\text{out} \times r_L}$
- $A \in \mathbb{R}^{r_L \times d_\text{in}}$

## A.2 Zero Initialization

At initialization:
- $A \sim \text{Kaiming}$
- $B = 0$

Then:
$$
\Delta W = \frac{\alpha}{r_L} \cdot 0 \cdot A = 0
\tag{A.2}
$$

So $W = W_0$ at step 0.

## A.3 Parameter Count

Original weight: $d_\text{out} \cdot d_\text{in}$

LoRA parameters: $r_L \cdot d_\text{in} + d_\text{out} \cdot r_L = r_L (d_\text{in} + d_\text{out})$

**Reduction ratio:**
$$
\frac{d_\text{out} \cdot d_\text{in}}{r_L (d_\text{in} + d_\text{out})}
\tag{A.3}
$$

For $d_\text{out} = d_\text{in} = 768$, $r_L = 16$:
$$
\frac{768^2}{16 \cdot 1536} = \frac{589,824}{24,576} = 24
\tag{A.4}
$$

## A.4 Attention Projections Total

For all four projections (with $H_{kv} = H$):

$$
\text{Trainable}_\text{LoRA} = r_L (2d + 2 H d_h)
\tag{A.5}
$$

For $d = 768$, $H = 12$, $d_h = 64$:
$$
= 16 \cdot (1536 + 1536) = 49,152
\tag{A.6}
$$

## A.5 Mergeability

After training, define:
$$
W_\text{merged} = W_0 + \frac{\alpha}{r_L} B A
\tag{A.7}
$$

Then:
$$
y = W_\text{merged} x = W_0 x + \frac{\alpha}{r_L} B A x
\tag{A.8}
$$

which matches the LoRA forward pass exactly.

## A.6 Memory Savings

Full fine-tuning requires (per parameter):
- Gradient: 2 bytes (FP16)
- AdamW $m$: 4 bytes (FP32)
- AdamW $v$: 4 bytes (FP32)
- Total: 10 bytes

LoRA requires:
- Gradient on $A, B$: $r_L(d_\text{in} + d_\text{out}) \times 2$ bytes
- AdamW state on $A, B$: $r_L(d_\text{in} + d_\text{out}) \times 8$ bytes

**Reduction ratio:** $10 \cdot d_\text{out} \cdot d_\text{in} / (10 \cdot r_L (d_\text{in} + d_\text{out})) = 24$ (same as parameter reduction).

## A.7 Reconstruction Error

LoRA can represent any $\Delta W$ of rank $\leq r_L$:

$$
\text{rank}(\Delta W) = \text{rank}(BA) \leq \min(r_L, \text{rank}(B), \text{rank}(A)) \leq r_L
\tag{A.9}
$$

So LoRA is **exact** for updates of rank $\leq r_L$.

## A.8 Nested Encoder Correctness

**Claim.** For nested encoders, skipping out_proj yields correct attention.

**Proof.** In the original architecture:
$$
\text{Out} = \text{BertSelfOutput.dense}(\text{Attention}(Q, K, V))
\tag{A.10}
$$

Our module returns raw attention; the outer layer applies `dense`. This 
matches the original flow. $\square$

---