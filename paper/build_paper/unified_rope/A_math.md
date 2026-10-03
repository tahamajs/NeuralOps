<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 RoPE Norm Preservation

**Proposition.** The rotation operator preserves L2 norm.

**Proof.** Rotation matrices $R(\theta)$ are orthogonal: $R R^\top = I$. 
Therefore $\|R x\|_2 = \|x\|_2$. $\square$

## A.2 RoPE Relative Position

**Proposition.** For $q_i$ at position $i$ and $k_j$ at position $j$:

$$
(R_i q_i)^\top (R_j k_j) = f(q_i, k_j, i - j)
\tag{A.1}
$$

**Proof.** Classical (Su et al., 2021). The rotation angle difference 
$R_i^\top R_j = R_{j-i}$ depends only on $j - i$. $\square$

## A.3 Partial RoPE

For rotary dimension $r < d_h$:

$$
\tilde{q}_i = \begin{pmatrix} R_i^{(r)} q_i^{[:r]} \\ q_i^{[r:]} \end{pmatrix}
\tag{A.2}
$$

The pass-through dimension $q_i^{[r:]}$ is unchanged.

## A.4 GQA Expansion Correctness

**Claim.** GQA with `repeat_kv` produces the same output as if KV heads 
were explicitly duplicated.

**Proof.** `repeat_kv` duplicates heads. The attention output for each 
query head uses its corresponding KV head. $\square$

## A.5 Precomputed Cache Consistency

**Claim.** The cached `cos_cached`, `sin_cached` tables match recomputation 
up to numerical precision.

**Proof.** The cache is computed once via `einsum` and stored. Lookup is 
exact. $\square$

## A.6 Nested Encoder Output

**Claim.** For a nested encoder, the outer `output.dense` applies the 
output projection; the inner attention returns raw context.

**Proof.** By construction of `BertAttention`. $\square$

## A.7 Parameter Count Unchanged

**Claim.** UnifiedRoPE does not change parameter count.

**Proof.** The `LoRALinear` projections have the same parameter shapes as 
the original `nn.Linear` projections. $\square$

---