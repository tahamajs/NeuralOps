<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Softmax attention scales as $\mathcal{O}(L^2)$ because it materializes the 
$L \times L$ attention matrix. For $L = 8192$, this is 64M entries — 
infeasible in memory. Linear attention approximates softmax by reordering 
the computation:

$$
\text{Attn}(Q_i, K, V) = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{1.1}
$$

The sums can be precomputed once, reducing cost to $\mathcal{O}(L d^2)$. The 
challenge is **choosing the feature map $\phi$** so the approximation is 
accurate.

## 1.2 The Performer's FAVOR+ Approach

Performer (Choromanski et al., 2021) proposes **FAVOR+** (**F**ast 
**A**ttention **V**ia **O**rthogonal **R**andom features). The kernel 
$\exp(q^\top k)$ is approximated by:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp\!\left(\omega^\top x - \frac{\|x\|^2}{2}\right)
\tag{1.2}
$$

where $\omega \in \mathbb{R}^{d \times F}$ is an **orthogonal** random 
matrix. FAVOR+ uses **positive** random features and provides theoretical 
bounds on the approximation error.

## 1.3 Our Contribution

Performer is a published method. Our contribution is a **production-hardened 
port** to pretrained transformers:

1. **Exact weight transfer** across four architecture families (flat, 
   nested, Conv1D, GQA).
2. **Two computational paths:** bidirectional (associative accumulation) 
   and causal (prefix-scan with chunk_size = 64).
3. **Numerical stabilization** via max-subtraction and norm-subtraction.
4. **LoRA recovery** for fine-tuning after injection.

## 1.4 Two Discrepancies Between Code and Documentation

**Discrepancy 1: `inject_performer` raises instead of returning None.**

The docstring says:

```python
"""... Safe injection returning None on failure."""
```

But the code:

```python
except Exception as e:
    logger.warning("PerformerAttention construction/weight copying failed: %s", e)
    raise RuntimeError(f"Injection failed")  # ← raises!
```

This is inconsistent with the docstring and could cause unexpected crashes 
in production.

**Discrepancy 2: `LoRALinear.output_scale` is a new parameter.**

```python
class LoRALinear(nn.Module):
    def __init__(self, ...):
        if not hasattr(self, 'output_scale'):
            self.output_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        ...
        return out * self.output_scale
```

This parameter is not present in the baseline checkpoint. On reload, it 
defaults to 1.0 (which is fine), but its presence causes `missing_keys` 
warnings and could cause issues with strict loading.

## 1.5 Contributions Summary

- **A production port of FAVOR+** to four pretrained transformer families.
- **Two computational paths** for bidirectional and causal attention.
- **An honest disclosure** of two code/documentation discrepancies.
- **Full reproducibility** including code and checkpoints.

## 1.6 Paper Organization

Section 2 reviews related work on kernel attention and RFF. Section 3 
provides background on softmax, kernel approximation, and FAVOR+. Section 4 
presents our implementation. Section 5 covers integration. Section 6 details 
the setup. Section 7 reports results. Section 8 analyzes the feature map 
and discrepancies. Section 9 discusses. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $F$ | Number of features | §1.2 |
| $\phi$ | Feature map | §1.1 |
| $\omega$ | Random feature matrix | §1.2 |
| $r_L$ | LoRA rank | §1.3 |
| $C$ | Chunk size | §1.3 |

---