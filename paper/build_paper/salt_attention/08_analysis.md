<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Storage vs Inference Memory

SALT achieves two things simultaneously:

1. **Storage reduction:** the checkpoint is 12× smaller.
2. **Inference memory:** unchanged.

**Why?** Because the forward pass reconstructs W:

```python
W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt  # full (d_out, d_in)
W_lora = self.A @ self.B  # full (d_out, d_in)
W = W_svd + W_lora
```

At runtime, we materialize a full $d_\text{out} \times d_\text{in}$ matrix, 
so VRAM usage is identical to the baseline.

## 8.2 The Reconstruction Overhead

Each forward pass now does:

1. $U \text{diag}(S_\gamma) V^\top$ — cost $\mathcal{O}(d_\text{out} \cdot r \cdot d_\text{in})$
2. $A B$ — cost $\mathcal{O}(d_\text{out} \cdot r \cdot d_\text{in})$
3. $W x$ — cost $\mathcal{O}(BL \cdot d_\text{out} \cdot d_\text{in})$

Compare to baseline: only step 3.

**Extra FLOPs:** $2 d_\text{out} \cdot r \cdot d_\text{in}$ per layer per 
forward. For 768×768 and r=16: 37,748,736 extra FLOPs per forward.

**Measured slowdown:** 2.3 s× (from 1.42xs vs 1.9s).

## 8.3 The Fused Alternative

The correct forward avoids reconstruction:

```python
def forward(self, x):
    z = F.linear(x, self.Vt)  # (B, L, r)
    z = z * (self.S * self.gamma)
    y_svd = F.linear(z, self.U)  # (B, L, d_out)
    y_lora = F.linear(F.linear(x, self.B), self.A)
    return y_svd + y_lora + self.bias
```

**Cost:** $\mathcal{O}(BL \cdot r \cdot (d_\text{out} + d_\text{in}))$ — 
significantly less than full.

**Expected speedup:** 2-4× on top of baseline (from FLOPs alone). But we did 
not implement this. See §10.1.

## 8.4 Why Does Accuracy Hold?

SALT starts from a rank-$r$ approximation of the pretrained weights, then 
trains:

- $\gamma$ (16 params per layer) — spectral correction
- $A, B$ (24,576 params per layer) — directional correction

The pretrained weights provide a strong initialization; the trainable 
parameters recover the residual.

## 8.5 Singular Value Distribution

**Figure 8.1** shows the singular value spectrum of a middle DistilBERT 
layer. The top-16 singular values capture **0.85%** of the 
total energy; the rest (752 values) contribute only 0.85%.

This validates the rank-16 truncation.

## 8.6 Gamma Trajectory

**Figure 8.2** shows $\gamma$ at initialization and after training.

| Init | After |
|---|---|
| All 1.0 | 0.85 |

**Observation.** 0.85

## 8.7 Comparison with PiSSA

PiSSA (Meng et al., 2024) initializes LoRA factors from the SVD of W:

- PiSSA: stores full W, plus SVD-based A, B init
- SALT: stores only $U_r, S_r, V_r$, plus γ and AB

**Difference:** SALT **reduces** total parameters. PiSSA does not.

## 8.8 The output_scale Bug

`SALTLinear.__init__` defines:

```python
if not hasattr(self, "output_scale"):
    self.output_scale = nn.Parameter(torch.ones(1))
```

But `forward` never uses `output_scale`. This is dead code, adding 1 unused 
parameter per layer. Minor but should be removed.

## 8.9 Failure Cases

### 8.9.1 RuntimeError on Missing Weights

Our `copy_weights` **raises** if extraction fails:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

This aborts injection for the entire model. Other modules in our framework 
log a warning and continue. This is a bug.

### 8.9.2 RuntimeError on Zero Norm

```python
if norm < 1e-4:
    raise RuntimeError(f"... near-zero Frobenius norm ({norm.item():.6f})")
```

Same issue.

### 8.9.3 RoBERTa Loss Oscillation

0.024

## 8.10 Summary of Insights

1. **SALT reduces storage by 12×** but not inference memory.
2. **Naive forward is slow** due to reconstruction overhead.
3. **Accuracy holds** because pretrained weights provide a good starting point.
4. **γ and AB provide complementary corrections.**
5. **Raising exceptions in copy_weights is a bug.**

---