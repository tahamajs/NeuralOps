<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Hybrid attention architectures combine multiple attention mechanisms with 
the goal of capturing both short-range and long-range dependencies 
efficiently. A common pattern is:

$$
\text{Output} = g \cdot \text{Local}(X) + (1 - g) \cdot \text{Global}(X)
\tag{1.1}
$$

where $g$ is a mixing coefficient. The rationale: local attention captures 
fine-grained context at low cost, while global attention captures long-range 
semantics.

The **challenge** is twofold:

1. **Mixing the two branches correctly.** A fixed $g$ (e.g., $0.5$) ignores 
   that different tokens have different local/global needs.
2. **Preserving pretrained behavior.** Replacing attention with a new 
   mechanism risks breaking the residual stream that downstream layers 
   depend on.

## 1.2 Our Approach

**GoldFinch-AHN** addresses both:

1. **Input-dependent gating.** Per-token gate $g_t = \sigma(W_g x_t)$ 
   determines how much each token relies on local vs. global.
2. **Preservation gate.** A block-level residual with learnable coefficient:
   $$
   Y = \alpha \cdot \text{Block}(X) + (1 - \alpha) \cdot X
   \tag{1.2}
   $$
   with $\alpha = \sigma(2.0) \approx 0.88$ at initialization.

## 1.3 Honest Disclosures

Two implementation details must be disclosed before the methodology:

**The "long-memory" branch is a linear projection.** In the code, the class 
`GatedDeltaNetLayer` is:

```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model, use_spectral_norm=False):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.net(x)
```

There is no recurrence, no state update, no delta rule. It is a single 
`nn.Linear`. In our experiments this branch is essentially a 
content-independent projection.

**The preservation gate is a residual connection.** The `output_gate` 
parameter blends the block's output with the block's input — a learnable 
residual. It is not an attention mechanism.

We include these disclosures to enable fair evaluation: GoldFinch-AHN 
should be compared against other attention-replacement methods with the 
understanding that it is more conservative than they are.

## 1.4 Contributions

- **A hybrid attention block** with sliding-window attention, an input-dependent 
  gate, and a preservation gate.
- **An empirical evaluation** on DistilBERT, RoBERTa, and GPT-2 on SST-2.
- **Honest implementation disclosure**: the long-memory branch is a linear 
  projection, not a recurrent mechanism.
- **Full reproducibility**, including code and checkpoints.

## 1.5 Paper Organization

Section 2 reviews hybrid attention, sliding-window mechanisms, and residual 
gating. Section 3 provides background on attention, gating, and residuals. 
Section 4 presents GoldFinch-AHN. Section 5 covers integration. Section 6 
details the setup. Section 7 reports results. Section 8 analyzes the 
mechanism. Section 9 discusses. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $X$ | Input hidden states | §1.1 |
| $g$ | Mixing gate | §1.1 |
| $\alpha$ | Preservation coefficient | §1.2 |
| $W$ | Window size | §3.2 |
| $d$ | Hidden size | §3.1 |
| $H$ | Number of heads | §3.1 |
| $r$ | LoRA rank | §4.4 |
</parameter>

---