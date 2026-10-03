# 10. Limitations

## 10.1 Causal Fallback

For causal decoders (GPT-2), NEXUS uses dense attention. **No sparsity 
benefit.**

## 10.2 Stripe Inactive

The stripe component requires $N > 256$. Our SST-2 experiments use 
$N = 128$, so stripe is inactive.

## 10.3 RFF Inactive

RFF rectification requires `enable_spectraformer=True` when pretrained 
weights are provided. Default is `False`, so RFF is inactive.

## 10.4 Softpick Risky

Softpick can collapse to uniform distribution when all logits are negative.

## 10.5 Copy_weights Raises

`copy_weights` raises exceptions on missing weights. May abort training.

## 10.6 Task Coverage

Only SST-2 evaluated.

## 10.7 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.8 Single Seed

SE ~1.7%.

## 10.9 Six Components, Four Active

Only 4 of 6 claimed components are active in our setup. The paper 
explicitly lists this.

## 10.10 No Comparison with Individual Methods

We do not compare with isolated XAttention, SeerAttention, etc.

## 10.11 Complexity for Encoder

Sub-quadratic **only when** $k B_s \ll N$.

## 10.12 Code Quality

- `raise` in copy_weights
- No fused kernel
- Some logging noise

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---