# 10. Limitations

## 10.1 Parameter Count

ExpertChoiceMoEAttention **increases** parameter count by approximately 
**48.5\%%** per layer. This is the primary limitation and we are 
explicit about it.

## 10.2 Python Loop Over Experts

The forward pass uses a Python loop over experts:

```python
for e in range(self.n_experts):
    ...
```

This is slow at high expert counts. A batched implementation (or Triton 
kernel) would be faster.

## 10.3 Modest Accuracy Gain

The MoE block improves accuracy by a small amount over the baseline. This 
does not justify the parameter cost.

## 10.4 Task Coverage

Only SST-2 is evaluated. Generation, translation, and summarization are 
untested.

## 10.5 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.6 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.7 No Auxiliary Loss Analysis

We do not use an auxiliary loss (unlike token-choice MoE). We do not 
compare with auxiliary-loss-based methods in detail.

## 10.8 Capacity Factor Choice

$CF = 1.25$ is a fixed default. Optimal CF is 
model-dependent.

## 10.9 Cross-Attention Not Fully Tested

We implement cross-attention but only test on T5 preliminarily.

## 10.10 Dynamic Router

We use a dynamic router for cases where input dimension differs from 
`d_model`. This is a fallback mechanism, not a well-designed feature.

## 10.11 No Theoretical Bound

We provide no formal approximation bound.

## 10.12 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.13 Code Quality

Research-grade:
- Broad exception handling
- Python loop over experts
- No unit tests for edge cases

## 10.14 Ethical Considerations

MoE scaling is dual-use. See Appendix E.

---