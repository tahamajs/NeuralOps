# 10. Limitations

## 10.1 Parameter Count

TCA **increases** parameter count by ~6.3% due to LoRA + score_net. This is 
explicitly stated and is the primary limitation.

## 10.2 Task Coverage

Only SST-2 is evaluated. Behavior on:
- Generation tasks: unknown
- Multi-class classification: unknown
- Token-level tasks (NER): unknown
- Long-context tasks (L > 512): unknown

## 10.3 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.4 Hardware

All experiments on RTX 4070 (8 GB). Behavior on other hardware untested.

## 10.5 Inference Latency

The score_net adds a small forward pass. For very small L, this overhead 
may dominate. We do not characterize the crossover point.

## 10.6 No Theoretical Guarantee

We provide no bound on downstream accuracy loss. The selection is learned, 
not provably optimal.

## 10.7 Training Dependence

The score_net must be trained. If you cannot run recovery fine-tuning, TCA 
cannot be applied.

## 10.8 Single-Seed Results

Single-seed results. SE ~1.7%. Differences below 3.4% may be noise.

## 10.9 No Comparison with All Baselines

We do not directly compare with:
- H2O (would need their exact implementation)
- StreamingLLM (different setup)
- SnapKV (different problem)
- KV quantization methods

## 10.10 Causal Mask Complexity

Our causal mask construction is O(L·k). This may become non-negligible 
for very large L. A fused kernel would be better.

## 10.11 Failure Modes Not Diagnosed

None

## 10.12 Code Quality

Research-grade implementation:
- Broad exception handling in copy_weights
- No fused kernel for score_net
- No unit tests for edge cases

## 10.13 Ethical Considerations

KV compression is dual-use. See Appendix E.

---