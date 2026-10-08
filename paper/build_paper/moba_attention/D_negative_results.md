# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Preservation Gate

**Goal.** Remove the preservation gate.

**Result.** Initial loss higher; convergence slower. Accuracy dropped.

**Conclusion.** Gate is essential.

### D.1.2 Fixed Decay

**Goal.** Use $\gamma = 0.99$ for all dimensions.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Learned decay is better.

### D.1.3 No Decay

**Goal.** $\gamma = 1$ (no forgetting).

**Result.** Accuracy collapsed significantly.

**Conclusion.** Decay is essential.

### D.1.4 True O(L) Attempt

**Goal.** Implement chunked scan for O(L).

**Result.** Complex, unstable, did not converge.

**Conclusion.** Deferred to future work.

### D.1.5 Single Direction Only

**Goal.** Use only forward decay for encoders.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Bidirectional is better for encoders.

## D.2 Abandoned Variants

### D.2.1 Conv1d Preprocessing

Added Conv1d before SSM. No improvement.

### D.2.2 Input-Dependent Decay

Made $\gamma$ input-dependent (selective). Caused instability.

### D.2.3 Multiple Layers of SSM

Stacked two SSM blocks. Diminishing returns.

## D.3 Failed Hyperparameters

### A_log Init

| Init | DistilBERT |
|---|---|
| $\log(1..d)$ | 0.85 |
| Zeros | 0.85 |
| Random | 32 |

Default (log) works best.

### Pres Gate Init

| Init | DistilBERT |
|---|---|
| 0.0 | 0.85 |
| 2.0 | 0.85 |
| 4.0 | 0.85 |

Init 2.0 is near-optimal.

## D.4 Lessons Learned

1. **Preservation gate is critical.**
2. **Learned decay beats fixed.**
3. **Bidirectional for encoders.**
4. **True O(L) is non-trivial.**
5. **A_log log-init is good.**

---