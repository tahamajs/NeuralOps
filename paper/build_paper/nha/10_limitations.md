<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Parameter Reduction

NHA adds gate parameters ($d + 1$) and RFF buffer ($d_h m$). Parameter 
count is essentially unchanged.

## 10.2 Task Coverage

Only SST-2 evaluated at $L = 128$. Long-context tasks ($L > 512$) not tested.

## 10.3 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Very wide models ($H > 32$)
- Mixture-of-Experts
- Vision transformers

## 10.4 Fixed Window Size

The window size $w$ is a hyperparameter, not learned.

## 10.5 Fixed RFF Count

$m$ is a hyperparameter. Optimal $m$ might vary per head.

## 10.6 Chunk Boundary Effects

For small chunk sizes (64), chunk boundary effects may slightly degrade 
quality.

## 10.7 No Speedup at Short $L$

At $L = 128$, NHA is 2.1\times× slower than baseline due to dual 
streams.

## 10.8 Single Seed

All results from seed 42.

## 10.9 No Comparison with Some Baselines

We do not compare with:
- Performer
- Linformer
- Longformer

## 10.10 Cumsum Memory

The RFF cumulative state $S \in \mathbb{R}^{B \times H \times m \times d_h}$ 
requires $\mathcal{O}(B H m d_h)$ memory. For $B=8$, $H=12$, $m=128$, 
$d_h=64$: 786 KB per state.

## 10.11 Failure Modes

0.85

## 10.12 Ethical Considerations

Dual-use. See Appendix E.

---