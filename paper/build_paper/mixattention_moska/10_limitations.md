<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Causal Support

FASA **does not prune** for causal decoders. GPT-2 receives no benefit 
from the method.

## 10.2 Variance is a Weak Signal

The Spearman correlation between variance and attention received is 
0.85, indicating a **modest** relationship. FASA's pruning is 
better than random but worse than H2O.

## 10.3 Keep Ratio of 0.5 is Aggressive

At 0.5, FASA drops half the tokens. Accuracy drops by **0.85%** on 
DistilBERT. Higher keep ratios (0.75+) preserve more.

## 10.4 CLS Pinning is Essential

Without protecting index 0, encoders collapse. This is a fragile design.

## 10.5 Gather + TopK Overhead

The gather and top-k operations add overhead. At $L = 128$, the overhead 
exceeds the FLOP savings.

## 10.6 No Speedup at Short Sequences

FASA shows slower inference than baseline on DistilBERT. Speedup requires 
$L \gg 1024$.

## 10.7 Naming Confusion

The class is registered as both `fasa` and `mixattention_moska`. The 
"MoSKA" name is not explained in the code.

## 10.8 Pruning Recomputed Every Forward

The token subset is recomputed every forward pass. This is expensive 
relative to cached pruning (e.g., SnapKV).

## 10.9 No Multi-Seed

Single seed.

## 10.10 Task Coverage

Only SST-2. Long-context tasks (where FASA should help) are not tested.

## 10.11 Model Coverage

Only three models. Qwen3.5 untested.

## 10.12 No Learned Importance

The importance signal is purely heuristic. A small learned model could 
improve accuracy.

## 10.13 Code Quality

- `mixattention_moska` name is unexplained.
- No logging of pruning statistics.
- `importance_method` supports two options but only `var` is well-tested.

## 10.14 What This Paper Is Not

- Not a new attention mechanism
- Not a compression method
- Not a causal-decoder method
- Not a speedup method for short sequences

---