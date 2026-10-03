<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 The Long-Memory Branch is a Linear Projection

Despite the name `GatedDeltaNetLayer`, the implementation is a single 
`nn.Linear`. It has no recurrence, no state, no delta rule. **The 
"long-memory" claim is not supported by the implementation.**

## 10.2 No Long-Range Capability

Sliding-window attention is local. The linear projection is per-token. 
Together, the module has no mechanism for long-range dependencies.

## 10.3 Gate Mean-Pooling

The gate is per-head but mean-pooled to scalar. This loses head-specific 
information.

## 10.4 Preservation Gate Dominance

At α ≈ 0.88, the block contributes only 88% of its output. This is 
conservative but limits gains.

## 10.5 Parameter Overhead

The module adds ~5-6% parameters. Most of this (590K per layer) is the 
long-memory linear projection, which contributes nothing.

## 10.6 Slower than Baseline

~2.3 s× slowdown from extra kernel launches.

## 10.7 Task Coverage

Only SST-2. Behavior on tasks requiring long-range dependencies (QA, 
summarization) is untested.

## 10.8 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 untested.

## 10.9 Single Seed

No multi-seed runs.

## 10.10 Window Size Choice

The window size is a hyperparameter. Optimal value depends on task.

## 10.11 Code Quality

- `output_scale` redundancy in `LoRALinear`
- `GatedDeltaNetLayer` is misleadingly named
- No logging of gate statistics during training
- Some `try/except` blocks silently pass

## 10.12 What This Paper Is Not

- Not a compression method
- Not a long-context method
- Not a new recurrent mechanism
- Not a speedup method

---