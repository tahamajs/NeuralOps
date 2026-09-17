<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. A3 may behave differently on:

- Generation tasks (summarization, translation)
- Multi-class classification (MNLI, AG News)
- Token-level tasks (NER, POS)
- Long-context tasks (>512 tokens)

## 10.2 Model Coverage

We test on models with ≤124M parameters. Not tested:

- Large models (>1B): SVD time grows quadratically
- Very deep models (>24 layers): compounding effect
- Mixture-of-Experts: not tested
- Vision transformers: not tested
- Multimodal: not tested

## 10.3 Hardware

All experiments on RTX 4070 Laptop (8 GB). Behavior on other hardware:

- A100/H100: SVD faster, benefit larger
- CPU: matmul-bound; A3 might help
- Edge: not tested

## 10.4 Calibration Dependency

A3 requires a calibration pass. If calibration data is unavailable:

- Fisher weighting degrades to unweighted SVD
- LayerNorm recalibration is skipped
- Accuracy may drop by up to 5%

This is a real limitation for models whose deployment environment differs 
from training data.

## 10.5 Inference Latency

Factorized layers are slower at inference:

- Two matmuls instead of one
- Intermediate rank-$r$ tensor must be materialized
- On GPU: ~1.3× slower
- On CPU: potentially faster

This is a **trade-off**: smaller size for slower latency.

## 10.6 Single-Seed Results

All main results are from one seed. With N=872 validation examples, SE is 
~1.69%. Differences below 3.4% (2×SE) may be noise.

## 10.7 No Theoretical Guarantee

We provide no formal bound on downstream accuracy loss. The Fisher proxy 
$\mathbb{E}[x^2]$ is not the true Fisher; our approximation error is untested.

## 10.8 Rank Selection

Our adaptive rank escalation increases rank until error is met. This is 
greedy and may not be optimal. A learned or Bayesian approach could be better.

## 10.9 LayerNorm Recalibration Heuristic

Our $\gamma \leftarrow \gamma \cdot \sigma_\text{target} / \sigma_\text{current}$ 
is a heuristic. It does not account for:

- Nonlinear interactions between layers
- Changes in higher moments
- Attention-specific distribution shifts

## 10.10 No Comparison with Some Baselines

We do not directly compare with:

- DistilBERT / TinyBERT (distillation)
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)

This limits the paper's external validity.

## 10.11 Failure Modes Not Fully Diagnosed

0.85

## 10.12 Code Quality

Our implementation is research-grade:

- No unit tests for edge cases
- Some error paths use broad exception handling
- Numerical stability could be improved in some paths

## 10.13 Ethical Considerations

Compression methods enable broader deployment. This is a **dual-use** 
capability. See Appendix E.
