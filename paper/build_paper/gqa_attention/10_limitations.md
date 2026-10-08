<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 evaluated. GQA may behave differently on:
- Generation tasks (summarization, translation)
- Long-context tasks (retrieval, QA)
- Multi-lingual tasks

## 10.2 Model Coverage

Tested on models with $H = 8$–$12$. Not tested on:
- Very wide models ($H = 32+$)
- Models with $H$ prime (limits $G$ choices)
- Mixture-of-Experts

## 10.3 Hardware Coverage

RTX 4070 Laptop only. Not tested on:
- A100/H100
- CPU
- Edge devices

## 10.4 No Long-Context Evaluation

We test with $L = 128$. Real GQA benefit is at $L = 4096+$. We do not 
measure accuracy at long contexts.

## 10.5 Quality Loss Not Analyzed Per-Head

We do not report which heads benefit most from grouping. A per-head analysis 
would reveal whether some heads should not be grouped.

## 10.6 Only Uniform Grouping

We use uniform grouping (each group same size). Non-uniform grouping 
(e.g., grouping similar heads) is not explored.

## 10.7 No Theoretical Bound

We provide no formal bound on expressiveness loss or task accuracy loss.

## 10.8 Single Seed

All results from seed 42. Differences below 3.4% may be noise.

## 10.9 Weight Averaging Choice

We average KV weights within a group. Other strategies (e.g., pick first, 
learned pooling) may work better but are not explored.

## 10.10 Comparison Limited

We do not compare with:
- Ainslie et al.'s original GQA
- MLA (DeepSeek-V2)
- Other recent variants

## 10.11 Failure Modes Not Fully Diagnosed

0.85

## 10.12 Code Quality

Research-grade implementation:
- No unit tests for edge cases
- Some paths use broad exception handling

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.
