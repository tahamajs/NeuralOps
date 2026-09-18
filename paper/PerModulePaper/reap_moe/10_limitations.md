# 10. Limitations

## 10.1 Task Coverage

Only SST-2 (classification) is evaluated. REAP-MoE may behave differently 
on generation, retrieval, or multilingual tasks.

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |

## 10.3 Hardware

All experiments on RTX 4070 Laptop.

## 10.4 Parameter Growth

**REAP-MoE increases parameter count by ~66.4M%.** This is a 
fundamental limitation of MoE architectures. SVD mitigates but does not 
eliminate the growth.

## 10.5 FFN Duplication

Because the framework replaces only the attention module, the original FFN 
still runs. This doubles the effective depth of the FFN stage. Future work 
should remove the FFN entirely.

## 10.6 Inference Overhead

The router adds overhead (~10-20%). For latency-critical applications, this 
may be unacceptable.

## 10.7 Expert Specialization

We do not analyze expert specialization in detail. It is unclear whether 
experts learn meaningful distinctions or merely distribute randomly.

## 10.8 No Formal MoE Bound

We provide no formal bound on MoE approximation error.

## 10.9 Single-Seed Results

All main results from a single seed.

## 10.10 Failure Modes Not Fully Diagnosed

0.85

## 10.11 Code Quality

Research-grade:
- Broad exception handling
- No unit tests
- Numerical stability could improve

## 10.12 Ethical Considerations

See Appendix E.

---