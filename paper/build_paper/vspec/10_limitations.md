# 10. Limitations

## 10.1 Attention-Only Compression

Only Q/K/V/O projections are factorized. The FFN is untouched. Total 
model reduction is modest.

## 10.2 Attention Still Quadratic

Token routing reduces the effective length but attention is still 
$\mathcal{O}(k^2)$. For very long L, this is still expensive.

## 10.3 Task Coverage

Only SST-2 is evaluated. Untested on:
- Generation
- Long-context
- Token-level tasks
- Multilingual

## 10.4 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.5 Hardware

Only RTX 4070.

## 10.6 Variational Training Overhead

Sampling adds a small overhead (~10%). At inference, this is removed.

## 10.7 Gate Saturation Risk

At high LR, gates may saturate. We mitigate with bias init but do not 
provide a formal guarantee.

## 10.8 Cross-Attention

Not supported.

## 10.9 Single-Seed

Single-seed results.

## 10.10 No Theoretical Bound

We provide no bound on accuracy after compression.

## 10.11 Nested Encoder Routing

For RoBERTa/BERT, we skip token routing. This limits compression for 
nested encoders.

## 10.12 Code Quality

Research-grade.

## 10.13 Ethical Considerations

See Appendix E.

---