# 11. Conclusion

We introduced A3, an activation-aware compressed factorization method for transformer layers. By weighting the low-rank approximation with activation statistics and compensating for distribution shift with LayerNorm recalibration, A3 achieves meaningful model reduction while preserving task accuracy. The method is simple, production-friendly, and compatible with the broader low-rank recovery pipeline used in this project.
