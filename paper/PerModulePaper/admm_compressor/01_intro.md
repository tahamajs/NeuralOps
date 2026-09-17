# 1. Introduction
Combining pruning and quantization via naive heuristics causes severe error accumulation. `admm_compressor` decomposes the joint optimization problem into alternating subproblems: continuous loss minimization, projection onto structured sparsity manifolds, and discrete codebook quantization, coupled via augmented Lagrangian multipliers.
