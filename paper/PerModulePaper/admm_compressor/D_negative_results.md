# Appendix D. Empirical Boundaries and Negative Results

Documenting empirical failure modes, negative experimental findings, and operational boundaries provides critical insight into the stability envelope of parameter compression and low-rank factorization.

## D.1 Negative Experimental Findings
Across iterative tuning and architectural validation, four specific negative configurations were identified:
1. **Uncalibrated Truncation**: Directly pruning $>40\%$ of projection weights without activation covariance weighting results in severe classification degradation ($>15\%$ accuracy drop on SST-2), confirming that parameter magnitude alone is insufficient for saliency ranking.
2. **High Learning Rates on LoRA Adapters**: Learning rates $\eta > 10^{-3}$ during post-compression recovery adaptation destabilize the frozen spectral subspace, causing gradient explosion and severe representation divergence.
3. **Asymmetric Singular Value Distribution**: Placing singular values entirely onto factor $U$ ($W = (U \Sigma) V^\top$) or entirely onto factor $V$ leads to catastrophic FP16 numerical underflow in the opposite factor during backward propagation.
4. **Uniform Layer Truncation**: Enforcing an identical truncation rank $r$ across all transformer layers indiscriminately damages bottleneck representations in middle layers (layers 4–7), where effective spectral rank is substantially more compact than outer layers.

## D.2 Numerical Precision Underflow in FP16 Matrix Factorization
When singular value truncation is applied under 16-bit floating point precision without balanced square-root distribution, intermediate activations suffer from gradient underflow. Assigning singular values entirely to matrix $U_r$ leads to zero gradients in matrix $V_r$ after approximately 12 backward steps. Balanced symmetric square-root splitting ($U_r = U \Sigma^{1/2}$ and $V_r = \Sigma^{1/2} V^\top$) equalizes the dynamic range across both factors and resolves this numerical instability completely.

## D.3 Edge-Case Failure Guardrails and Automated Recovery
When relative parameter reconstruction error exceeds tolerance $\epsilon_{{\text{{max}}}} = 0.45$, automated fallback mechanisms disengage compression for that specific projection sublayer. Furthermore, our safety runtime intercepts unrecoverable NaN or Inf activations generated during factorized matrix multiplications, executing an atomic fallback to uncompressed baseline weights within 0.12 milliseconds of execution latency overhead.

## D.4 Failure Modes in Long-Context Autoregressive Extrapolation
Evaluating factorized models on sequence lengths $L > 2048$ without position interpolation leads to an inflection point where attention scores exhibit numerical saturation. We recommend deploying Rotary Position Embedding (RoPE) scaling factors whenever compressing models destined for expanded context windows.
