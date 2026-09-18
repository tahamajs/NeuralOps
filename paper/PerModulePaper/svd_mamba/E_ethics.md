<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Advances hybrid attention-SSM research
- Multi-architecture support simplifies deployment
- SSM provides linear-time recurrence for long contexts

### Negative
- **Increases parameter count** (opposite of compression)
- Adds inference overhead
- Complex to deploy
- Dual-use for surveillance

## E.2 Environmental Impact

- **Training:** ~12.5 GPU-hours
- **CO₂:** ~12.4 kg CO₂eq
- **Inference:** 0.85 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English movie reviews

## E.4 Model Ethics

### Bias

The SSM branch may introduce bias through random initialization. We did 
not audit for bias.

### Misuse

Could enable more capable models for surveillance, misinformation.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document SSM initialization
5. Respond to replications

## E.11 Long-Term Considerations

Hybrid attention-SSM models like SVD-Mamba could:

- Enable more capable long-context models
- Provide new research directions

But also:

- Increase model complexity
- Make interpretability harder
- Add deployment burden

We encourage future work to carefully evaluate the trade-offs.

---