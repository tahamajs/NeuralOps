# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Increases model capacity without proportional compute
- Enables expert specialization
- Relevant for multi-task learning

### Negative
- Parameter growth may not be acceptable
- Dual-use concerns
- Complexity increases deployment burden

## E.2 Environmental Impact

- **Total GPU-hours:** ~12.5
- **Estimated CO₂:** 2.4 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public)
- **PII:** None
- **Bias:** English-only

## E.4 Model Ethics

### Bias

Not audited. Experts may specialize by demographic features, potentially 
amplifying bias.

### Misuse

Standard concerns for any LLM deployment.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters

## E.11 Long-Term Considerations

MoE architecture trends suggest:
- Larger models with sparse activation
- Routing as a learning problem
- Expert specialization as a research direction

REAP-MoE contributes to this direction.

---