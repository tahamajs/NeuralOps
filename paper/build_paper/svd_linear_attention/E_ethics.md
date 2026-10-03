<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Real compression AND speedup
- Streaming-ready
- Multi-architecture support
- Reduced carbon footprint for inference

### Negative
- Approximation error may affect fairness
- Dual-use
- Concentration risk

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

The linear attention approximation may distort attention patterns in ways 
that disproportionately affect minority inputs. We did not audit for bias.

### Misuse

Efficient long-context models could enable:

- Mass surveillance
- Automated misinformation
- Censorship at scale

We do not endorse these uses.

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
4. Document the decay initialization issue
5. Respond to replications

## E.11 Long-Term Considerations

SVD-Linear Attention is a step toward efficient, deployable transformers. 
It is one of the few modules that achieves both real compression and real 
speedup. We hope it will be used for beneficial applications such as:

- On-device AI
- Privacy-preserving inference
- Green computing

But we acknowledge the dual-use potential and encourage responsible 
deployment.

---