<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Democratizes AI:** Smaller models → broader access.
- **Reduces carbon:** Less compute per inference.
- **Enables edge AI:** Private, offline inference.

### Negative
- **Dual-use:** Efficiency benefits malicious and beneficial applications.
- **Concentration:** Large labs may benefit more.
- **Rebound effect:** Cheaper inference → more usage.

## E.2 Environmental Impact

- **Total GPU-hours:** ~15
- **Estimated CO₂:** 1.0 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English-only, movie reviews

## E.4 Model Ethics

### Bias

We did not audit AASVD for bias. Compression may amplify or suppress biases 
depending on which directions are preserved.

### Misuse

Could enable surveillance, misinformation, censorship. Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters
5. Respond to replications

## E.11 Long-Term Considerations

Compression could:

- Enable AI on billions of devices
- Reduce AI's carbon footprint
- Democratize access

But also:

- Enable surveillance at scale
- Concentrate AI capabilities
- Widen gap between compressed and full models
