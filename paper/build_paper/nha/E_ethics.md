<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Linear-time inference:** Enables long-context on consumer hardware.
- **Interpretability:** Gate analysis reveals local vs global structure.
- **Cheaper inference:** Reduces compute per token.

### Negative
- **Dual-use:** Enables surveillance at scale.
- **Concentration:** Favors large labs.
- **Rebound effect:** Cheaper inference may increase usage.

## E.2 Environmental Impact

- **Total GPU-hours:** ~12.5
- **Estimated CO₂:** 2.4 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias
Not audited. Gates may correlate with content features.

### Misuse
Not endorsed.

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
4. Document all hyperparameters

## E.11 Long-Term Considerations

Linear-time attention could:

- Enable long-context inference on any hardware
- Reduce AI's carbon footprint
- Democratize access

But also:

- Enable surveillance at scale
- Concentrate AI capabilities

We encourage community discussion.

---