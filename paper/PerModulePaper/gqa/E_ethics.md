<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Longer context:** Enables more capable language applications.
- **Cheaper inference:** Reduces compute cost per token.
- **Edge deployment:** Makes on-device LLMs feasible.

### Negative
- **Dual-use:** Long-context enables surveillance at scale.
- **Concentration:** Favors large labs with long-context models.
- **Environmental:** Cheaper inference may increase total usage (Jevons paradox).

## E.2 Environmental Impact

- **Total GPU-hours:** ~10
- **Estimated CO₂:** 0.7 kg CO₂eq

Compression methods generally reduce emissions per inference.

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English movie reviews)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias

We do not audit GQA for bias. Compression may affect head specialization 
and could amplify or suppress existing biases.

### Misuse

Could enable surveillance, misinformation. Not endorsed.

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
5. Respond to replication attempts

## E.11 Long-Term Considerations

Efficient inference could:

- Democratize LLM deployment
- Reduce AI carbon footprint
- Enable edge applications

But also:

- Enable long-context surveillance
- Concentrate AI capabilities
- Widen the digital divide

We encourage community discussion of these trade-offs.
