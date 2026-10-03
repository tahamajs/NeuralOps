<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Long-context inference:** Enables more capable applications.
- **Cheaper inference:** Reduces compute cost per token.
- **Edge deployment:** Makes on-device LLMs more feasible.

### Negative
- **Dual-use:** Long-context enables surveillance at scale.
- **Concentration:** Favors large labs.
- **Environmental:** Efficiency gains may increase total usage.

## E.2 Environmental Impact

- **Total GPU-hours:** ~12
- **Estimated CO₂:** 0.85 kg CO₂eq

Primal Attention reduces compute at long $L$ but increases it at short $L$ 
(overhead of $F_1, F_2, F_3$).

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias

Primal Attention may amplify or suppress biases depending on which directions 
landmarks capture.

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

## E.11 Long-Term Considerations

Efficient attention could:

- Enable long-context AI on consumer hardware
- Reduce AI's carbon footprint
- Democratize access

But also:

- Enable surveillance at scale
- Concentrate AI capabilities

We encourage community discussion.

---
