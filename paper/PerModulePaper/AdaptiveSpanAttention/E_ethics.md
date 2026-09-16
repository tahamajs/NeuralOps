<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Interpretability:** Reveals head structure.
- **Foundation for pruning:** Basis for hard-window deployment.
- **Efficiency:** Enables fixed-window deployment.

### Negative
- **Dual-use:** Efficiency gains benefit malicious applications.
- **Concentration:** Favors large labs.

## E.2 Environmental Impact

- **Total GPU-hours:** ~{{COMPUTE_HOURS}}
- **Estimated CO₂:** {{CO2_KG}} kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English movie reviews)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias

Learned spans may correlate with language-specific structures. For English 
SST-2, this is fine; for multi-lingual, unclear.

### Misuse

Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

{{FUNDING}}

## E.8 Acknowledgements

{{ACKNOWLEDGMENTS}}

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document all hyperparameters

## E.11 Long-Term Considerations

Adaptive spans could:

- Reveal structural properties of transformer heads
- Enable efficient deployment on edge
- Reduce AI's carbon footprint

But also:

- Enable widespread AI deployment
- Concentrate capabilities

We encourage community discussion.
```

---