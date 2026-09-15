<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Efficient inference:** Reduces memory and compute per token.
- **Longer context:** Enables more capable applications.
- **Interpretability:** Head budgets reveal attention structure.

### Negative
- **Dual-use:** Efficiency gains benefit both beneficial and harmful 
  applications.
- **Concentration:** Favors large labs.
- **Rebound effect:** Cheaper inference may increase total usage.

## E.2 Environmental Impact

- **Total GPU-hours:** ~{{COMPUTE_HOURS}}
- **Estimated CO₂:** {{CO2_KG}} kg CO₂eq

Sparse attention reduces compute at inference.

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English movie reviews)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias

Learned sparsity may amplify or suppress biases depending on which heads 
are retained. Future work should audit.

### Misuse

Could enable surveillance at scale. Not endorsed.

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

Sparse attention could:

- Enable efficient LLM inference on consumer hardware
- Reduce AI's carbon footprint
- Reveal structural properties of attention

But also:

- Enable widespread AI deployment
- Concentrate capabilities
- Widen the digital divide

We encourage community discussion.
