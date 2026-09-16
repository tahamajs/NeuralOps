<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Real compression** democratizes AI
- **Reduces carbon footprint**
- **Enables edge deployment**

### Negative
- **Dual-use** — efficiency benefits malicious actors
- **Concentration** — large labs benefit more
- **Rebound effect** — cheaper inference increases usage

## E.2 Environmental Impact

- **Training:** ~{{COMPUTE_HOURS}} GPU-hours
- **CO₂:** ~{{CO2}} kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English movie reviews

## E.4 Model Ethics

### Bias

Compression may amplify or suppress biases depending on which directions 
are preserved. We did not audit for bias.

### Misuse

Could enable surveillance, misinformation, censorship.

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
4. Document hyperparameters

## E.11 Long-Term Considerations

Compression could:

- Enable AI on billions of devices
- Reduce AI's carbon footprint
- Democratize access

But also:

- Enable mass surveillance
- Concentrate AI capabilities
- Widen gap between compressed and full models

---