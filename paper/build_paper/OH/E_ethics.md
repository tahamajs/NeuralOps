<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Long-context AI** on consumer hardware
- **Multi-architecture support**
- **Data-adaptive compression** respects input structure

### Negative
- **Not a compressor** — adds parameters
- **Long-context surveillance** potential
- **Dual-use** for beneficial and harmful applications

## E.2 Environmental Impact

- **Training:** ~10 GPU-hours
- **CO₂:** ~12.4 kg CO₂eq
- **Inference:** 0.15 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English movie reviews

## E.4 Model Ethics

### Bias

The data-dependent compression may amplify biases present in the training 
distribution. We did not audit this.

### Misuse

Long-context attention could enable:

- Mass surveillance of long text streams
- Automated manipulation
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

Parts drafted with LLM assistance. All content reviewed by authors.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document hyperparameters
5. Respond to replications

## E.11 Long-Term Considerations

OHA could:

- Enable long-context AI on edge devices
- Adapt to input structure

But also:

- Enable surveillance at scale
- Concentrate AI capabilities

We encourage the community to weigh these trade-offs.

---
