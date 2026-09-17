<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Faster inference** for long-context applications
- **Multi-architecture support** simplifies deployment
- **LoRA recovery** preserves pretrained knowledge

### Negative
- **Not a compressor** — adds parameters
- **Pruning may remove content** critical for fairness
- **Dual-use** — efficiency enables both beneficial and harmful applications

## E.2 Environmental Impact

- **Training:** ~12.5 GPU-hours, ~12.4 kg CO₂eq
- **Inference:** 0.85 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** English-only movie reviews

## E.4 Model Ethics

### Bias and Pruning

Token pruning may disproportionately remove tokens from certain dialects, 
languages, or content types. We did not audit EAT for bias in pruning. 
**Recommendation:** Measure per-subgroup accuracy when using EAT with 
pruning.

### Misuse

EAT could enable more efficient surveillance. We do not endorse this use.

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

EAT's efficiency gains could:

- Make long-context models practical
- Reduce per-query cost

But also:

- Enable mass surveillance
- Concentrate AI capabilities

We encourage the community to weigh these trade-offs.

---