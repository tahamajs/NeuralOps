<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Democratizes long-context AI:** LSH enables longer sequences.
- **Reduces memory:** Less VRAM for deployment.
- **Enables edge AI:** Private, offline inference.

### Negative
- **Complexity:** Two-axis compression may hinder adoption.
- **Dual-use:** Efficiency benefits both benign and harmful applications.
- **Concentration risk:** Large labs may benefit more.

## E.2 Environmental Impact

- **Total GPU-hours:** ~12.5
- **Estimated CO₂:** 2.4 kg CO₂eq

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias concerns:** English-only, movie reviews

## E.4 Model Ethics

### Bias

Not audited for bias. Compression may amplify or suppress.

### Misuse

Could enable surveillance, misinformation, censorship. Not endorsed.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgments

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

PHANTOM enables:
- Long-context inference on consumer hardware
- Sparse attention research
- Multi-axis compression studies

But also:
- Potential for cheaper harmful deployment
- Complexity may widen gap between research and production

---