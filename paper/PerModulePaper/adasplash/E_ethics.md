<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical and broader impact
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 – 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Interpretability research.** Sparse attention is easier to analyze.
- **Attention visualization.** Clearer patterns for understanding model behavior.
- **Sparse hardware.** Potential for 2:4 sparsity speedup on specialized hardware.
- **Attention pattern analysis.** Enables research on attention mechanisms.

### E.1.2 Negative

- **Not a compression method.** Users may mistakenly expect size savings.
- **Slower inference.** Not suitable for production.
- **Dual-use.** Sparsity techniques can be used for surveillance or harmful 
  applications.

### E.1.3 Environmental Cost

- **Total compute:** ~12 GPU-hours
- **CO₂ estimate:** ~0.8 kg CO₂eq

## E.2 Data Ethics

- **Dataset:** SST-2 (public, English, movie reviews)
- **PII:** None
- **Bias:** English-only, Western-centric

## E.3 Model Ethics

**Bias.** We did not audit AdaSplash for bias. Sparsity may amplify or suppress 
biases depending on which attention patterns are preserved.

**Misuse.** Could enable surveillance or misinformation. We do not endorse 
these uses.

## E.4 Authorship and Credit

### Author Contributions

- **Taha Majs** designed the method, implemented the code, ran experiments, 
  wrote the paper.
- **Collaborative Research Team** provided feedback and edited the manuscript.

### License

Code and weights released under **MIT**.

## E.5 Conflicts of Interest

None.

## E.6 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.7 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.8 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All content reviewed 
by authors.

## E.9 Reproducibility Commitments

We commit to:
1. Release code
2. Release checkpoints
3. Report negative results (Appendix D)
4. Document hyperparameters
5. Respond to replication attempts

## E.10 Long-Term Considerations

AdaSplash could:

**Enable:**
- Interpretability research
- Sparse hardware deployment
- Attention pattern analysis

**Risk:**
- Misleading claims about "compression"
- Slower production systems
- Concentration of research capabilities

We encourage honest reporting of AdaSplash as a **research tool**, not a 
production method.

## E.11 Institutional Review

No human subjects. No IRB required.
