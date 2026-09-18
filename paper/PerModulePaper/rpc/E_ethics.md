<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Reduces compute** for long sequences
- **Multi-architecture support**
- **Mathematically principled** (JL lemma)

### Negative
- **Not a compressor** — adds parameters
- **Does not reduce memory** — limited benefit
- **Approximation error** may affect fairness
- **Dual-use** for surveillance

## E.2 Environmental Impact

- **Training:** ~12.5 GPU-hours
- **CO₂:** ~12.4 kg CO₂eq
- **Inference:** 0.85 g CO₂eq per 1000 queries

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None

## E.4 Model Ethics

### Bias

The random projection may distort attention patterns in ways that 
disproportionately affect minority inputs. We did not audit for bias.

### Misuse

RPC's efficiency gains could enable:

- More efficient surveillance
- Mass manipulation

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

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document the random seed for `proj`
5. Respond to replications

## E.11 Long-Term Considerations

RPC is a small step toward efficient attention. Its main value is 
mathematical (JL-based) rather than practical. We encourage future work 
that combines RPC's principled approach with AASVD-style compression.

---