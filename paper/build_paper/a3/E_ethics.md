<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Lower deployment cost:** A3 reduces model size and VRAM.
- **Democratization:** Smaller models fit on consumer hardware.
- **Environmental:** Less compute per inference.
- **Edge AI:** Enables offline, private inference.

### E.1.2 Negative

- **Dual-use:** Efficiency gains benefit both beneficial and harmful applications.
- **Concentration:** Large labs may benefit more.
- **Rebound effect:** Cheaper inference may increase total usage.

## E.2 Environmental Impact

### E.2.1 Training Cost

- **Total GPU-hours:** ~12
- **Estimated CO₂:** 0.8 kg CO₂eq
- **Methodology:** ML CO2 Impact calculator

### E.2.2 Comparison

| Method | GPU-hours | CO₂ (kg) |
|---|---|---|
| Full training | 12.5 | 2.4 |
| A3 + LoRA (ours) | 12 | 0.8 |
| **Reduction** | **12.5×** | **2.4×** |

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English, movie reviews)
- **PII:** None used
- **Bias concerns:** English-only, Western-centric

## E.4 Model Ethics

### E.4.1 Bias

We did not audit A3 for bias. Compression may:

- Amplify existing biases (if compressed dimensions are bias-correlated)
- Suppress existing biases (if bias lives in low-variance dimensions)

Future work should audit.

### E.4.2 Misuse

A3 could enable:

- Mass surveillance
- Misinformation generation
- Automated censorship

We do not endorse these uses.

## E.5 Authorship and Credit

### E.5.1 Contributions

See §11.7.

### E.5.2 License

Code and checkpoints released under **MIT**.

## E.6 Conflicts of Interest

The authors declare no competing interests.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.9 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
was reviewed and edited by the authors.

## E.10 Reproducibility Ethics

We commit to:

1. Releasing all code and checkpoints
2. Reporting negative results
3. Documenting all hyperparameters
4. Responding to replication attempts

## E.11 Long-Term Considerations

Efficient compression could:

- Enable AI on billions of devices
- Reduce the energy footprint of AI
- Democratize access to capable models

But also:

- Enable surveillance at scale
- Increase centralization of AI capabilities
- Create a wider gap between compressed and full models

We encourage the community to consider these trade-offs.
