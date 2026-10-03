<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical and broader impact considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 – 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive Impacts

**Democratization of AI.** Adaptive SVD reduces model size and VRAM without 
requiring calibration data. This makes compression accessible to researchers 
and practitioners who lack access to target-task data or extensive compute.

**Environmental benefits.** Smaller models consume less energy at inference. 
The compression process itself uses ~10 GPU-hours, minimal 
compared to training.

**Edge deployment.** Compressed models fit on consumer hardware, enabling 
private on-device inference without cloud dependency.

### E.1.2 Negative Impacts

**Dual use.** Efficiency gains benefit both beneficial and harmful 
applications. Nothing in the method prevents malicious use.

**Concentration risk.** Large labs may benefit more from compression 
improvements due to their scale.

**Rebound effect.** Cheaper inference may increase total usage, offsetting 
energy savings.

### E.1.3 Environmental Cost

- **Compression compute:** ~10 GPU-hours
- **CO₂ estimate:** ~0.7 kg CO₂eq
- **Methodology:** [ML CO₂ Impact Calculator](https://mlco2.github.io/impact/)

## E.2 Data Ethics

**Dataset.** SST-2 (Socher et al., 2013):
- Public benchmark
- English movie reviews
- No personally identifiable information (PII)
- Western-centric distribution

**No calibration data.** Adaptive SVD requires no additional data collection, 
reducing privacy risk.

**Bias.** We did not audit for bias. Compression may:
- Amplify existing biases (if bias lives in high-rank directions)
- Suppress existing biases (if bias lives in low-rank directions)

## E.3 Model Ethics

**Dual-use potential.** The method could be used for:
- Automated content moderation
- Privacy-preserving on-device AI
- Efficient scientific computing

Or misused for:
- Mass surveillance
- Misinformation generation
- Automated censorship

We do not endorse harmful uses.

## E.4 Authorship and Credit

### Author Contributions

- **Taha Majs** designed the method, implemented the code, ran 
  experiments, wrote the paper.
- **Collaborative Research Team** provided methodological feedback and edited the manuscript.

### License

Code, weights, and this paper released under **MIT**.

## E.5 Conflicts of Interest

The authors declare no competing interests.

## E.6 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.7 Acknowledgements

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.8 LLM Usage Disclosure

Parts of this paper were drafted with large language model assistance. All 
technical content, experimental design, and final text were reviewed and 
edited by the authors.

## E.9 Reproducibility Commitments

We commit to:
1. Releasing all code under MIT
2. Releasing checkpoints on HuggingFace
3. Reporting negative results (Appendix D)
4. Documenting all hyperparameters
5. Responding to replication attempts

## E.10 Long-Term Considerations

Adaptive SVD could:

**Enable:**
- AI on billions of consumer devices
- Reduced AI carbon footprint
- Democratized access to capable models
- Multi-task deployment without re-compression

**Risk:**
- Increased surveillance capability
- Concentration of AI capabilities
- Widening gap between compressed and full models

We encourage the community to consider these trade-offs alongside technical 
merit.

## E.11 Institutional Review

This work does not involve human subjects. No IRB approval was required.

## E.12 Transparency

All experimental results (positive and negative) are reported in Sections 
7–8 and Appendix D. Code and checkpoints are released. Hyperparameters are 
documented in Section 6 and Appendix B.
