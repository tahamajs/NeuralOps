# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Real compression
- Multi-architecture support
- Parameter-efficient fine-tuning

### Negative
- Silent injection failure
- Cross-attention not supported

## E.2 Environmental Impact

- Training: ~12.5 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None

## E.4 Model Ethics

### Bias

SVD truncation may suppress bias-correlated directions or amplify them. 
We did not audit for bias.

### Misuse

Standard.

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
4. Document spectral initialization

## E.11 Long-Term Considerations

Res-SVD is a practical compression method with PEFT structure. We 
encourage responsible deployment.

---