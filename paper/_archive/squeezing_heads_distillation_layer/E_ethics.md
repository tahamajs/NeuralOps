<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- **Cheaper fine-tuning:** LoRA trains 1% of parameters.
- **Preserves pretrained:** No catastrophic forgetting.
- **Democratizes adaptation:** Smaller compute for task adaptation.

### Negative
- **Dual-use:** Fine-tuning benefits malicious applications.
- **Concentration:** Large labs benefit from efficient methods.

## E.2 Environmental Impact

- **Total GPU-hours:** ~10
- **Estimated CO₂:** 0.7 kg CO₂eq

LoRA reduces training compute significantly.

## E.3 Data Ethics

- **Dataset:** SST-2 (public, English)
- **PII:** None
- **Bias:** Western-centric

## E.4 Model Ethics

### Bias
Not audited. LoRA may preserve or amplify existing biases.

### Misuse
Not endorsed.

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
4. Document all hyperparameters

## E.11 Long-Term Considerations

Efficient fine-tuning could:

- Reduce barriers to model adaptation
- Reduce compute costs
- Enable personalization on consumer hardware

But also:

- Enable fine-tuning for harmful tasks at low cost
- Concentrate capabilities

We encourage community discussion.

---
