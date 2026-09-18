# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Democratizes fine-tuning on consumer hardware
- Reduces environmental cost per model
- Complements compression for edge deployment

### Negative
- May enable malicious fine-tuning (dual-use)
- Adds a training step

## E.2 Environmental

- GPU-hours: ~12.5
- CO₂: 2.4 kg

**Comparison:**
- Full FT: ~10× more energy
- LoRA: energy proportional to rank

## E.3 Data Ethics

- SST-2 public
- No PII
- English-only

## E.4 Model Ethics

### Bias
Not audited. LoRA can adapt to biases in fine-tuning data.

### Misuse
Standard dual-use concerns.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.

---