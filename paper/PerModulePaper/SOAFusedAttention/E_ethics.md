# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Reduces model size for edge deployment.
- **Zero-shot** — no privacy concerns from calibration data.
- Simple and reproducible.

### Negative
- Attention still quadratic — not a long-context solution.
- Modest total reduction (attention only).
- Compression is dual-use.

## E.2 Environmental

- Total GPU-hours: ~8.
- CO₂: 0.6 kg.

## E.3 Data Ethics

- SST-2: public, English.
- No PII.
- **No calibration data used** — a privacy advantage.

## E.4 Model Ethics

Not audited for bias.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance. Content reviewed by authors.

## E.9 Long-Term

Low-rank compression could:
- Enable AI on smaller devices.
- Reduce deployment costs.
- Democratize access to capable models.

But also:
- Enable surveillance at scale.
- Widen the gap between compressed and full models.

---