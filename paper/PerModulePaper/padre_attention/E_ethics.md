# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Provides polynomial attention baseline
- Multi-architecture support
- Open-source implementation

### Negative
- Kernelization deferred (no speedup yet)
- Polynomial breaks for large scores

## E.2 Environmental

- GPU-hours: ~{{COMPUTE_HOURS}}
- CO₂: {{CO2_KG}} kg

## E.3 Data Ethics

- SST-2 public
- No PII
- English-only

## E.4 Model Ethics

### Bias
Not audited.

### Misuse
Standard dual-use.

## E.5 Authorship

See §11.7.

## E.6 Conflicts

None.

## E.7 Funding

{{FUNDING}}

## E.8 LLM Usage

Drafted with LLM assistance. Content reviewed by authors.

## E.9 Honesty Notes

We explicitly document:
- **No kernelization** in current implementation
- **Runtime is unchanged** from softmax
- **Polynomial breaks for |s| > 5**

---