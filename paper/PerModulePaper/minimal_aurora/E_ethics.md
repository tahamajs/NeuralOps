# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Real compression
- Simple, auditable implementation
- Multi-architecture support

### Negative
- Silent injection failure (risk)
- `pres_gate` dead code (minor)

## E.2 Environmental Impact

- Training: ~{{COMPUTE_HOURS}} GPU-hours
- CO₂: ~{{CO2}} kg

## E.3 Data Ethics

- Dataset: SST-2 (public, English)
- PII: None

## E.4 Model Ethics

### Bias

SVD truncation may suppress or amplify bias-correlated directions. 
Not audited.

### Misuse

Standard dual-use.

## E.5 Authorship and Credit

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

{{FUNDING}}

## E.8 Acknowledgements

{{ACKNOWLEDGMENTS}}

## E.9 LLM Usage

Parts drafted with LLM assistance.

## E.10 Reproducibility Commitments

1. Release code
2. Release checkpoints
3. Report negative results
4. Document `pres_gate` dead code

## E.11 Long-Term Considerations

MinimalAURORA is a baseline compression method. It should be used 
responsibly.
```

---

## 📄 `build.py` و `apply_vars.py`

(همان‌طور که در تمپلیت‌های قبلی آمده بود — کپی کن)

---
```
