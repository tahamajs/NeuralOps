# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Long-context efficiency
- Content-aware sparsity
- Length-aware dispatch prevents overhead on short sequences
- Composable with other efficiency methods

### Negative
- Parameter overhead (+66.4M%)
- Silent at short sequences
- Dual-use

## E.2 Environmental

- Total compute: ~12.5 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

SST-2 public. No PII.

## E.4 Model Ethics

- No bias audit
- Block selection may encode positional bias

## E.5 Authorship

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Institutional Review

No human subjects.
```

---

## 📄 `build.py` و `apply_vars.py`

(مشابه ماژول‌های قبلی)

---
```
