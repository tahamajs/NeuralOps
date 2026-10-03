# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Faster inference → cheaper deployment
- Memory savings → enables long context
- Exact attention → no accuracy trade-off
- Broad hardware support via fallback

### Negative
- Hopper-only optimization → hardware lock-in
- Marginal benefits on consumer GPUs
- Dual-use (efficiency helps malicious and beneficial applications)

## E.2 Environmental

- Total compute: ~12.5 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

SST-2 public dataset. No PII.

## E.4 Model Ethics

- No bias audit
- Kernel-level changes may have subtle interactions with model behavior

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

(مشابه ماژول‌های قبلی — همان کد)

---
```
