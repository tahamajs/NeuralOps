<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethics
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Cheaper than attention-based pruning** — no attention scores needed.
- **Simple to implement** — no learned parameters.

### E.1.2 Negative

- **Accuracy drop on encoders** — not suitable for all deployments.
- **No decoder support** — limited applicability.

## E.2 Environmental Impact

- GPU-hours: ~12.5
- CO₂: ~2.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

Token pruning methods trade accuracy for compute. It is important to 
document their limitations accurately so practitioners can make informed 
decisions. Our analysis of FASA's heuristic weakness and decoder-path 
no-op is intended to enable such decision-making.
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---
```
