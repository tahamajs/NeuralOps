<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethics
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Three-branch hybrid** demonstrates a flexible attention architecture.
- **`softpick` fix** improves robustness over prior versions.
- **Full code release** enables reproducibility.

### E.1.2 Negative

- **Robustness regression** in `inject_nasc`.
- **Raises on failure** — breaks production safety.

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

Robustness and reproducibility are critical for production deployment. 
Our disclosure of `inject_nasc`'s exception behavior is intended to 
prevent silent failures in deployed systems.
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---
```
