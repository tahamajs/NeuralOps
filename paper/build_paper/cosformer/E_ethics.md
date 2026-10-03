<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethics
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Linear attention** reduces memory for long sequences.
- **Port to pretrained models** enables reuse without retraining.
- **Reproducibility** — full code, weights, and metrics released.

### E.1.2 Negative

- **Marginal speedup** on short sequences.
- **RoBERTa instability** could mislead practitioners.

## E.2 Environmental Impact

- **GPU-hours:** ~4
- **CO₂:** ~0.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

Linear attention is important for long-context deployment. Cosformer is one 
of the better variants. We hope this work contributes to a broader 
understanding of when linear attention works and when it doesn't.
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---
```
