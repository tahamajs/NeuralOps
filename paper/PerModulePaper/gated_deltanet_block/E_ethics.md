<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Documents specific failure modes of block-level replacement.
- Warns against `try/except: pass` in critical code.
- Highlights LayerNorm's role in pretrained weight transfer.

### E.1.2 Negative

- Compute consumed for a non-working method.

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

Block-level replacement is a promising direction for hybrid architectures 
(RetNet, Mamba). However, it introduces new challenges:

- **LayerNorm preservation** is critical but often overlooked.
- **Silent failures** are dangerous in production.
- **Correctness** for causal models requires explicit attention.

We hope this work contributes to more rigorous block-level replacement 
research.
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---
```
