<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Documents implementation bottlenecks** that the community can avoid.
- **Provides a reference implementation** for future kernel development.
- **Warns against premature deployment** of linear attention.

### E.1.2 Negative

- **Compute consumed** for slow, incomplete runs.
- **May discourage research** into the algorithm (which is sound).

## E.2 Environmental Impact

- **GPU-hours:** ~12.5
- **CO₂:** ~2.4 kg

Larger than necessary due to slow implementation.

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

Linear attention is a promising direction for long-context models. However:

- **Implementation quality matters as much as algorithm design.**
- **Published results assume optimized kernels.**
- **Reproducing them requires equivalent engineering.**

We hope this work encourages:
- Better tooling for linear attention (e.g., `flash-linear-attention`)
- Standardized benchmarks that measure wall-clock, not just FLOPs
- Honest reporting of implementation gaps

## E.7 Recommendations

1. **Use existing optimized libraries** (Mamba, GLA, FLA) instead of 
   reimplementing.
2. **Measure end-to-end time** on target hardware before publishing.
3. **Document implementation details** for reproducibility.
```

---

## 📄 `build.py` and `apply_vars.py`

همان‌طور که در مقالات قبلی بود. فقط `SECTIONS` لیست را تغییر بده اگر فایل اضافه/کم شد.

---
```
