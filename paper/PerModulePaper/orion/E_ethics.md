<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethics
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Genuine parameter reduction enables wider deployment.
- Hybrid attention is a promising research direction.
- Bug disclosure aids reproducibility.

### E.1.2 Negative

- Two production bugs could affect downstream users.
- Slower than baseline.

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

Bugs in production code can cause silent failures. We document ORION's 
two bugs (`copy_weights` raise, `rff_proj` persistence) to prevent 
downstream issues.

---