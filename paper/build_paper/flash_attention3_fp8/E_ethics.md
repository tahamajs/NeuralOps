<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- **Portability:** Code runs on any GPU.
- **Reproducibility:** Hardware detection logged.
- **Honesty:** FP8 limitations clearly documented.

### E.1.2 Negative

- **Expectation mismatch:** Users may expect H100 speedups on consumer GPUs.
- **Mitigation:** We document hardware requirements prominently.

## E.2 Environmental Impact

- **GPU-hours:** ~12.5
- **CO₂:** ~2.4 kg

Small compared to full training.

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.5 Conflicts of Interest

None.

## E.6 Long-Term Considerations

FP8 attention democratizes access to fast inference — but only on hardware 
that supports it. This may:

- Widen the gap between labs with Hopper GPUs and those without.
- Incentivize hardware upgrades.
- Reduce the value of software-only optimizations.

We do not claim to solve these issues; we simply document the trade-offs.

---