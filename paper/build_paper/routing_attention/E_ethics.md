<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive

- Contributes to sparse attention literature.
- Honest reporting of negative results.
- Open-source code for replication.

### E.1.2 Negative

- Not production-ready.
- May set incorrect expectations if not read carefully.

## E.2 Environmental Impact

- **GPU-hours:** ~12.5
- **CO₂:** ~2.4 kg

## E.3 Data Ethics

- SST-2 public dataset.
- No PII.

## E.4 Model Ethics

No new risks beyond base models.

## E.5 Authorship and License

- Code under Apache 2.0.
- Checkpoints under same.

## E.6 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.7 Acknowledgments

We thank the anonymous reviewers and the compute cluster administrators for their support.

## E.8 LLM Usage Disclosure

Parts of this paper were drafted with LLM assistance. All technical content 
reviewed by authors.

## E.9 Conflicts of Interest

None.
```

---

## 📄 `build.py` و `apply_vars.py`

همان اسکریپت‌های قبلی. `SECTIONS` را مطابق فایل‌های این ماژول تنظیم کن:

```python
SECTIONS = [
    "00_meta.md", "01_intro.md", "02_related.md", "03_background.md",
    "04_method.md", "05_architecture.md", "06_setup.md", "07_results.md",
    "08_analysis.md", "09_discussion.md", "10_limitations.md",
    "11_conclusion.md", "12_reproducibility.md", "13_references.md",
    "A_math.md", "B_implementation.md", "C_data.md",
    "D_negative_results.md", "E_ethics.md",
]

---