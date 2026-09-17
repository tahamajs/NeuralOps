# Appendix E. Ethical Considerations

## E.1 Broader Impact

### Positive
- Capacity expansion research
- Post-hoc MoE methods
- Multi-task learning potential

### Negative
- **Not a compression method** — may mislead
- Large parameter overhead (edge deployment impossible)
- Dual-use

## E.2 Environmental Cost

- Total compute: ~20 GPU-hours
- CO₂: ~12.4 kg CO₂eq

## E.3 Data Ethics

SST-2 public dataset, no PII.

## E.4 Model Ethics

- No bias audit performed
- Misuse potential (surveillance, misinformation)

## E.5 Authorship

See §11.7.

## E.6 Conflicts of Interest

None.

## E.7 Funding

This research was supported by the Advanced Machine Learning & Efficiency Laboratory.

## E.8 LLM Usage

Parts drafted with LLM assistance.

## E.9 Long-Term Considerations

BAM-UMoE is a research tool, not a production method. Honest reporting 
of parameter cost is essential.

## E.10 Institutional Review

No human subjects.
```

---

## 📄 `build.py` و `apply_vars.py`

(مشابه A3/AASVD/Adaptive SVD/AdaSplash — همان کد)

```python
# build.py and apply_vars.py: identical structure
# Change SECTIONS list to same as other modules

---