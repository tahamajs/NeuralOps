# Abstract Template

## Structure (200-250 words)

1. **Problem** (2 sentences)
   - What is expensive / broken / unclear
   - Who cares (cite 1-2 papers)

2. **Method** (3 sentences)
   - What you propose, in one sentence
   - Key idea in one sentence
   - Why it works (mechanism)

3. **Results** (3 sentences)
   - Best accuracy per backbone with delta vs. baseline
   - Compression ratio or efficiency gain
   - Comparison to 1-2 published baselines

4. **Honest caveat** (1 sentence)
   - Where it fails
   - What was not tested

## Anti-patterns

- Never say "significantly better" without a p-value
- Never say "state-of-the-art" without comparing to actual SOTA
- Never hide negative results
- Never use "novel" — say what specifically is new

## Checklist

- [ ] Under 250 words
- [ ] Every number matches `metrics/aggregated.json`
- [ ] If delta is negative, stated in abstract
- [ ] If parameters increased, stated in abstract
- [ ] If inference is slower, stated in abstract
