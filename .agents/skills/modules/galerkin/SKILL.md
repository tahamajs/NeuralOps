# Skill — Galerkin Attention

## Purpose
Petrov-Galerkin projection onto orthonormal harmonic basis.

## Rank argument
`rank` (basis rank)

## Known failure
- `rank=64` on `L=128` loses half the sequence
- Basis has no positional anchor
- Softmax over basis coefficients, not token similarities

## Special handling
- `04-rank-arg-mapping.md` (use `rank >= L`)
- `06-forward-sanity-check.md` (essential)
- `11-paper-specific-sections.md` (method section is complex)

## Skip on
Nothing — but requires `rank >= 128` for L=128
