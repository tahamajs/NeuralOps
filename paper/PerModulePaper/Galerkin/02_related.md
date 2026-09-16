<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Petrov-Galerkin Methods

Petrov-Galerkin methods (Quarteroni & Valli, 1994) approximate solutions of 
differential and integral equations by projecting onto finite-dimensional 
subspaces. Unlike Galerkin methods (test = trial space), Petrov-Galerkin 
uses different test and trial spaces, allowing non-symmetric operators to be 
handled. Our approach uses the same basis for trial and test — a **Bubnov-Galerkin** 
variant — but we retain the "Petrov" terminology following the original 
motivation of the module.

**Kernel methods.** The Nyström method (Williams & Seeger, 2001) also 
approximates kernel matrices via low-rank projection. Our approach differs 
in using a **fixed basis** rather than sampled landmarks.

## 2.2 Spectral Attention

**FNet** (Lee-Thorp et al., 2021). Replaces attention entirely with a 2D 
Fourier transform, achieving $\mathcal{O}(N \log N)$ complexity. This is the 
closest work to ours in spirit — but FNet replaces attention, while we 
project it.

**Random feature attention** (Peng et al., 2021). Uses random features to 
approximate softmax. Performer (Choromanski et al., 2021) uses positive 
orthogonal random features.

**Nyströmformer** (Xiong et al., 2021). Uses Nyström landmarks for low-rank 
attention approximation. Landmarks are data-dependent; our basis is fixed.

## 2.3 Low-Rank Attention

**Linformer** (Wang et al., 2020). Projects K and V to a lower rank via 
learned linear maps. Our projection is a fixed DCT basis.

**Synthesizer** (Tay et al., 2021). Uses learned or random attention 
matrices. Random Synthesizer is data-independent like ours.

## 2.4 Position of Our Work

| Axis | Prior work | Galerkin |
|---|---|---|
| **Basis type** | Learned (Linformer) or landmarks (Nyström) | Fixed DCT |
| **Attention replacement** | Full or partial | Projection only |
| **Parameter cost** | Zero or reduced | **Added LoRA** |
| **Encoder support** | Yes | Yes (but fails) |
| **Decoder support** | Yes | Falls back |

**Key distinction.** Unlike Linformer which **learns** low-rank projections, 
our approach uses a **fixed** DCT basis. This eliminates the parameter cost 
of projection matrices but sacrifices adaptivity. Our negative results 
suggest this trade-off is not worth it for attention.

---

## Comparison Table

| Method | Basis | Learned? | Params | SST-2 (DistilBERT) |
|---|---|---|---|---|
| Standard MHSA | — | — | 66.9M | 90.71% |
| Linformer | Linear | Yes | Similar | ~88% |
| FNet | Fourier | No | 66.9M | ~85% |
| Random Synthesizer | Random | No | 66.9M | ~85% |
| **Galerkin** | **DCT** | **No** | **+0.9%** | **{{D_ACC}}%** |
```

---