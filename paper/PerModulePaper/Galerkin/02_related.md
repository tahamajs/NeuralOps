# 2. Related Work

## 2.1 Linear-Complexity Attention

**Linformer.** Wang et al. (2020) project $K, V$ to fixed low rank with 
learned projection matrices. Galerkin uses a **fixed DCT basis**, not learned.

**Performer.** Choromanski et al. (2021) approximate softmax via random 
features. Galerkin computes **exact** attention in basis space.

**Nyströmformer.** Xiong et al. (2021) use Nyström landmarks. Galerkin uses 
a full DCT basis.

**Linear Transformer.** Katharopoulos et al. (2020) reorder attention as 
associative products. Different mathematical primitive.

## 2.2 Basis Projection Methods

**FNet.** Lee-Thorp et al. (2021) replace self-attention with an unparameterized 
Fourier transform. Galerkin uses a truncated DCT basis plus a softmax 
attention in basis space.

**GFNet.** Rao et al. (2021) use global filter operations in Fourier space 
for vision. Similar spirit, different domain.

**AFNO.** Guibas et al. (2022) use adaptive Fourier neural operators for 
PDEs. Galerkin is a Galerkin discretization of the attention operator.

## 2.3 Numerical Analysis of Attention

**Neural Operators.** Kovachki et al. (2023) formalize neural operators as 
mappings between function spaces. Attention is a specific case.

**Galerkin Methods.** Classical theory (Strang, 1973; Brenner & Scott, 2008) 
for variational discretization of PDEs.

**Petrov-Galerkin.** Uses different test and trial bases. Galerkin Attention 
uses a **single** orthonormal basis (Galerkin, not Petrov-Galerkin in the 
strict sense).

## 2.4 Position of Our Work

| Method | Basis | Learned | Encoder | Decoder |
|---|---|---|---|---|
| Linformer | Linear proj | Yes | Yes | Yes |
| Performer | Random feats | No | Yes | Yes |
| FNet | FFT | No | Yes | No |
| Nyströmformer | Landmarks | Yes | Yes | Yes |
| **Galerkin** | **DCT (fixed)** | **No** | **Yes** | **Fallback** |

**Key distinction.** Galerkin is the only method that:
1. Uses a **fixed orthonormal basis** (DCT).
2. Dispatches encoder vs decoder.
3. Provides an **exact** solution in basis space.

## 2.5 Comparison Table

| Feature | Longformer | Linformer | Galerkin |
|---|---|---|---|
| Complexity | $O(LW)$ | $O(Lr)$ | $O(Lr)$ |
| Basis | Window | Learned | **DCT** |
| Exact? | Yes | Approx. | **Yes in basis space** |
| Decoder support | Yes | Partial | **Fallback** |
| Multi-arch | Partial | Partial | Full |
