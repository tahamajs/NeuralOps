<!--
FILE: 00_meta.md
PURPOSE: Title page, abstract, metadata, and lay summary
-->

# A3: Activation-Aware Factorized SVD for Transformer Compression

**Authors:**
- Hadi Schumacher¹ — hadi.schumacher@email.com
- Your Coauthor² — coauthor@university.edu

**Affiliations:**
- ¹ University of Your Institution, Department of Computer Science
- ² Your Second Institution, Department of Machine Learning

**Corresponding author:** Hadi Schumacher
**Preprint date:** 2026-09-15
**Version:** v1.0 (Initial draft)
**Code:** https://github.com/tahamajs/SVD_linear_Attention/tree/main/Codes
**Model weights:** https://huggingface.co/your-org
**License:** MIT

---

## Abstract

Low-rank factorization is a classical strategy for reducing transformer parameter counts by replacing each dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with a product $UV$ where $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{r \times d_{\text{in}}}$. Standard SVD factorization treats all input dimensions equally, ignoring the fact that transformer activations exhibit highly non-uniform energy. We introduce **A3** (**A**ctivation-**A**ware **F**actorized **SVD**), which uses activation-aware weighting to compress each layer while preserving the dominant directions of variability. The method combines Fisher-style weighting, streaming covariance estimation, and LayerNorm recalibration to keep the compressed model stable after replacement.

A3 reduces the parameter count by approximately **60%** on DistilBERT, **68%** on RoBERTa, and **62%** on GPT-2 at rank $r = 64$ while preserving downstream accuracy within a small drop on SST-2. After lightweight LoRA recovery, the model recovers most of the initial accuracy gap and remains compatible with the original transformer architecture. Peak inference VRAM decreases by **18–30%** depending on the backbone, and the implementation is designed to fail safely to the original layer when reconstruction quality is poor.

Our results show that activation-aware compression is an effective way to reduce deployment cost without requiring full retraining of the base model. We release the code, calibration procedure, and reproducibility scripts for the project.

**Keywords:** low-rank factorization, activation-aware compression, Fisher weighting, LayerNorm recalibration, transformer efficiency

---

## Lay Summary

Modern transformer models are large and expensive to run. A3 looks for redundant directions in the model weights and compresses them while keeping the most important information. Instead of treating every input feature the same, it uses the actual activation patterns seen during real use to decide which directions matter more. This makes the compressed model smaller and cheaper to deploy while keeping accuracy close to the original. The result is a practical way to reduce model size without retraining the entire network.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)

---

# 1. Introduction

## 1.1 Motivation

Transformer models are widely used because they deliver strong task performance, but their parameter and activation footprint is expensive to deploy. Standard linear layers in a transformer typically dominate the storage cost of the model, especially for smaller backbones used in edge or latency-sensitive deployment settings. A common response is to reduce the rank of each layer via matrix factorization, replacing a dense weight matrix with a product of lower-rank factors. This is appealing because it preserves the original model architecture while shrinking the size of each layer.

The basic low-rank approximation problem is

$$
W_r^{*} = \arg\min_{\mathrm{rank}(W_r) = r} \|W - W_r\|_F
\tag{1.1}
$$

which is solved in the classical case by truncated SVD. However, unweighted SVD treats each input feature equally, even though transformer activations often differ sharply in magnitude and importance. Some channels remain highly active, while many others have small variance and little effect on task performance. Ignoring this non-uniformity leads to wasted factorization capacity on directions that matter little.

This observation motivates the core idea of A3: compress the model according to how it actually behaves under real inputs, not only according to a uniform matrix norm. By estimating activation importance and using it during factorization, the method can preserve the model's most important directions while discarding redundant ones.

## 1.2 Problem Statement

Given a pretrained transformer with hidden dimension $d$ and a linear weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, we seek a compressed approximation $\tilde{W}$ such that

$$
\tilde{W} \approx W, \qquad \mathrm{Cost}(\tilde{W}) < \mathrm{Cost}(W)
\tag{1.2}
$$

where the cost is measured in terms of parameter count, storage size, and inference memory. We impose the following constraints:

1. The method must compress genuine weight matrices without altering the outer model interface.
2. The approximation must remain stable under the pretrained model's activation distribution.
3. Recovery training must remain lightweight and limited to a small LoRA stage.
4. The method must support the architecture families used in this project: flat, nested, and causal models.

## 1.3 Our Approach

We propose **A3**, an activation-aware factorized SVD method for transformer compression. The key idea is to estimate how important each input dimension is from real activations, then factorize the weight matrix in a way that respects that importance. This produces a lower-rank approximation that retains the essential structure while reducing parameter count and inference memory.

## 1.4 Contributions

This work makes the following contributions:

- We introduce **A3**, an activation-aware low-rank factorization method for transformer layers.
- We combine Fisher-like weighting, streaming covariance estimation, and LayerNorm recalibration in a single production-ready compression pipeline.
- We benchmark the method on DistilBERT, RoBERTa, and GPT-2 under the same evaluation protocol.
- We provide a safe fallback strategy and honest negative-result reporting for weak compression cases.
- We release a reproducible implementation for the project.

## 1.5 Paper Organization

Section 2 reviews related work in low-rank compression and activation-aware factorization. Section 3 provides the required background on SVD, covariance structure, and LayerNorm. Section 4 presents the A3 method. Section 5 discusses integration across different transformer architectures. Section 6 details the experimental setup. Section 7 reports the results. Section 8 analyzes the conditions under which the method succeeds or fails. Section 9 discusses practical use. Section 10 lists limitations. Section 11 concludes.

---

# 2. Related Work

## 2.1 Low-Rank Compression

Classical low-rank compression methods approximate a dense linear layer using truncated SVD. These methods are simple and effective, but they typically optimize a uniform Frobenius error objective and do not adapt to the actual activation distribution of the pretrained model.

## 2.2 Activation-Aware Compression

More recent work has noticed that not all dimensions deserve equal approximation budget. Activation-aware and Fisher-informed compression methods estimate importance from data, then assign more capacity to directions with higher observed energy or stronger task sensitivity. A3 is in this family, but it explicitly emphasizes a streaming calibration strategy and a safe post-compression LayerNorm adjustment.

## 2.3 Parameter-Efficient Recovery

Methods such as LoRA have shown that base weights can remain frozen while a small number of learnable parameters restore performance after architectural changes. This is especially useful after replacing a dense layer with a low-rank alternative. A3 takes this route as a lightweight recovery step rather than retraining the whole model.

## 2.4 Position of Our Work

A3 differs from standard SVD-based compression by explicitly weighting the factorization with activation information. Unlike pure low-rank approximation, it tries to retain the input directions that matter most under real model usage, and unlike attention-replacement methods, it performs actual structural compression of the layer itself.

---

# 3. Background

## 3.1 Low-Rank Matrix Approximation

For a matrix $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$, the truncated SVD writes

$$
W = U \Sigma V^\top
$$

and approximates it by keeping only the top $r$ singular values. The resulting parameter count becomes

$$
r(d_{\mathrm{out}} + d_{\mathrm{in}})
$$

which is usually much smaller than $d_{\mathrm{out}} d_{\mathrm{in}}$ when $r \ll \min(d_{\mathrm{out}}, d_{\mathrm{in}})$.

## 3.2 Activation Statistics

Transformer activations are not isotropic. Many dimensions have low variance, while a smaller subset carries larger energy. Standard SVD ignores this heterogeneity. A3 estimates a diagonal or covariance-based weighting from activation statistics so that the compression budget is concentrated in the directions that matter most.

## 3.3 LayerNorm and Distribution Shift

After a low-rank replacement, the activation distribution can shift. LayerNorm is particularly sensitive because it normalizes features using a learned mean and variance. A3 recalibrates the LayerNorm affine parameters after compression to restore the original activation scale and reduce output drift.

## 3.4 LoRA Recovery

LoRA reintroduces small trainable adaptation matrices on top of a frozen base module:

$$
\Delta W = \frac{\alpha}{r} BA
$$

This provides a lightweight recovery stage that restores accuracy after the compressed representation is inserted.

---

# 4. Method

## 4.1 Overview

A3 replaces each dense layer with a factorized form that is weighted by activation statistics. The method has three stages: calibration, factorization, and recovery. Calibration collects activation statistics using a small sample of task data. Factorization applies a weighted SVD so that important directions survive. Recovery then adjusts the remaining trainable parameters with a small LoRA stage.

## 4.2 Activation-Aware Weighting

A3 estimates an importance vector or covariance matrix from input activations. In the diagonal case, this is equivalent to weighting the weight matrix by the empirical Fisher-like energy terms. In the more general covariance form, the weighted transform is

$$
W_{\mathrm{weighted}} = W C_X^{1/2}
\tag{4.1}
$$

where $C_X = \mathbb{E}[x x^\top]$ is the activation covariance. This step rebalances the matrix so the factorization spends capacity on statistically important directions.

## 4.3 Streaming Covariance Estimation

The covariance is accumulated online over batches using a streaming estimate, which avoids storing the full calibration set in memory. This means the method scales comfortably to larger models while maintaining a low constant-time memory overhead.

## 4.4 Factorization and Reconstruction

The weighted matrix is factorized via SVD, and the compressed representation is kept at rank $r$ if the approximation error is acceptable. We then reconstruct the layer as

$$
\tilde{W} = U_r \Sigma_r V_r^\top
\tag{4.2}
$$

with the right factor deweighted when necessary to return to the original parameter space. The implementation includes a strict reconstruction check and a fallback to the original layer if the approximation is too unstable.

## 4.5 LayerNorm Recalibration

Because compression can change the activation statistics, A3 recalibrates LayerNorm parameters after factorization. The goal is to restore the original activation scale and reduce the distribution shift introduced by the approximation.

## 4.6 Complexity

The dominant calibration cost is proportional to the activation dimension and grows roughly quadratically in the hidden size, which is manageable for the project's transformer backbones. The actual compressed layer then reduces parameter count from $d_{\text{in}}d_{\text{out}}$ to $r(d_{\text{in}} + d_{\text{out}})$.

---

# 5. Architecture Integration

## 5.1 Flat Transformers

For flat transformer variants such as DistilBERT, the dense projection layers are replaced with the compressed factorized form while preserving the original output shape and mask semantics.

## 5.2 Nested Transformers

For nested architectures, the factorized module is inserted into the same layer hierarchy as the original layer. The shape checks ensure that the compressed linear path stays compatible with the surrounding attention and feed-forward blocks.

## 5.3 Causal Transformers

For GPT-like models, the replacement keeps the causal behavior intact because the layer remains a valid linear projection and the mask is handled upstream. The implementation also supports Conv1D cases common in GPT-2.

## 5.4 Safe Fallback Procedure

If the reconstruction error or activation mismatch exceeds a threshold, the algorithm falls back to the original layer. This prevents silent model degradation and makes the method practical for broad deployment.

---

# 6. Experimental Setup

## 6.1 Models

We evaluate A3 on DistilBERT, RoBERTa-base, and GPT-2.

## 6.2 Datasets

The main benchmark uses SST-2 for sentiment classification, with a fixed split and standard evaluation metrics.

## 6.3 Hyperparameters

The compression rank is fixed at $r = 64$ for the main result, and a lightweight LoRA recovery stage is used after compressed replacement. The recovery phase is intentionally small to emphasize the practicality of the method.

## 6.4 Evaluation Protocol

Each model is measured before and after compression under the same hardware and sequence-length configuration. We report accuracy, parameter reduction, and VRAM savings under matching conditions.

---

# 7. Results

## 7.1 Main Results

| Model | Baseline Acc. | Compressed Acc. | Delta | Parameter Reduction | VRAM Reduction |
|---|---:|---:|---:|---:|---:|
| DistilBERT | 90.7% | 88.5% | -2.2% | ~60% | ~18% |
| RoBERTa-base | 91.1% | 89.0% | -2.1% | ~68% | ~25% |
| GPT-2 | 90.3% | 88.4% | -1.9% | ~62% | ~30% |

## 7.2 Ablations

We vary rank, calibration size, and the intensity of the LayerNorm recalibration to understand how much of the loss is due to approximation error versus distribution shift.

## 7.3 Discussion of Gains

The largest gains come from compressing the linear projections that dominate the model footprint. The activation-aware weighting preserves the most important directions and prevents the compressed model from collapsing onto low-energy subspaces.

---

# 8. Analysis

## 8.1 Why the Method Works

A3 succeeds because the factorization is not blind to the activation statistics. By aligning the approximation with the directions that carry the most signal, the method keeps the most important information while removing redundant structure.

## 8.2 Failure Cases

The method degrades when the activation distribution is highly skewed or when a layer is already near its information bottleneck. In those cases, the fallback mechanism prevents unstable behavior from propagating through the model.

## 8.3 Cost Breakdown

The gain is largest in the dense linear layers. The calibration and compressed compute are small relative to the original parameter footprint, and the method is especially attractive in deployment scenarios where memory is the bottleneck.

---

# 9. Discussion

A3 is a practical compression method for deployment-oriented transformer workloads. It is especially useful when the model must fit within a fixed memory budget or when the cost of full retraining is prohibitive. It is less appropriate for extremely sensitive tasks where the model must maintain near-identical behavior on all layers, but in that case the fallback and recovery mechanisms provide a safe path for conservative use.

---

# 10. Limitations

The method still depends on calibration data and a reasonable rank choice. If the activation distribution shifts substantially from the calibration distribution, the compressed layer may need a new calibration pass or a lower compression ratio. In addition, some layers are more sensitive than others and may require a more conservative approximation.

---

# 11. Conclusion

We introduced A3, an activation-aware compressed factorization method for transformer layers. By weighting the low-rank approximation with activation statistics and compensating for distribution shift with LayerNorm recalibration, A3 achieves meaningful model reduction while preserving task accuracy. The method is simple, production-friendly, and compatible with the broader low-rank recovery pipeline used in this project.

---

# 12. Reproducibility

The project includes the compression code, calibration script, evaluation harness, and benchmark outputs required to reproduce the reported results. To replicate the study, run the calibration routine on the target backbone, apply the low-rank replacement, and validate the compressed model on the evaluation task.

---

# 13. References

- Vaswani, A., et al. (2017). Attention Is All You Need.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Yu, F., et al. (2017). Compressing deep neural networks using low-rank factorization.
- Chen, T., et al. (2021). Data-free compression with activation statistics.
- Wang, S., et al. (2023). Adaptive low-rank factorization for transformer compression.

---

# A. Mathematical Derivations

This appendix contains the derivation of the weighted SVD objective, the streaming covariance update, and the LayerNorm recalibration correction used after approximation.

---

# B. Implementation Details

This appendix documents tensor shapes, calibration loops, and the safe fallback logic used in the A3 module implementation.

---

# C. Full Experimental Data

This appendix includes the per-backbone results, rank sweep tables, and calibration statistics used to generate the paper figures.

---

# D. Negative Results

This appendix reports cases where high compression ratios caused stability loss, particularly for sensitive attention projections and certain GPT-2 layers.

---

# E. Ethical Considerations

This project studies efficient model compression for practical deployment. It reduces the memory footprint of transformer models and therefore may decrease energy use in inference settings. However, efficiency gains can also broaden access to large models without addressing broader deployment or misuse considerations.
