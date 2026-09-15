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
