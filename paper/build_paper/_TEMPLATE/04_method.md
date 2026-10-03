# 4. Method

## 4.1 Overview

We replace the original attention computation with a hybrid mechanism that combines {{COMPONENT_1}}, {{COMPONENT_2}}, and {{COMPONENT_3}}. The goal is to preserve the pretrained behavior as much as possible while reducing the dominant cost term. The method is designed for compatibility with masked, causal, and grouped-query attention settings.

The pipeline is:

1. Project $X$ into $Q$, $K$, and $V$.
2. Apply RoPE or other positional encoding when required.
3. Compute the local and global branches.
4. Fuse them with a learnable gate.
5. Project the result back to the hidden dimension.

## 4.2 Component 1: {{COMPONENT_1_NAME}}

We define {{COMPONENT_1_NAME}} as

$$
{{COMPONENT_1_FORMULA}}
	ag{4.1}
$$

where {{VAR_1_DESC}} and {{VAR_2_DESC}}. The hyperparameter {{HYPERPARAM_1}} controls {{BEHAVIOR}}.

## 4.3 Component 2: {{COMPONENT_2_NAME}}

The second branch computes

$$
{{COMPONENT_2_FORMULA}}
	ag{4.2}
$$

This component handles {{SECOND_COMPONENT_ROLE}} and is designed to preserve local context or global structure depending on the architecture.

## 4.4 Component 3: {{COMPONENT_3_NAME}}

The third branch is defined as

$$
{{COMPONENT_3_FORMULA}}
	ag{4.3}
$$

This branch supports {{THIRD_COMPONENT_ROLE}} and stabilizes the approximation when the sequence is long or position-sensitive.

## 4.5 Combined Attention

The final attention output is computed as

$$
\mathcal{A}(Q, K, V) = \mathrm{Fuse}ig(\mathcal{A}_1, \mathcal{A}_2, \mathcal{A}_3ig)
	ag{4.4}
$$

with gate values $\gamma \in [0,1]$ learned per token. The gate is initialized to favor the original branch and is updated during recovery training.

## 4.6 Edge-Case Handling

- Padding tokens are masked before attention.
- Causal attention uses the standard lower-triangular mask.
- GQA is supported by repeating KV heads to match query heads.
- Short sequences are handled by falling back to dense local attention or a stabilized branch.

## 4.7 Complexity Analysis

The baseline complexity is quadratic in the sequence length. The proposed method reduces the dominant term to {{TOTAL_TIME}} while preserving the same output dimension and supporting the same pretrained model family.

## 4.8 Parameter Count and Initialization

The base weights are copied from the original attention modules. Trainable adapters are initialized to zero or near-zero, so the method starts from a faithful copy of the original behavior. This is important for preserving stability during the recovery stage.
