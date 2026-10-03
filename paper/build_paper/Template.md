```
# Per-Module Paper Template

This is a clean, fillable template for a single-module paper in this project. It is designed for attention and compression modules, but it can also be adapted to any other architecture-level contribution.

## Folder structure

```text
papers/
├── _TEMPLATE/
│   ├── 00_meta.md
│   ├── 01_intro.md
│   ├── 02_related.md
│   ├── 03_background.md
│   ├── 04_method.md
│   ├── 05_architecture.md
│   ├── 06_setup.md
│   ├── 07_results.md
│   ├── 08_analysis.md
│   ├── 09_discussion.md
│   ├── 10_limitations.md
│   ├── 11_conclusion.md
│   ├── 12_reproducibility.md
│   ├── 13_references.md
│   ├── A_math.md
│   ├── B_implementation.md
│   ├── C_data.md
│   ├── D_negative_results.md
│   ├── E_ethics.md
│   ├── build.py
│   ├── variables.json
│   └── figures/
│       └── .gitkeep
└── <module_name>/
    ├── 00_meta.md
    ├── 01_intro.md
    ├── 02_related.md
    ├── 03_background.md
    ├── 04_method.md
    ├── 05_architecture.md
    ├── 06_setup.md
    ├── 07_results.md
    ├── 08_analysis.md
    ├── 09_discussion.md
    ├── 10_limitations.md
    ├── 11_conclusion.md
    ├── 12_reproducibility.md
    ├── 13_references.md
    ├── A_math.md
    ├── B_implementation.md
    ├── C_data.md
    ├── D_negative_results.md
    ├── E_ethics.md
    ├── build.py
    ├── variables.json
    └── figures/
```

## Recommended workflow

1. Copy this template to a new module folder.
2. Fill in the metadata in `00_meta.md`.
3. Replace all `{{PLACEHOLDER}}` fields with real values.
4. Use the same model names, evaluation setup, and metrics as your benchmark runs.
5. Keep the phrasing honest: if runtime increases or parameters change, state it explicitly.
6. Keep the paper aligned with the actual code and benchmark outputs.

---

## 00_meta.md

```markdown
<!--
FILE: 00_meta.md
PURPOSE: Title page, metadata, abstract, lay summary
-->

# {{FULL_NAME}} ({{MODULE_NAME}}): {{SUBTITLE}}

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}} ({{CHANGELOG}})
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/{{MODULE_PATH}}
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Transformer self-attention scales quadratically with sequence length, making long-context inference expensive in memory-constrained deployment settings. Existing solutions based on sparse patterns, low-rank approximations, and KV-cache pruning trade off accuracy, stability, or implementation complexity. We introduce **{{FULL_NAME}}** (**{{MODULE_NAME}}**), a {{CATEGORY}} that {{CORE_BEHAVIOR_IN_ONE_SENTENCE}}.

{{MODULE_NAME}} combines {{COMPONENT_1}}, {{COMPONENT_2}}, and {{COMPONENT_3}} without retraining the base model. We transfer pretrained attention weights into the modified module and initialize trainable adapters to zero so the baseline behavior is preserved before recovery training. We evaluate on {{N_MODELS}} backbones on the {{TASK}} task with {{RECOVERY_STEPS}} LoRA recovery steps at rank {{LORA_RANK}}.

The method achieves **{{ACC_D}}%** on DistilBERT, **{{ACC_R}}%** on RoBERTa-base, and **{{ACC_G}}%** on GPT-2, relative to baselines of **{{BASE_D}}%**, **{{BASE_R}}%**, and **{{BASE_G}}%**, respectively. Parameter counts change by **{{PARAM_CHANGE_PCT}}%** ({{PARAM_CHANGE_NOTE}}), while peak inference VRAM decreases by **{{VRAM_CHANGE_PCT}}%** and wall-clock latency changes by **{{LATENCY_CHANGE}}×** due to {{LATENCY_REASON_SHORT}}.

Our results suggest that {{WHAT_WORKS}} and that {{WHAT_FAILS}} under {{FAILURE_CONDITION}}. We release the implementation, evaluation scripts, and checkpoints to support reproducibility.

**Keywords:** efficient attention, {{KEYWORD_2}}, {{KEYWORD_3}}, {{KEYWORD_4}}, transformer compression

---

## Lay Summary

{{LAY_SUMMARY}}

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
```

---

## 01_intro.md

```markdown
# 1. Introduction

## 1.1 Motivation

Deploying transformer models in resource-constrained settings is increasingly common in edge inference, latency-sensitive serving, and long-context workloads. Standard self-attention has quadratic complexity in the sequence length, which creates a bottleneck when sequence lengths grow beyond a few thousand tokens. Sparse attention (Longformer, BigBird), low-rank attention (Linformer, Nyströmformer), and kernelized attention (Performer, Linear Transformer) each address this challenge in different ways, but they often require architectural changes, retraining, or careful tuning to remain stable.

Standard scaled dot-product attention computes

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\tag{1.1}
$$

where $Q, K, V \in \mathbb{R}^{L \times d_h}$. The core bottleneck is the $QK^\top$ matrix, which requires $\mathcal{O}(L^2)$ time and memory.

Recent work has highlighted that this issue matters not only for latency but also for memory footprint, throughput, and deployability under tight hardware limits. In parallel, parameter-efficient adaptation methods such as LoRA have shown that pretrained models can often be recovered with a small number of trainable parameters. This creates an opportunity to replace expensive attention components in a pretrained model without retraining the full backbone.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with hidden size $d$, $H$ attention heads, and $N_\ell$ layers, we seek a modified attention function $\mathcal{A}'$ such that

$$
\mathcal{A}' \approx \mathcal{A}, \qquad \mathrm{Cost}(\mathcal{A}') < \mathrm{Cost}(\mathcal{A})
\tag{1.2}
$$

where the cost is measured in terms of {{TARGET_METRIC}}. We impose the following constraints:

1. No retraining of base weights.
2. Accuracy degradation must remain within {{ACC_TOLERANCE}}% on {{EVAL_TASK}}.
3. Peak inference VRAM must remain below {{VRAM_TARGET}} GB on {{GPU_MODEL}}.
4. The method must work across flat, nested, and causal transformer architectures.

## 1.3 Our Approach

We propose {{FULL_NAME}} ({{MODULE_NAME}}), which {{HIGH_LEVEL_DESCRIPTION}}. The key idea is to preserve the original pretrained attention structure while replacing the expensive component with a lower-cost approximation that can be recovered through a lightweight LoRA-based fine-tuning stage. Section 4 formalizes the method and derives the resulting attention formulation.

## 1.4 Contributions

We make the following contributions:

- We introduce **{{MODULE_NAME}}**, a {{CATEGORY}} that {{BEHAVIOR_1}}.
- We provide a production-hardened PyTorch implementation that handles {{ENGINEERING_1}}, {{ENGINEERING_2}}, and {{ENGINEERING_3}} across three architectures.
- We benchmark {{N_MODELS}} model families on {{TASK}} with {{N_ABLATIONS}} ablations over rank, recovery steps, and {{HYPERPARAM_3}}.
- We report honest negative results for {{FAILING_MODELS}} and explain the observed failure modes.
- We release code, checkpoints, and evaluation scripts under {{LICENSE}}.

## 1.5 Paper Organization

Section 2 reviews related work in efficient attention, KV-cache compression, and parameter-efficient fine-tuning. Section 3 provides background on fundamentals relevant to the method. Section 4 presents the proposed module. Section 5 describes architecture integration. Section 6 explains the experimental setup. Section 7 reports results. Section 8 analyzes failure modes and inference costs. Section 9 discusses practical usage. Section 10 describes limitations. Section 11 concludes.
```

---

## 02_related.md

```markdown
# 2. Related Work

## 2.1 Efficient Attention Mechanisms

Sparse attention methods such as Longformer and BigBird reduce the attention pattern to a subset of tokens while keeping a small number of global connections. Low-rank approaches such as Linformer and Nyströmformer compress the projection space to approximate the attention matrix. Kernel methods such as Performer and Linear Transformer replace the softmax kernel with a more tractable feature map, enabling near-linear complexity in some settings.

## 2.2 KV-Cache Compression

Methods such as H2O, StreamingLLM, and KV quantization focus on reducing the cost of caching keys and values during generation. These methods are generally complementary to attention approximation and can be combined with architectural changes when memory constraints are dominant.

## 2.3 Parameter-Efficient Fine-Tuning

LoRA freezes the base weights and learns low-rank adapters, making it a natural mechanism for recovery after replacing a pretrained attention module. AdaLoRA and DoRA improve the choice of rank or decomposition, but the core principle remains the same: adapt only a small trainable subset while preserving the base model.

## 2.4 Position of Our Work

{{MODULE_NAME}} differs from these methods in three ways. First, it targets architecture portability across flat, nested, and causal models without retraining the original module. Second, it preserves the pretrained attention weights and uses LoRA-like recovery only for adaptation. Third, it is explicitly benchmarked under a fixed evaluation protocol with honest failure reporting and implementation guardrails.
```

---

## 03_background.md

```markdown
# 3. Background

## 3.1 Multi-Head Self-Attention

For input $X \in \mathbb{R}^{B \times L \times d}$, multi-head attention projects keys, queries, and values via learned matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

The attention output is computed as

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

This produces the standard quadratic attention bottleneck in sequence length.

## 3.2 Grouped-Query Attention

Grouped-query attention reduces the number of KV heads relative to Q heads, lowering cache size and memory cost while preserving a large query head count. This design is common in modern LLMs and matters for efficient inference.

## 3.3 Rotary Position Embeddings

RoPE injects position-dependent rotations into query and key vectors to encode relative order. This matters for causal generation and for any module that must preserve positional behavior under long-context usage.

## 3.4 LoRA Recovery

LoRA learns a low-rank residual update of the form

$$
\Delta W = \frac{\alpha}{r}BA
$$

where $B$ and $A$ are trainable low-rank matrices and $W$ is the frozen base weight. In the recovery phase, the model can adapt to a new attention structure while preserving the original pretrained weights.

## 3.5 Attention Sinks and Heavy Hitters

Recent work shows that a small number of tokens can dominate attention concentration. This motivates guardrails for padding, causal masking, and stable initialization when replacing attention paths.
```

---

## 04_method.md

```markdown
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
\tag{4.1}
$$

where {{VAR_1_DESC}} and {{VAR_2_DESC}}. The hyperparameter {{HYPERPARAM_1}} controls {{BEHAVIOR}}.

## 4.3 Component 2: {{COMPONENT_2_NAME}}

The second branch computes

$$
{{COMPONENT_2_FORMULA}}
\tag{4.2}
$$

This component handles {{SECOND_COMPONENT_ROLE}} and is designed to preserve local context or global structure depending on the architecture.

## 4.4 Component 3: {{COMPONENT_3_NAME}}

The third branch is defined as

$$
{{COMPONENT_3_FORMULA}}
\tag{4.3}
$$

This branch supports {{THIRD_COMPONENT_ROLE}} and stabilizes the approximation when the sequence is long or position-sensitive.

## 4.5 Combined Attention

The final attention output is computed as

$$
\mathcal{A}(Q, K, V) = \mathrm{Fuse}\big(\mathcal{A}_1, \mathcal{A}_2, \mathcal{A}_3\big)
\tag{4.4}
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
```

---

## 05_architecture.md

```markdown
# 5. Architecture Integration

## 5.1 Flat Transformers

For flat transformers such as DistilBERT, we directly replace the attention module in each layer. The implementation preserves the same projection layout and mask semantics as the original architecture.

## 5.2 Nested Transformers

For nested architectures, the attention replacement must respect the internal grouping patterns and head arrangements. We handle this by wrapping the module logic in a compatibility layer that flattens and restores the expected tensor shapes.

## 5.3 Causal Transformers

For GPT-like causal models, causal masks and rotary position embeddings must remain intact. The replacement module verifies that the mask is applied before the attention combination and that the positional encoding remains consistent with the original model.

## 5.4 Weight Copy Procedure

We copy the original $W_Q$, $W_K$, $W_V$, and $W_O$ weights into the replacement module when possible, then initialize the residual adapters to zero. This keeps the module close to the pretrained behavior at initialization and ensures a stable recovery process.
```

---

## 06_setup.md

```markdown
# 6. Experimental Setup

## 6.1 Models

We evaluate on {{MODEL_LIST}}.

## 6.2 Datasets

We use {{DATASET_NAME}} for {{TASK}}. The benchmark uses the standard train/validation/test split and reports accuracy as the primary metric.

## 6.3 Hyperparameters

Training uses {{RECOVERY_STEPS}} recovery steps at rank {{LORA_RANK}} and scale factor {{LORA_ALPHA}}. The optimizer is {{OPTIMIZER}} with learning rate {{LR}}.

## 6.4 Evaluation Protocol

We report baseline, recovered, and recovered-with-module results for each model family. We also report inference memory and latency under the same hardware conditions and sequence lengths.
```

---

## 07_results.md

```markdown
# 7. Results

## 7.1 Main Results

| Model | Baseline Acc. | Recovered Acc. | Delta | Peak VRAM | Latency |
|---|---:|---:|---:|---:|---:|
| DistilBERT | {{BASE_D}}% | {{ACC_D}}% | {{DELTA_D}} | {{VRAM_D}} | {{LAT_D}} |
| RoBERTa-base | {{BASE_R}}% | {{ACC_R}}% | {{DELTA_R}} | {{VRAM_R}} | {{LAT_R}} |
| GPT-2 | {{BASE_G}}% | {{ACC_G}}% | {{DELTA_G}} | {{VRAM_G}} | {{LAT_G}} |

## 7.2 Ablations

We vary rank, recovery steps, and gate initialization to understand stability and accuracy trade-offs.

## 7.3 Inference Cost Breakdown

We report memory reduction and runtime effects relative to the dense baseline under matched settings.
```

---

## 08_analysis.md

```markdown
# 8. Analysis

## 8.1 Why the Method Works

The method works because the module retains the pretrained attention geometry while replacing only the most expensive part of the computation. The gate learns when to emphasize local or global information while maintaining compatibility with the underlying model.

## 8.2 Failure Modes

The main failure modes are {{FAILURE_MODE_1}}, {{FAILURE_MODE_2}}, and {{FAILURE_MODE_3}}. These are most visible in settings with short sequences, strong causal constraints, or mismatched head configurations.

## 8.3 Cost Breakdown

The dominant cost in the original attention path is the quadratic attention matrix. The new path reduces this cost by shifting most work into a structured and lower-memory formulation.
```

---

## 09_discussion.md

```markdown
# 9. Discussion

{{MODULE_NAME}} is most useful when the deployment constraint is memory or long-context latency rather than raw task accuracy. It is less appropriate when the model is known to rely strongly on dense all-to-all interaction for short sequence tasks. In those cases, the local or low-rank branch may need a larger rank or more recovery steps.

The approach is intentionally simple: it keeps the base model intact, performs a narrow recovery stage, and validates compatibility across multiple transformer families. This makes it practical for iterative experimentation and real deployment.
```

---

## 10_limitations.md

```markdown
# 10. Limitations

This method does not eliminate the need for model-specific tuning. In some architectures, the interaction with RoPE or grouped-query attention may require additional calibration. The approach is also not universally optimal for very short sequences, where the overhead of decomposition or gating may outweigh the savings.
```

---

## 11_conclusion.md

```markdown
# 11. Conclusion

We introduced {{MODULE_NAME}}, a {{CATEGORY}} designed to reduce the attention bottleneck while preserving pretrained model behavior. The method remains compatible with multiple transformer architectures, requires only a lightweight recovery stage, and shows practical improvements in memory and inference cost under our benchmark settings.
```

---

## 12_reproducibility.md

```markdown
# 12. Reproducibility

All code and evaluation scripts are provided in the repository. To reproduce the results:

1. Install dependencies from `requirements.txt`.
2. Run the training script for the target backbone.
3. Run the evaluation script with the same seed and configuration file.
4. Validate the reported metrics against the benchmark output.
```

---

## 13_references.md

```markdown
# 13. References

- Vaswani, A., et al. (2017). Attention Is All You Need.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Beltagy, I., et al. (2020). Longformer: The Long-Document Transformer.
- Choromanski, K., et al. (2021). Rethinking Attention with Performers.
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Zhang, Y., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference.
```

---

## A_math.md

```markdown
# A. Mathematical Derivations

This appendix contains the derivations for the module's attention approximation, the fused gate update, and the complexity reduction arguments.
```

---

## B_implementation.md

```markdown
# B. Implementation Details

This appendix describes tensor shapes, masking rules, and compatibility logic for flat, nested, and causal architectures.
```

---

## C_data.md

```markdown
# C. Full Experimental Data

This appendix includes the complete per-model metrics, ablation tables, and aggregate benchmark results.
```

---

## D_negative_results.md

```markdown
# D. Negative Results

This appendix documents settings where the method failed to improve or where instability was observed, including the conditions and attempted fixes.
```

---

## E_ethics.md

```markdown
# E. Ethical Considerations

This work studies efficiency-oriented transformer modification for deployment under hardware constraints. It does not create new personal data or sensitive inference tasks. We note that efficiency gains can reduce energy usage but may also enable broader deployment of large models without addressing underlying misuse concerns.
```

---

## build.py

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_variables(path: str = "variables.json"):
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def render_section(section_name: str, values: dict) -> str:
    template = (ROOT / f"{section_name}.md").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def main():
    variables = load_variables()
    for section in [
        "00_meta",
        "01_intro",
        "02_related",
        "03_background",
        "04_method",
        "05_architecture",
        "06_setup",
        "07_results",
        "08_analysis",
        "09_discussion",
        "10_limitations",
        "11_conclusion",
        "12_reproducibility",
        "13_references",
        "A_math",
        "B_implementation",
        "C_data",
        "D_negative_results",
        "E_ethics",
    ]:
        render_section(section, variables)


if __name__ == "__main__":
    main()
```

---

## variables.json

```json
{
  "FULL_NAME": "Adaptive Sparse Recovery Attention",
  "MODULE_NAME": "ASRA",
  "SUBTITLE": "A Hybrid Attention Recovery Module for Efficient Transformers",
  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "your.name@example.edu",
  "AUTHOR_2_NAME": "Coauthor",
  "AUTHOR_2_EMAIL": "coauthor@example.edu",
  "UNIVERSITY_1": "Your University",
  "DEPARTMENT_1": "Department of Computer Science",
  "UNIVERSITY_2": "Your Institution",
  "DEPARTMENT_2": "Department of Electrical Engineering",
  "CORRESPONDING_AUTHOR": "Your Name",
  "DATE": "2026-09-15",
  "VERSION": "1.0",
  "CHANGELOG": "Initial draft",
  "ORG": "your-org",
  "HF_ORG": "your-hf-org",
  "LICENSE": "MIT",
  "CATEGORY": "efficient attention module",
  "CORE_BEHAVIOR_IN_ONE_SENTENCE": "reduces the attention bottleneck by combining local and global structure while preserving the pretrained model behavior",
  "COMPONENT_1": "a local windowed branch",
  "COMPONENT_2": "a kernelized global branch",
  "COMPONENT_3": "a gate-based fusion layer",
  "RECOVERY_STEPS": 600,
  "LORA_RANK": 8,
  "LORA_ALPHA": 16,
  "TASK": "SST-2 sentiment classification",
  "N_MODELS": 3,
  "N_ABLATIONS": 6,
  "ACC_D": 90.4,
  "BASE_D": 90.7,
  "DELTA_D": -0.3,
  "ACC_R": 89.9,
  "BASE_R": 90.5,
  "DELTA_R": -0.6,
  "ACC_G": 89.1,
  "BASE_G": 89.8,
  "DELTA_G": -0.7,
  "PARAM_CHANGE_PCT": 0.8,
  "PARAM_CHANGE_NOTE": "adapter-only increase after preserving base weights",
  "VRAM_CHANGE_PCT": 18,
  "LATENCY_CHANGE": 1.3,
  "LATENCY_REASON_SHORT": "the fusion gate and global branch add a small amount of extra computation",
  "WHAT_WORKS": "local-global fusion helps preserve task accuracy under reduced memory",
  "WHAT_FAILS": "very short sequences and mismatched RoPE assumptions can still destabilize the approximation",
  "FAILURE_CONDITION": "short sequences and strong causal pressure",
  "LAY_SUMMARY": "This paper introduces a lightweight attention module that keeps a model's pretrained behavior while cutting memory and compute costs. Instead of retraining the whole model, it adds a small recovery stage and a smart fusion between local and global information. The result is a practical way to make transformers more efficient without discarding the original design.",
  "TARGET_METRIC": "attention FLOPs and peak memory",
  "ACC_TOLERANCE": 2.0,
  "VRAM_TARGET": 8,
  "GPU_MODEL": "A100 40GB",
  "HIGH_LEVEL_DESCRIPTION": "reduces the attention bottleneck by replacing the expensive dense interaction with a more structured local-global formulation",
  "KEY_INSIGHT": "the pretrained attention can be preserved through a weight-copy and recovery procedure, while the computationally heavy branch is replaced with a tractable approximation",
  "BEHAVIOR_1": "preserves the base model while lowering attention cost through structured fusion",
  "DIFFERENTIATOR": "it is architecture-portable and recovery-driven rather than retraining-intensive",
  "ENGINEERING_1": "padding-safe masked attention",
  "ENGINEERING_2": "causal compatibility for GPT-style decoding",
  "ENGINEERING_3": "weight-copy and zero-init recovery for flat, nested, and causal models",
  "MODEL_LIST": "DistilBERT, RoBERTa-base, and GPT-2",
  "DATASET_NAME": "SST-2",
  "OPTIMIZER": "AdamW",
  "LR": "2e-4",
  "HYPERPARAM_3": "rank and gate initialization"
}
```

This template is ready to be copied into a module-specific paper folder and customized for the exact method you are writing about.

so the initial update $\Delta W = 0$.

### 4.8.3 {{MODULE_SPECIFIC_PARAMS}}

{{MODULE_SPECIFIC_INIT}}

## 4.9 Theoretical Analysis

### 4.9.1 Approximation Error

<!-- 
INSTRUCTIONS: If you have bounds, prove them. If not, say so.
-->

{{APPROX_ANALYSIS}}

### 4.9.2 Causality Preservation

For decoder models, {{MODULE_NAME}} maintains causality because {{CAUSAL_PROOF}}.

**Claim.** For any position $i$, the output $Y_i$ depends only on $X_1, 
\dots, X_i$.

**Proof sketch.** {{PROOF_SKETCH}} $\square$

### 4.9.3 Gradient Flow

{{GRADIENT_ANALYSIS}}

## 4.10 Implementation Notes

### 4.10.1 Numerical Stability

{{NUMERICAL_STABILITY_NOTES}}

### 4.10.2 Hardware Considerations

{{HARDWARE_NOTES}}

### 4.10.3 Memory Layout

{{MEMORY_NOTES}}
```

---

## 📄 `05_architecture.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: How {{MODULE_NAME}} integrates with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages (800 - 1200 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

{{MODULE_NAME}} supports the following attention variants:

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT-base | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa-base | `BertSelfAttention` | Yes | No | No | {{R_VERIFIED}} |
| GPT-2 | `GPT2Attention` | No | No | No | {{G_VERIFIED}} |
| Qwen3.5-0.8B | `Qwen3Attention` | No | Yes | Yes | {{Q_VERIFIED}} |

**Nested** means the attention module is wrapped by an outer projection 
(e.g., RoBERTa's `BertAttention` has both `self` and `output.dense`). 
**Flat** means attention includes its own output projection 
(e.g., DistilBERT's `out_lin`).

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

DistilBERT attention has four projections:

```python
class MultiHeadSelfAttention:
    q_lin: Linear(768, 768)
    k_lin: Linear(768, 768)
    v_lin: Linear(768, 768)
    out_lin: Linear(768, 768)
```

**Copy procedure.** Direct copy:

```python
target.q_proj.weight.copy_(src.q_lin.weight)
target.k_proj.weight.copy_(src.k_lin.weight)
target.v_proj.weight.copy_(src.v_lin.weight)
target.out_proj.weight.copy_(src.out_lin.weight)
```

**Nested encoder.** No. DistilBERT applies `out_lin` internally, so we set 
`is_nested_encoder = False` and apply our own `out_proj`.

### 5.2.2 RoBERTa (Nested)

RoBERTa attention is nested:

```python
class BertAttention:
    self: BertSelfAttention      # has query, key, value
    output: BertSelfOutput       # has dense (output projection)
```

**Copy procedure.** Extract from `self`:

```python
target.q_proj.weight.copy_(src.self.query.weight)
target.k_proj.weight.copy_(src.self.key.weight)
target.v_proj.weight.copy_(src.self.value.weight)
# out_proj is skipped — outer layer applies output.dense
```

**Nested encoder.** Yes. We set `is_nested_encoder = True`, so `forward` 
returns the raw context without applying `out_proj`. The outer `BertSelfOutput.dense` 
applies the projection.

### 5.2.3 GPT-2 (Fused Conv1D)

GPT-2 fuses Q, K, V into a single `c_attn` Conv1D:

```python
class GPT2Attention:
    c_attn: Conv1D(768, 2304)   # fused Q, K, V
    c_proj: Conv1D(768, 768)    # output
```

**Copy procedure.** Split fused weight:

```python
w = src.c_attn.weight.t()        # Conv1D stores (in, out) → (out, in)
q_w, k_w, v_w = w.chunk(3, dim=0)
target.q_proj.weight.copy_(q_w)
target.k_proj.weight.copy_(k_w)
target.v_proj.weight.copy_(v_w)
target.out_proj.weight.copy_(src.c_proj.weight.t())
```

**Nested encoder.** No. GPT-2 is causal but flat.

### 5.2.4 Qwen3.5 (GQA + RoPE)

Qwen3.5 has GQA with $H_{kv} = 2 < H = 8$:

```python
class Qwen3Attention:
    q_proj: Linear(1024, 2048)   # H * d_h = 8 * 256
    k_proj: Linear(1024, 512)    # H_kv * d_h = 2 * 256
    v_proj: Linear(1024, 512)
    o_proj: Linear(2048, 1024)
```

**Copy procedure.** Direct copy, then `repeat_kv` at runtime:

```python
target.q_proj.weight.copy_(src.q_proj.weight)
target.k_proj.weight.copy_(src.k_proj.weight)
target.v_proj.weight.copy_(src.v_proj.weight)
target.out_proj.weight.copy_(src.o_proj.weight)
```

**Nested encoder.** No.

## 5.3 Injection Algorithm

**Algorithm 2:** Inject {{MODULE_NAME}} into a full model
```
Input:  model M, config c
Output: M with all attention blocks replaced

 1: for ℓ in transformer_layers(M) do
 2:     src ← extract_attention(ℓ)
 3:     if src is None: continue
 4:     new ← {{MODULE_NAME}}(c, old=src)
 5:     new.copy_weights(src)
 6:     replace_attention(ℓ, new)
 7: end for
 8: return M
```

### 5.3.1 Handling Different Layer Paths

| Model | Layer path | Attention path |
|---|---|---|
| DistilBERT | `model.distilbert.transformer.layer[i]` | `.attention` |
| RoBERTa | `model.roberta.encoder.layer[i]` | `.attention` |
| GPT-2 | `model.transformer.h[i]` | `.attn` |
| Qwen3.5 | `model.model.layers[i]` | `.self_attn` |

## 5.4 Causality Preservation

For decoder models, {{MODULE_NAME}} maintains strict causality via:

1. **Local branch:** Causal band mask $k_j \le q_i$ and $q_i - k_j \leq W$.
2. **Global branch:** Prefix scan with cumulative state.

### 5.4.1 Verification

We verify causality by checking that the gradient of $Y_i$ w.r.t. $X_j$ 
is zero for $j > i$:

```python
X = torch.randn(1, L, d, requires_grad=True)
Y = model(X)
loss = Y[0, i].sum()
loss.backward()
assert X.grad[0, i+1:].abs().max() < 1e-6
```

{{CAUSALITY_RESULT}}

## 5.5 Numerical Precision

{{MODULE_NAME}} operates in FP16 for projections and attention scores but 
promotes to FP32 for {{FP32_OPS}}. This is necessary because {{FP32_REASON}}.

## 5.6 Error Handling and Fallbacks

<!-- 
INSTRUCTIONS: Describe what happens when things go wrong.
-->

| Failure | Detection | Fallback |
|---|---|---|
| Missing weight | `copy_weights` raises | Abort injection |
| Shape mismatch | Assertion fails | Skip layer |
| CUDA OOM | RuntimeError | Reduce batch |
| NaN in forward | `torch.isnan` check | Skip layer |

## 5.7 Compatibility Notes

### 5.7.1 Grouped-Query Attention

{{GQA_COMPAT}}

### 5.7.2 Rotary Position Embeddings

{{ROPE_COMPAT}}

### 5.7.3 Cross-Attention

{{CROSS_ATTN_COMPAT}}
```

---

## 📄 `06_setup.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup — models, data, hyperparameters, hardware
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages (600 - 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

We evaluate on three pretrained backbones spanning three architectural families:

| Model | Family | Layers | Hidden | Heads | KV heads | Params | Attention |
|---|---|---|---|---|---|---|---|
| DistilBERT-base | BERT (distilled) | 6 | 768 | 12 | 12 | 66.9M | Flat |
| RoBERTa-base | BERT (robust) | 12 | 768 | 12 | 12 | 124.6M | Nested |
| GPT-2 | GPT (causal) | 12 | 768 | 12 | 12 | 124.4M | Causal |

All models use `num_labels=2` for SST-2. Base weights are initialized from 
the finetuned checkpoints in `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

### 6.2.1 SST-2

The Stanford Sentiment Treebank (Socher et al., 2013) is a binary sentiment 
classification task.

| Split | Examples | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 (50.2%) | 33,542 |
| Validation | 872 | 444 (50.9%) | 428 |
| Test | 1,821 | — | — |

**Label distribution.** Class balance is near 50/50 on validation, giving a 
**majority baseline of 50.92%**.

**Sequence length.** We truncate to 128 tokens.

**Tokenization.** Standard HuggingFace tokenizers for each model family.

### 6.2.2 Why SST-2?

- Small (evaluates in seconds on RTX 4070)
- Standard for Transformer evaluation
- Binary (clear failure detection via majority baseline)
- Well-understood difficulty (typical accuracy 78-95%)

## 6.3 Training Protocol

### 6.3.1 Recovery Fine-Tuning

After injection, we freeze the {{MODULE_NAME}} module and train only LoRA 
adapters:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| LR schedule | {{SCHEDULE}} |
| Warmup | {{WARMUP}} steps |
| Batch size | {{BATCH_SIZE}} |
| Gradient accumulation | {{GRAD_ACCUM}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| LoRA rank $r$ | {{LORA_RANK}} |
| LoRA $\alpha$ | {{LORA_ALPHA}} |
| Precision | FP16 |
| Dropout | {{DROPOUT}} |
| Max sequence length | 128 |

### 6.3.2 Head Fine-Tuning

Before injection, the classifier head is fine-tuned for 200 steps at 
learning rate $10^{-3}$ to ensure the head matches the base model. This is 
standard practice when loading a base model with a randomly initialized head.

### 6.3.3 Loss Function

We use standard cross-entropy:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)
\tag{6.1}
$$

No regularization is added unless noted in Appendix C.

## 6.4 Evaluation

### 6.4.1 Metrics

- **Accuracy:** $\text{Acc} = \frac{\text{correct}}{\text{total}}$
- **F1 (binary):** $\frac{2 P R}{P + R}$
- **F1 (macro):** Average over both classes
- **Precision, Recall:** Per class
- **Validation loss:** Cross-entropy

### 6.4.2 Efficiency Metrics

- **Total params:** `sum(p.numel() for p in model.parameters())`
- **Trainable params:** `sum(p.numel() for p in model.parameters() if p.requires_grad)`
- **Peak VRAM:** `torch.cuda.max_memory_allocated() / 1e6` MB
- **Inference time:** Median of 20 forward passes, batch size 8, sequence 128
- **Throughput:** tokens/sec = $B \cdot L / t$

### 6.4.3 Quality Gate

We define three pass levels:

| Level | Threshold | Meaning |
|---|---|---|
| **PASS** | $\geq 56.0\%$ | Statistically above chance |
| **PASS_LOW** | $50.92\% \leq$ acc $< 56.0\%$ | Above majority but not significant |
| **FAIL** | $< 50.92\%$ | Below majority |

The 56% threshold is majority baseline + 3×SE where SE $= \sqrt{p(1-p)/N} 
\approx 1.69\%$.

## 6.5 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| GPU VRAM | 8 GB GDDR6 |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |
| CPU | {{CPU}} |
| RAM | {{RAM}} |
| OS | {{OS}} |

## 6.6 Statistical Significance

All numbers are from a single seed ({{SEED}}) unless noted. With $N = 872$ 
validation examples, the standard error is $\sim 1.69\%$. We do not run 
multiple seeds for the main results due to compute constraints.

**Limitations.** Single-seed results may vary by $\pm 1\%$ on reruns. We 
report this transparently and encourage replications.

## 6.7 Baselines

We compare against:

1. **`full`** — the original pretrained model with no modification.
2. **`{{MODULE_NAME}}`** — our method.
3. **Prior methods** (from our larger benchmark suite):
   - `nha`, `nasc`, `kda`, `aegis_v5`, etc.

Direct comparison with published methods (Performer, Linformer, etc.) is 
deferred to future work.
```

---

## 📄 `07_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results — main tables, ablations
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages (1000 - 1500 words + tables)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1** reports accuracy on SST-2 validation for all three models. 
We include the baseline (`full`), our method (`{{MODULE_NAME}}`), and the 
delta.

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Prec | Rec | Loss |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0.9075 | 0.9068 | 0.303 |
| DistilBERT | **{{MODULE}}** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_P}} | {{D_R}} | {{D_LOSS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 0.8051 | 0.7832 | 0.438 |
| RoBERTa | **{{MODULE}}** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_P}} | {{R_R}} | {{R_LOSS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 0.7935 | 0.7935 | 0.488 |
| GPT-2 | **{{MODULE}}** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_P}} | {{G_R}} | {{G_LOSS}} |

**Key observations:**

1. {{OBS_1}}
2. {{OBS_2}}
3. {{OBS_3}}

## 7.2 Parameter Analysis

<!-- 
CRITICAL: Be honest about parameter counts.
-->

**Table 7.2: Parameter counts before and after injection.**

| Model | Baseline params | After | Δ Absolute | Δ % | Note |
|---|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | {{D_P_ABS}} | {{D_P_PCT}} | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | {{R_P_ABS}} | {{R_P_PCT}} | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | {{G_P_ABS}} | {{G_P_PCT}} | {{G_NOTE}} |

**Interpretation.** {{PARAM_INTERPRETATION}}

<!-- 
Example honest statements:
- "Parameter count does not decrease because base weights are preserved."
- "Parameter count increases by 0.89% due to LoRA adapters."
- "Parameter count decreases by 42% because attention projections are factorized."
-->

## 7.3 Efficiency Profile

**Table 7.3: Efficiency comparison.**

| Metric | Baseline | Ours | Ratio | Note |
|---|---|---|---|---|
| Inference time (DistilBERT) | 1.90 s | {{D_INF}} s | {{D_INF_R}}× | {{D_INF_NOTE}} |
| Inference time (GPT-2) | 4.20 s | {{G_INF}} s | {{G_INF_R}}× | {{G_INF_NOTE}} |
| Peak VRAM (DistilBERT) | 183 MB | {{D_VRAM}} MB | {{D_VRAM_R}}× | {{D_VRAM_NOTE}} |
| Peak VRAM (GPT-2) | 793 MB | {{G_VRAM}} MB | {{G_VRAM_R}}× | {{G_VRAM_NOTE}} |
| Throughput (tokens/s) | {{BASE_TPS}} | {{OUR_TPS}} | {{TPS_R}}× | — |

## 7.4 Ablation: LoRA Rank

**Table 7.4: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 | Trainable params |
|---|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} | {{TP_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} | {{TP_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} | {{TP_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} | {{TP_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} | {{TP_R64}} |

**Observation.** {{RANK_OBSERVATION}}

## 7.5 Ablation: Recovery Steps

**Table 7.5: Accuracy vs recovery steps (LoRA rank = 16).**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |
| 200 | {{D_S200}} | {{R_S200}} | {{G_S200}} |

**Observation.** {{STEPS_OBSERVATION}}

## 7.6 Ablation: {{HYPERPARAM_3}}

**Table 7.6: Effect of {{HYPERPARAM_3}}.**

| Value | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| {{V1}} | {{D_V1}} | {{R_V1}} | {{G_V1}} |
| {{V2}} | {{D_V2}} | {{R_V2}} | {{G_V2}} |
| {{V3}} | {{D_V3}} | {{R_V3}} | {{G_V3}} |
| {{V4}} | {{D_V4}} | {{R_V4}} | {{G_V4}} |

## 7.7 Ablation: {{HYPERPARAM_4}}

**Table 7.7: Effect of {{HYPERPARAM_4}}.**

| Value | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| {{V1}} | {{D_V1}} | {{R_V1}} | {{G_V1}} |
| {{V2}} | {{D_V2}} | {{R_V2}} | {{G_V2}} |
| {{V3}} | {{D_V3}} | {{R_V3}} | {{G_V3}} |

## 7.8 Scaling with Sequence Length

<!-- 
INSTRUCTIONS: Only if you have multiple sequence lengths.
-->

**Table 7.8: Inference time vs sequence length (GPT-2).**

| Seq len | Baseline | Ours | Speedup |
|---|---|---|---|
| 64 | {{B64}} s | {{O64}} s | {{S64}}× |
| 128 | {{B128}} s | {{O128}} s | {{S128}}× |
| 256 | {{B256}} s | {{O256}} s | {{S256}}× |
| 512 | {{B512}} s | {{O512}} s | {{S512}}× |

**Observation.** {{SCALING_OBSERVATION}}

## 7.9 Training Curves

**Figure 7.1** shows the recovery loss curves. For DistilBERT, loss decreases 
from `{{D_START_LOSS}}` to `{{D_END_LOSS}}` over {{STEPS}} steps. For 
RoBERTa, the loss **oscillates** between `{{R_MIN_LOSS}}` and `{{R_MAX_LOSS}}`, 
suggesting {{ROBERTA_LOSS_INTERPRETATION}}.

<!-- Attach loss-vs-step plot for all three models. -->

## 7.10 Qualitative Analysis

### 7.10.1 Attention Patterns

**Figure 7.2** visualizes attention patterns for baseline vs ours. We observe:

- {{PATTERN_OBS_1}}
- {{PATTERN_OBS_2}}
- {{PATTERN_OBS_3}}

### 7.10.2 Example Predictions

| Input | Baseline | Ours | Ground truth |
|---|---|---|---|
| "{{EX_1}}" | {{B1}} | {{O1}} | {{GT1}} |
| "{{EX_2}}" | {{B2}} | {{O2}} | {{GT2}} |
| "{{EX_3}}" | {{B3}} | {{O3}} | {{GT3}} |

## 7.11 Summary

**What works:**

- {{WORKS_1}}
- {{WORKS_2}}

**What doesn't:**

- {{FAILS_1}}
- {{FAILS_2}}

See Section 8 for detailed analysis of failures.
```

---

## 📄 `08_analysis.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Why does it work? Why does it fail?
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages (1000 - 1500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why Does It Work on DistilBERT?

DistilBERT shows only {{D_DELTA}}% drop while RoBERTa shows {{R_DELTA}}%. 
We investigate why.

### 8.1.1 Hypothesis 1: Fewer Layers

DistilBERT has 6 layers vs RoBERTa's 12. Fewer layers mean:

- Less compounding of approximation error
- Fewer attention blocks to replace
- Smaller total activation memory

**Test.** {{TEST_1}}

**Result.** {{RESULT_1}}

### 8.1.2 Hypothesis 2: Flat Architecture

DistilBERT is flat — its `out_lin` is inside the attention module. RoBERTa is 
nested — its output projection is outside.

**Test.** {{TEST_2}}

**Result.** {{RESULT_2}}

### 8.1.3 Hypothesis 3: {{H3}}

**Test.** {{TEST_3}}

**Result.** {{RESULT_3}}

## 8.2 Why Does It Fail on RoBERTa?

<!-- 
INSTRUCTIONS:
- Detailed analysis
- List hypotheses
- Show tests
- Be honest if you don't know
-->

{{R_FAILURE_INTRO}}

### 8.2.1 Hypothesis 1: Nested Encoder Mis-Set

If `is_nested_encoder` is set incorrectly, the output projection is applied 
twice.

**Test.** We instrumented the forward pass to check the output norm:

```python
with torch.no_grad():
    out = model.attention(...)
    print('context norm:', out.norm().item())
    print('expected norm:', context_baseline.norm().item())
```

**Result.** {{NESTED_TEST_RESULT}}

### 8.2.2 Hypothesis 2: LoRA Rank Insufficient

RoBERTa has 12 layers vs DistilBERT's 6. The same rank must cover 2× more 
attention blocks.

**Test.** We ran RoBERTa with `lora_rank=32` instead of 16:

| Rank | Accuracy | Δ from baseline |
|---|---|---|
| 8 | {{R_R8}} | {{R_R8_D}} |
| 16 | {{R_R16}} | {{R_R16_D}} |
| 32 | {{R_R32}} | {{R_R32_D}} |
| 64 | {{R_R64}} | {{R_R64_D}} |

**Result.** {{LORA_RANK_RESULT}}

### 8.2.3 Hypothesis 3: Attention Sink Loss

If {{MODULE_NAME}} does not preserve attention sinks, RoBERTa may lose 
stability.

**Test.** We added an explicit sink-preservation term. See Appendix D.2.

**Result.** {{SINK_RESULT}}

### 8.2.4 Hypothesis 4: Weight Copy Issue

Maybe `copy_weights` silently fails for RoBERTa's nested structure.

**Test.** We inspected the copied weights:

```python
for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
    w = getattr(new_attn, proj).weight
    print(proj, w.shape, w.norm().item(), w.isnan().any().item())
```

**Result.** {{WEIGHT_COPY_RESULT}}

### 8.2.5 Hypothesis 5: Unknown

{{UNKNOWN_DISCUSSION}}

## 8.3 Inference Cost Breakdown

The 5× slowdown on GPT-2 comes from four sources:

### 8.3.1 Kernel Launch Overhead

| Operation | Baseline launches | Ours launches |
|---|---|---|
| Q/K/V projections | 3 | 3 |
| Local attention | 1 | {{LAUNCHES_LOCAL}} |
| Global attention | 0 | {{LAUNCHES_GLOBAL}} |
| Gate | 0 | 2 |
| Output projection | 1 | 1 |
| **Total** | **5** | **{{TOTAL_LAUNCHES}}** |

Each kernel launch has ~5-10 μs overhead. For {{TOTAL_LAUNCHES}} launches, 
this is {{KERNEL_OVERHEAD}} ms per forward.

### 8.3.2 Python Loop Overhead

`_local_window_attention` uses a Python for-loop over sequence chunks:

```python
for i in range(0, N, chunk):
    ...
```

For $N = 128$ and `chunk = 64`, this is 2 iterations × 5 ops each = 10 
Python-level operations per head. Across 12 heads × 12 layers, this 
compounds.

### 8.3.3 FP16 → FP32 Cast

LoRA branch promotes to FP32:

```python
F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B)
#                     ^^^^^^^ FP16 → FP32
```

Each cast involves a kernel launch and memory copy.

### 8.3.4 Sequential Matmuls

Global attention requires 4 sequential matmuls:

1. `phi_Q @ KV` — $B \times H \times L \times m \times d$
2. `phi_Q @ Z^T` — $B \times H \times L \times m$
3. Division
4. Padding multiply

## 8.4 The Compression Illusion

<!-- 
CRITICAL SECTION: Be explicit about what "compression" means.
-->

Despite the term "attention compression", **{{MODULE_NAME}} does not reduce 
parameter count**. From Table 7.2:

- Baseline: {{BASE_PARAMS}}
- Ours: {{OUR_PARAMS}}
- Change: {{PARAM_CHANGE}}%

The label "compressed" refers only to {{WHAT_COMPRESSED}}. In our case:

- ✅ **VRAM during inference** decreases by {{VRAM_DEC}}%
- ❌ **Total parameters** increase by {{PARAM_INC}}%
- ❌ **Inference latency** increases by {{LAT_INC}}×

**Why?** {{WHY_NO_COMPRESSION}}

## 8.5 Where the Gains Come From

We ablate each component:

| Configuration | Accuracy | Δ |
|---|---|---|
| Baseline (no change) | {{BASE}} | — |
| Only weight transfer | {{WT_ACC}} | {{WT_D}} |
| + LoRA recovery | {{LORA_ACC}} | {{LORA_D}} |
| + {{COMPONENT_1}} | {{C1_ACC}} | {{C1_D}} |
| + {{COMPONENT_2}} | {{C2_ACC}} | {{C2_D}} |
| Full {{MODULE_NAME}} | {{FULL_ACC}} | {{FULL_D}} |

**Interpretation.** {{CONTRIB_INTERPRETATION}}

## 8.6 Failure Mode Taxonomy

| Failure Mode | Symptom | Root cause | Fix |
|---|---|---|---|
| **Collapse** | Acc ≈ 50.92% | Forward pass broken | Fix weight copy |
| **OOM** | CUDA error | Activation memory | Reduce batch |
| **Slow** | >5× latency | Kernel overhead | Fuse kernels |
| **No-op** | Δ = 0.00% | No real replacement | Fix injection |
| **Degrade** | Δ < -5% | Insufficient recovery | More steps |
| **NaN** | Loss = NaN | Overflow/underflow | FP32 |

## 8.7 Comparison to Prior Modules

| Module | DistilBERT | RoBERTa | GPT-2 | Category |
|---|---|---|---|---|
| `full` | 90.71% | 78.56% | 79.36% | Baseline |
| `nha` | 90.25% | FAIL | 78.78% | Hybrid |
| `nasc` | 90.71% | 64.56% | 78.44% | Neural |
| `nasc_attention` | 90.71% | 67.89% | 78.44% | Neural |
| `hamming` | 79.13% | 50.92% | 71.22% | Hashing |
| `galerkin` | 65.94% | 48.97% | 79.36% | Galerkin |
| **`{{MODULE}}`** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | {{CATEGORY}} |

## 8.8 Sensitivity Analysis

### 8.8.1 Effect of Batch Size

| Batch size | DistilBERT acc | Training time |
|---|---|---|
| 2 | {{BS2}} | {{BS2_T}} |
| 4 | {{BS4}} | {{BS4_T}} |
| 8 | {{BS8}} | {{BS8_T}} |
| 16 | {{BS16}} | {{BS16_T}} |

### 8.8.2 Effect of Learning Rate

| LR | DistilBERT acc | Loss stability |
|---|---|---|
| 1e-5 | {{LR1e5}} | {{LR1e5_STAB}} |
| 5e-5 | {{LR5e5}} | {{LR5e5_STAB}} |
| 1e-4 | {{LR1e4}} | {{LR1e4_STAB}} |
| 2e-4 | {{LR2e4}} | {{LR2e4_STAB}} |
| 5e-4 | {{LR5e4}} | {{LR5e4_STAB}} |

## 8.9 Statistical Notes

With $N = 872$, standard error is 1.69%. Differences below 3.4% (2×SE) are 
within noise. We therefore distinguish:

- **Large effects (>5%):** statistically meaningful
- **Small effects (<3%):** need multiple seeds to confirm
- **Exact matches:** suspicious, likely no-op

We run single-seed for the main results due to compute constraints.
```

---

## 📄 `09_discussion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpret results — when to use, when not to
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages (600 - 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: {{TAKEAWAY_1_TITLE}}

{{TAKEAWAY_1_BODY}}

### 9.1.2 Takeaway 2: {{TAKEAWAY_2_TITLE}}

{{TAKEAWAY_2_BODY}}

### 9.1.3 Takeaway 3: {{TAKEAWAY_3_TITLE}}

{{TAKEAWAY_3_BODY}}

## 9.2 When to Use {{MODULE_NAME}}

### 9.2.1 Recommended

- **{{USE_1}}:** {{USE_1_WHY}}
- **{{USE_2}}:** {{USE_2_WHY}}
- **{{USE_3}}:** {{USE_3_WHY}}

### 9.2.2 Not Recommended

- **{{AVOID_1}}:** {{AVOID_1_WHY}}
- **{{AVOID_2}}:** {{AVOID_2_WHY}}
- **{{AVOID_3}}:** {{AVOID_3_WHY}}

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Does the target model have ≤6 layers? ─── Yes ─── Use {{MODULE_NAME}}
  │
  No
  ▼
Do you need to reduce parameter count? ─── Yes ─── DON'T use {{MODULE_NAME}}
  │
  No
  ▼
Is inference latency critical? ─── Yes ─── DON'T use {{MODULE_NAME}}
  │
  No
  ▼
Is VRAM the bottleneck? ─── Yes ─── Use {{MODULE_NAME}}
  │
  No
  ▼
Consider other methods
```

## 9.4 Cost-Benefit Analysis

For a DistilBERT deployment on RTX 4070:

| Metric | Baseline | {{MODULE_NAME}} | Trade-off |
|---|---|---|---|
| Accuracy | 90.71% | {{D_ACC}}% | {{D_ACC_DELTA}} |
| Params | 66.95M | {{D_PARAMS}} | {{D_PARAM_DELTA}} |
| Inference VRAM | 183 MB | {{D_VRAM}} MB | {{D_VRAM_DELTA}} |
| Inference time | 1.90 s | {{D_TIME}} s | {{D_TIME_DELTA}} |
| Model size on disk | {{D_DISK_B}} MB | {{D_DISK_M}} MB | {{D_DISK_DELTA}} |

**Verdict.** {{VERDICT}}

## 9.5 Comparison to Published Methods

We don't run direct comparison with Performer/Linformer/Longformer, but we 
can compare theoretically:

| Method | Complexity | Approx. error | Trains base? |
|---|---|---|---|
| Standard MHSA | $\mathcal{O}(L^2)$ | 0 | — |
| Performer | $\mathcal{O}(L)$ | {{PERF_ERR}} | Yes |
| Linformer | $\mathcal{O}(L)$ | {{LIN_ERR}} | Yes |
| Longformer | $\mathcal{O}(L\sqrt{L})$ | 0 (exact) | Yes |
| H2O | $\mathcal{O}(L)$ | {{H2O_ERR}} | No |
| **{{MODULE_NAME}}** | $\mathcal{O}({{COMPLEX}})$ | {{OUR_ERR}} | **No** |

## 9.6 Deployment Considerations

### 9.6.1 Memory

{{DEPLOY_MEMORY}}

### 9.6.2 Latency

{{DEPLOY_LATENCY}}

### 9.6.3 Batching

{{DEPLOY_BATCHING}}

## 9.7 Open Questions

1. **{{OQ_1}}** — {{OQ_1_DESC}}
2. **{{OQ_2}}** — {{OQ_2_DESC}}
3. **{{OQ_3}}** — {{OQ_3_DESC}}

## 9.8 Broader Impact

### 9.8.1 Positive

{{POSITIVE_IMPACT}}

### 9.8.2 Negative

{{NEGATIVE_IMPACT}}

### 9.8.3 Environmental

Training {{MODULE_NAME}} on 3 models × 50 recovery steps × RTX 4070 
consumes approximately {{CO2_KG}} kg CO₂eq (estimated via ML CO2 Impact 
calculator).
```

---

## 📄 `10_limitations.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations — reviewers expect this
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page (300 - 600 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

We acknowledge the following limitations of our work.

## 10.1 Task Coverage

We only evaluate on **SST-2**, a binary sentiment classification task. Our 
method may behave differently on:

- **Generation tasks** (summarization, translation)
- **Multi-class classification** (MNLI, AG News)
- **Token-level tasks** (NER, POS tagging)
- **Retrieval tasks** (dense passage retrieval)

We make no claim about generalization beyond SST-2.

## 10.2 Model Coverage

We test on three models with ≤12 layers. We do not test:

- **Large models** (>1B parameters): {{LARGE_MODEL_STATUS}}
- **Deep models** (>24 layers): {{DEEP_MODEL_STATUS}}
- **Mixture-of-Experts**: {{MOE_STATUS}}
- **Vision transformers**: Not tested
- **Multimodal transformers**: Not tested

Our method may not scale as expected.

## 10.3 Hardware Coverage

All experiments are on **RTX 4070 Laptop (8 GB)**. We do not test:

- **Datacenter GPUs** (A100, H100): May have different efficiency profile
- **Consumer GPUs** (RTX 3060, 4080): Different VRAM budgets
- **TPUs**: Not tested
- **CPU inference**: Not tested
- **Edge devices** (Jetson, mobile): Not tested

## 10.4 Inference Overhead

{{INFERENCE_LIMITATION}}

<!-- 
Example: "Inference latency increases by 5× on GPT-2 and 4× on DistilBERT 
due to the lack of fused kernels. Future work should implement fused Triton 
kernels for the local and global branches."
-->

## 10.5 Parameter Count

{{PARAM_LIMITATION}}

<!-- 
Example: "Parameter count does not decrease because base weights are 
preserved. The 'compression' label refers only to VRAM during inference, 
not to model size on disk."
-->

## 10.6 Recovery Stability

{{RECOVERY_LIMITATION}}

<!-- 
Example: "RoBERTa's loss oscillates during recovery, indicating the 
optimization landscape is unstable. We did not resolve this; future work 
should investigate learning rate schedules or orthogonal regularization."
-->

## 10.7 The 56% Quality Gate

We set the PASS threshold at 56% (majority + 3·SE) to separate meaningful 
from noise. This is a **practical** threshold, not a theoretical one. A 
method achieving 55% may be meaningfully better than baseline but fails our 
gate.

## 10.8 Single Seed

All main results are from a single random seed ({{SEED}}). We do not report 
confidence intervals because we could not afford multiple runs. Differences 
below 3.4% (2×SE) may be noise.

## 10.9 Comparison with Published Methods

We do not run head-to-head comparisons with Performer, Linformer, Longformer, 
or H2O. Our analysis is therefore limited to:

- Internal comparisons (`full` vs `{{MODULE_NAME}}`)
- Theoretical complexity analysis
- Prior-module comparisons within our framework

## 10.10 Theoretical Guarantees

We provide no formal approximation error bound. Our method is empirical. 
The absence of theory limits our ability to predict behavior on unseen 
tasks or models.

## 10.11 Failure Modes We Did Not Fully Diagnose

{{UNRESOLVED_FAILURES}}

## 10.12 Code Quality

Our implementation is research-grade, not production-grade. Known issues:

- No unit tests for attention modules
- No fused kernels for GPU acceleration
- Some error paths raise exceptions instead of logging
- `copy_weights` silently succeeds for some architectures

We encourage the community to help harden the implementation.
```

---

## 📄 `11_conclusion.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 11_conclusion.md
     PURPOSE: Wrap up — summary, key takeaways, future work
     DEPENDENCIES: 10_limitations.md
     LENGTH TARGET: 0.5 - 1 page (300 - 500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 11. Conclusion

## 11.1 Summary

We presented **{{FULL_NAME}}** (**{{MODULE_NAME}}**), a {{CATEGORY}} that 
{{CORE_BEHAVIOR}}. The method combines {{COMPONENT_1}}, {{COMPONENT_2}}, and 
{{COMPONENT_3}}, requiring no retraining of base weights.

## 11.2 Results Recap

| Model | Accuracy | Δ | Params Δ | VRAM Δ | Latency Δ |
|---|---|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_DELTA}} | {{D_P_DELTA}} | {{D_VRAM_DELTA}} | {{D_LAT_DELTA}} |
| RoBERTa | {{R_ACC}}% | {{R_DELTA}} | {{R_P_DELTA}} | {{R_VRAM_DELTA}} | {{R_LAT_DELTA}} |
| GPT-2 | {{G_ACC}}% | {{G_DELTA}} | {{G_P_DELTA}} | {{G_VRAM_DELTA}} | {{G_LAT_DELTA}} |

## 11.3 Key Takeaways

1. **{{KT_1}}**
2. **{{KT_2}}**
3. **{{KT_3}}**

## 11.4 Future Work

### 11.4.1 Short-Term

- **Fused kernels:** Implement Triton kernels for local + global attention.
  Expected speedup: 2-3×.
- **Fix RoBERTa:** Investigate nested-encoder mis-set. Hypothesis: output 
  projection applied twice.
- **Multi-seed runs:** Report confidence intervals.

### 11.4.2 Long-Term

- **Task generalization:** Evaluate on MNLI, SQuAD, XSum.
- **Model scaling:** Test on Qwen3.5-7B, LLaMA-2-7B, Mistral-7B.
- **Theory:** Derive approximation bounds for the {{COMPONENT_1}} branch.
- **Fusion with other methods:** Combine with KV-cache quantization.

### 11.4.3 Methodological

- **Standardized benchmark:** Contribute to a common benchmark for attention 
  modules.
- **Reproducibility study:** Independent replication on different hardware.

## 11.5 Closing Remarks

{{CLOSING_REMARKS}}

<!-- 
Example: "We hope this work contributes to a broader conversation about 
what 'compression' means in practice — VRAM, parameters, latency, or 
something else. We release all code and checkpoints to enable honest 
replication and incremental improvement."
-->

## 11.6 Acknowledgments

We thank {{ACKNOWLEDGMENTS}} for helpful discussions and feedback. This 
work was supported by {{FUNDING}}.

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed the method, implemented the code, ran experiments, wrote the paper.
- **{{AUTHOR_2}}** provided feedback on the method and edited the manuscript.

## 11.8 Conflict of Interest

The authors declare no competing interests.
```

---

## 📄 `12_reproducibility.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Everything needed to reproduce the results
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page (500 - 700 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment Setup

### 12.1.1 Prerequisites

- Python ≥ 3.10
- CUDA ≥ 11.8
- 8 GB GPU VRAM minimum
- 20 GB free disk space

### 12.1.2 Installation

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 12.1.3 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |
| huggingface_hub | {{HF_HUB_VERSION}} |
| numpy | {{NUMPY_VERSION}} |
| pandas | {{PANDAS_VERSION}} |

## 12.2 Running the Experiments

### 12.2.1 Full Benchmark

```bash
python run_all_4_{{module}}.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}} \
    --batch_size {{BATCH_SIZE}}
```

**Expected runtime:** ~{{RUNTIME}} minutes on RTX 4070.

### 12.2.2 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --dataset_dir neuralops/data/glue_data/sst2 \
    --output_dir results/unified_{{module}}/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,{{module}} \
    --batch_size {{BATCH_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --learning_rate {{LR}} \
    --lora_rank {{LORA_RANK}} \
    --rank {{RANK}} \
    --precision fp16 \
    --fresh \
    --defer_export \
    --timeout 180
```

## 12.3 Downloading Checkpoints

### 12.3.1 From Hugging Face

```python
from huggingface_hub import hf_hub_download

# DistilBERT
path = hf_hub_download(
    repo_id="{{HF_ORG}}/svd-linear-attention-distilbert-{{module}}",
    filename="weights.pt"
)

# RoBERTa
path = hf_hub_download(
    repo_id="{{HF_ORG}}/svd-linear-attention-roberta-{{module}}",
    filename="weights.pt"
)

# GPT-2
path = hf_hub_download(
    repo_id="{{HF_ORG}}/svd-linear-attention-gpt2-{{module}}",
    filename="weights.pt"
)
```

### 12.3.2 Loading Locally

```python
import torch

# Load checkpoint
checkpoint = torch.load("weights.pt", map_location="cpu")

# Apply to model
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
```

## 12.4 Reproducing Individual Tables

### 12.4.1 Table 7.1 (Main Results)

```bash
python run_all_4_{{module}}.py --models distilbert,roberta,gpt2
# → results/unified_{{module}}/{model}/summary.csv
```

### 12.4.2 Table 7.4 (LoRA Rank Ablation)

```bash
for rank in 4 8 16 32 64; do
    python run_all_4_{{module}}.py \
        --models distilbert,roberta,gpt2 \
        --lora_rank $rank \
        --output_suffix rank_$rank
done
```

## 12.5 Random Seeds

| Source | Seed | Purpose |
|---|---|---|
| PyTorch | {{TORCH_SEED}} | Global RNG |
| NumPy | {{NUMPY_SEED}} | Data shuffling |
| CUDA | {{CUDA_SEED}} | Dropout, init |
| DataLoader | {{DL_SEED}} | Batch order |

To reproduce exactly:

```python
import random
import numpy as np
import torch

seed = {{SEED}}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 12.6 Expected Results

After running the full benchmark, you should see:

| Model | Accuracy | Status |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | {{D_STATUS}} |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_STATUS}} |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_STATUS}} |

Deviations greater than 2% indicate environment differences. Report issues 
to {{ISSUE_URL}}.

## 12.7 Known Reproducibility Issues

- **Windows encoding:** `cp1256` codec errors may occur. Set `PYTHONIOENCODING=utf-8`.
- **CUDA version mismatch:** Ensure CUDA ≥ 11.8 for FP16.
- **Checkpoint paths:** Some baseline checkpoints may be missing. Fall back 
  to the model name (e.g., `distilbert-base-uncased`).

## 12.8 Data Availability

- **SST-2:** Available via HuggingFace `datasets` or the GLUE benchmark.
- **Checkpoints:** Available at `{{HF_ORG}}/svd-linear-attention-*`.
- **Raw metrics:** Available at `results/unified_{{module}}/*/summary.csv`.
- **Training logs:** Available at `benchmark_logs/{{module}}_*.log`.
```

---

## 📄 `13_references.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 13_references.md
     PURPOSE: Complete bibliography
     DEPENDENCIES: All sections
     ═══════════════════════════════════════════════════════════════════════════ -->

# 13. References

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and 
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and 
          Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@inproceedings{hu2021lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
          Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
          Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{choromanski2021rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and 
          Dohan, David and Song, Xingyou and Gane, Andreea and 
          Sarlos, Tamas and Hawkins, Peter and Davis, Jared and 
          Mohiuddin, Afroz and Kaiser, Lukasz and others},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}

@inproceedings{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and 
          Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and 
          Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and 
          others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}

@inproceedings{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and Li, Belinda Z and Khabsa, Madian and 
          Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}

@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNNs}: Fast autoregressive transformers with 
         linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and 
          Fleuret, Fran{\c{c}}ois},
  booktitle={International Conference on Machine Learning},
  year={2020}
}

@inproceedings{xiong2021nystromformer,
  title={Nystr{\"o}mformer: A Nystr{\"o}m-based algorithm for 
         approximating self-attention},
  author={Xiong, Yunyang and Zeng, Zhanpeng and Chakraborty, Rudrasis and 
          Tan, Mingxing and Fung, Glenn and Li, Yin and Singh, Vikas},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2021}
}

@inproceedings{zhang2023h2o,
  title={{H2O}: Heavy-hitter oracle for efficient generative inference 
         of large language models},
  author={Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and 
          Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and 
          R{\'e}, Christopher and Barrett, Clark and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@inproceedings{xiao2024efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and 
          Han, Song and Lewis, Mike},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{liu2024kivi,
  title={{KIVI}: A tuning-free asymmetric 2bit quantization for {KV} cache},
  author={Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and 
          Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  booktitle={International Conference on Machine Learning},
  year={2024}
}

@article{su2021roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and 
          Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}

@inproceedings{sochor2013recursive,
  title={Recursive deep models for semantic compositionality over a 
         sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and 
          Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and 
          Potts, Christopher},
  booktitle={Conference on Empirical Methods in Natural Language Processing},
  year={2013}
}

@inproceedings{sanh2019distilbert,
  title={{DistilBERT}, a distilled version of {BERT}: smaller, faster, 
         cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and 
          Wolf, Thomas},
  booktitle={NeurIPS EMC² Workshop},
  year={2019}
}

@inproceedings{liu2019roberta,
  title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and 
          Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and 
          Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}

@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and 
          Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Blog},
  year={2019}
}

@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@inproceedings{zhang2023adalora,
  title={{AdaLoRA}: Parameter-efficient fine-tuning with adaptive budget allocation},
  author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and 
          He, Pengcheng and Cheng, Yu and Chen, Weizhu and Zhao, Tuo},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{liu2024dora,
  title={{DoRA}: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and 
          Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and 
          Chen, Min-Hung},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Citation Format

We use a numeric citation style: `[1]`, `[2]`, etc. References are ordered 
by appearance in the text.

For BibTeX, use:

```bibtex
@article{key,
  author = {...},
  title = {...},
  journal = {...},
  year = {...},
  volume = {...},
  pages = {...}
}
```
```

---

## 📄 `A_math.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Formal proofs and derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Notation

We restate notation used throughout the derivations.

| Symbol | Meaning |
|---|---|
| $d$ | Hidden size |
| $L$ | Sequence length |
| $H$ | Query head count |
| $d_h$ | Per-head dimension |
| $m$ | RFF feature count |
| $\phi$ | Feature map |
| $\sigma$ | Softmax function |

## A.2 Derivation of the Global Attention

### A.2.1 Standard Softmax Attention

Standard attention:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_{j} e^{q_i \cdot k_j / \sqrt{d_h}} v_j}{\sum_{j} e^{q_i \cdot k_j / \sqrt{d_h}}}
\tag{A.1}
$$

### A.2.2 Kernel Approximation

We approximate $e^{q \cdot k}$ via random features:

$$
e^{q \cdot k} \approx \phi(q)^\top \phi(k)
\tag{A.2}
$$

where $\phi(x) = \frac{1}{\sqrt{m}} \left( e^{\omega_1 \cdot x}, \dots, e^{\omega_m \cdot x} \right)^\top$ 
and $\omega_i \sim \mathcal{N}(0, I)$.

### A.2.3 Linear Attention

Substituting (A.2) into (A.1):

$$
\text{Attn}(Q, K, V)_i \approx \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{A.3}
$$

Define:

$$
KV = \sum_j \phi(k_j) v_j^\top, \qquad
Z = \sum_j \phi(k_j)
\tag{A.4}
$$

Then:

$$
\text{Attn}(Q, K, V)_i \approx \frac{\phi(q_i)^\top KV}{\phi(q_i)^\top Z}
\tag{A.5}
$$

**Complexity.** Computing $KV$ and $Z$ is $\mathcal{O}(L m d_h)$, and 
evaluating (A.5) is $\mathcal{O}(L m d_h)$. Total: $\mathcal{O}(L m d_h)$, 
linear in $L$.

## A.3 Causality Preservation

**Claim.** For decoder models, the output at position $i$ depends only on 
inputs $X_1, \dots, X_i$.

**Proof.** Let $S_i$ denote the cumulative state after processing tokens 
$1..i$:

$$
S_i = \sum_{j \le i} \phi(k_j) v_j^\top
\tag{A.6}
$$

By definition of the prefix scan, $S_i$ depends only on $K_1, \dots, K_i$ and 
$V_1, \dots, V_i$. Similarly, $Z_i = \sum_{j \le i} \phi(k_j)$.

The output at position $i$ is:

$$
Y_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top Z_i}
\tag{A.7}
$$

Since $q_i$ depends only on $X_i$, and $S_i, Z_i$ depend only on 
$X_1, \dots, X_i$, we conclude $Y_i = f(X_1, \dots, X_i)$. $\square$

## A.4 Approximation Error Bound

<!-- 
INSTRUCTIONS: If you have a bound, state it. If not, say so.
-->

We provide no formal approximation bound. The RFF approximation error 
depends on the number of features $m$, the data distribution, and the 
choice of kernel. Standard results (Rahimi & Recht, 2008) give a uniform 
bound:

$$
\sup_{x, y} |\phi(x)^\top \phi(y) - e^{x \cdot y}| \le \epsilon
\tag{A.8}
$$

with $m = \mathcal{O}(d \log(1/\epsilon^2))$, but constants are loose and 
do not directly bound attention output error.

## A.5 Weight Transfer Correctness

**Claim.** After `copy_weights`, the module's base weights equal the 
pretrained attention weights (for the corresponding projections).

**Proof.** We handle three cases:

### A.5.1 DistilBERT (Flat)

By direct copy:

$$
W_Q^{\text{new}} = W_{q\_lin}^{\text{old}}, \quad
W_K^{\text{new}} = W_{k\_lin}^{\text{old}}, \quad
W_V^{\text{new}} = W_{v\_lin}^{\text{old}}, \quad
W_O^{\text{new}} = W_{out\_lin}^{\text{old}}
\tag{A.9}
$$

### A.5.2 RoBERTa (Nested)

Extracted from inner module:

$$
W_Q^{\text{new}} = W_{\text{self.query}}^{\text{old}}, \quad
W_K^{\text{new}} = W_{\text{self.key}}^{\text{old}}, \quad
W_V^{\text{new}} = W_{\text{self.value}}^{\text{old}}
\tag{A.10}
$$

$W_O$ remains identity (outer `output.dense` applies projection).

### A.5.3 GPT-2 (Fused Conv1D)

Split fused weight:

$$
W_{\text{attn}}^{\text{old}} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix} 
\quad \text{(after transpose)}
\tag{A.11}
$$

Then copy each block. $\square$

## A.6 Orthogonality of RFF Projection

<!-- 
INSTRUCTIONS: If applicable to your method.
-->

For the RFF projection matrix $\Omega \in \mathbb{R}^{d_h \times m}$, we 
enforce column orthogonality via QR:

$$
\Omega = Q R, \qquad Q^\top Q = I_m
\tag{A.12}
$$

This ensures $\phi(q)^\top \phi(k)$ has the correct scale 
$\approx \frac{d_h}{m}$.
```

---

## 📄 `B_implementation.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code-level details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 Module Structure

```python
class {{MODULE_CLASS}}(BaseAttention):
    """{{FULL_NAME}} attention module."""
    
    def __init__(
        self,
        config: Any,
        rank: int = 64,
        window_size: int = 64,
        lora_rank: int = 16,
        **kwargs
    ):
        super().__init__(config, rank)
        
        # Resolve dimensions
        self.dim = kwargs.get("hidden_size", 768)
        self.num_heads = kwargs.get("num_heads", 12)
        self.head_dim = self.dim // self.num_heads
        
        # Projections with LoRA
        self.q_proj = LoRALinear(self.dim, self.all_head_size, ...)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, ...)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, ...)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, ...)
        
        # {{MODULE_SPECIFIC}}
        ...
        
        # Weight transfer
        if old_attn is not None:
            self.copy_weights(old_attn)
```

## B.2 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, **kwargs):
    B, N, D = hidden_states.shape
    
    # 1. Project
    Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
    K = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim)
    V = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim)
    Q = Q.transpose(1, 2)  # (B, H, N, d_h)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    
    # 2. Position encoding
    if position_embeddings is not None:
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    
    # 3. GQA
    if self.num_kv_heads < self.num_heads:
        K = repeat_kv(K, self.kv_repeat)
        V = repeat_kv(V, self.kv_repeat)
    
    # 4. Compute attention
    local_out = self._local_attention(Q, K, V, ...)
    global_out = self._global_attention(Q, K, V, ...)
    
    # 5. Fuse
    gate = torch.sigmoid(self.gate(hidden_states)).unsqueeze(1)
    out = gate * local_out + (1 - gate) * global_out
    
    # 6. Output projection
    out = out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
    
    if self.is_nested_encoder:
        return out, None
    
    return self.out_proj(out), None
```

## B.3 Weight Transfer Details

### B.3.1 Handling `Conv1D` (GPT-2)

```python
# GPT-2 stores weights as (in_features, out_features)
# Standard PyTorch Linear stores (out_features, in_features)
if isinstance(src.c_attn, Conv1D):
    w = src.c_attn.weight.t()  # transpose to (out, in)
else:
    w = src.c_attn.weight

# Split into Q, K, V
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### B.3.2 Handling Nested Encoders (RoBERTa)

```python
if hasattr(src, "self") and hasattr(src, "output"):
    # Nested — extract from inner module
    self.is_nested_encoder = True
    q_w = src.self.query.weight
    k_w = src.self.key.weight
    v_w = src.self.value.weight
    # out_proj is NOT extracted — outer layer applies it
```

### B.3.3 Handling GQA (Qwen)

```python
# Qwen has H_kv < H
if hasattr(config, "num_key_value_heads"):
    num_kv_heads = config.num_key_value_heads
else:
    num_kv_heads = num_heads

# Projections have different shapes
q_out = num_heads * head_dim
kv_out = num_kv_heads * head_dim
```

## B.4 Initialization

```python
def reset_parameters(self):
    # LoRA A: Kaiming uniform
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    # LoRA B: zeros (so ΔW = 0 at start)
    nn.init.zeros_(self.lora_B)
    
    # Gate: bias toward local attention
    nn.init.zeros_(self.gate.weight)
    nn.init.constant_(self.gate.bias, 3.0)
```

## B.5 Numerical Precision

- **Projections:** FP16 (matches model dtype)
- **Attention scores:** FP32 (numerical stability)
- **Softmax:** FP32
- **LoRA:** FP32 (for training stability)
- **Final output:** FP16

## B.6 Common Pitfalls

### B.6.1 Forgetting to Freeze Base Weights

```python
# WRONG
self.q_proj.weight.requires_grad = True

# RIGHT (in copy_weights)
self.q_proj.freeze_base_weights()  # sets requires_grad=False
```

### B.6.2 Wrong `is_nested_encoder`

```python
# For DistilBERT (flat):
self.is_nested_encoder = False  # Apply out_lin internally

# For RoBERTa (nested):
self.is_nested_encoder = True  # Skip out_proj; outer layer applies it
```

### B.6.3 Missing GQA Handling

```python
# WRONG: assume H_kv == H
K = K.view(B, N, H, d_h)

# RIGHT:
K = K.view(B, N, self.num_kv_heads, self.head_dim)
if self.num_kv_heads < H:
    K = repeat_kv(K, H // self.num_kv_heads)
```

## B.7 Full Source Code

The complete source is at:

```
framework/attentions/{{module_file}}.py
```

Key entry points:

- `{{MODULE_CLASS}}` — the attention module
- `inject_{{module}}()` — model-level injection
- `LoRALinear` — reused from `framework.attentions.{{base_module}}`

## B.8 Registry

```python
@ATTENTION_REGISTRY.register("{{module}}")
@ATTENTION_REGISTRY.register("{{module}}_attention")
class {{MODULE_CLASS}}(BaseAttention):
    ...
```

## B.9 Diagnostics

To verify injection worked:

```python
from framework.attentions.{{module_file}} import inject_{{module}}

model = DistilBertForSequenceClassification.from_pretrained("...")
inject_{{module}}(model)

# Verify attention class
for name, mod in model.named_modules():
    if "attention" in name and name.count(".") <= 5:
        print(name, type(mod).__name__)
```

Expected output:

```
distilbert.transformer.layer.0.attention -> {{MODULE_CLASS}}
distilbert.transformer.layer.1.attention -> {{MODULE_CLASS}}
...
```
```

---

## 📄 `C_data.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Complete numerical results, per-layer stats
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 Complete Metrics (All Models)

### C.1.1 DistilBERT

| Metric | Value |
|---|---|
| Module | {{D_MODULE}} |
| Model | DistilBERT-base |
| Accuracy | {{D_ACC}}% |
| F1 (binary) | {{D_F1}} |
| F1 (macro) | {{D_F1_MACRO}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Validation loss | {{D_LOSS}} |
| Total params (before) | 66,955,010 |
| Total params (after) | {{D_PARAMS}} |
| Trainable (recovery) | {{D_TRAINABLE}} |
| Compression rate | {{D_RATE}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Tokens/sec | {{D_TPS}} |
| Compressed layers | {{D_COMPRESSED}} |
| Fallback layers | {{D_FALLBACK}} |
| Avg reconstruction error | {{D_AVG_ERR}} |
| Path | {{D_PATH}} |
| Timestamp | {{D_TS}} |

### C.1.2 RoBERTa

| Metric | Value |
|---|---|
| Module | {{R_MODULE}} |
| Model | RoBERTa-base |
| Accuracy | {{R_ACC}}% |
| F1 (binary) | {{R_F1}} |
| F1 (macro) | {{R_F1_MACRO}} |
| Precision | {{R_PREC}} |
| Recall | {{R_REC}} |
| Validation loss | {{R_LOSS}} |
| Total params (before) | 124,647,170 |
| Total params (after) | {{R_PARAMS}} |
| Trainable (recovery) | {{R_TRAINABLE}} |
| Compression rate | {{R_RATE}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |
| Tokens/sec | {{R_TPS}} |
| Compressed layers | {{R_COMPRESSED}} |
| Fallback layers | {{R_FALLBACK}} |
| Avg reconstruction error | {{R_AVG_ERR}} |
| Path | {{R_PATH}} |
| Timestamp | {{R_TS}} |
| Status | {{R_STATUS}} |
| Error (if any) | {{R_ERROR}} |

### C.1.3 GPT-2

| Metric | Value |
|---|---|
| Module | {{G_MODULE}} |
| Model | GPT-2 |
| Accuracy | {{G_ACC}}% |
| F1 (binary) | {{G_F1}} |
| F1 (macro) | {{G_F1_MACRO}} |
| Precision | {{G_PREC}} |
| Recall | {{G_REC}} |
| Validation loss | {{G_LOSS}} |
| Total params (before) | 124,441,344 |
| Total params (after) | {{G_PARAMS}} |
| Trainable (recovery) | {{G_TRAINABLE}} |
| Compression rate | {{G_RATE}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |
| Tokens/sec | {{G_TPS}} |
| Compressed layers | {{G_COMPRESSED}} |
| Fallback layers | {{G_FALLBACK}} |
| Avg reconstruction error | {{G_AVG_ERR}} |
| Path | {{G_PATH}} |
| Timestamp | {{G_TS}} |
| Status | {{G_STATUS}} |

## C.2 Per-Layer Reconstruction Error

### C.2.1 DistilBERT

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} | {{D_L2_O}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} | {{D_L4_O}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} |

### C.2.2 RoBERTa

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | {{R_L0_Q}} | {{R_L0_K}} | {{R_L0_V}} | {{R_L0_O}} |
| 1 | {{R_L1_Q}} | {{R_L1_K}} | {{R_L1_V}} | {{R_L1_O}} |
| ... | ... | ... | ... | ... |
| 11 | {{R_L11_Q}} | {{R_L11_K}} | {{R_L11_V}} | {{R_L11_O}} |

### C.2.3 GPT-2

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | {{G_L0_Q}} | {{G_L0_K}} | {{G_L0_V}} | {{G_L0_O}} |
| ... | ... | ... | ... | ... |

## C.3 Loss Curves

### C.3.1 DistilBERT

```
Step  Loss
1     {{D_S1_LOSS}}
5     {{D_S5_LOSS}}
10    {{D_S10_LOSS}}
15    {{D_S15_LOSS}}
20    {{D_S20_LOSS}}
25    {{D_S25_LOSS}}
30    {{D_S30_LOSS}}
35    {{D_S35_LOSS}}
40    {{D_S40_LOSS}}
45    {{D_S45_LOSS}}
50    {{D_S50_LOSS}}
```

### C.3.2 RoBERTa

```
Step  Loss
1     {{R_S1_LOSS}}
5     {{R_S5_LOSS}}
10    {{R_S10_LOSS}}
...
50    {{R_S50_LOSS}}
```

### C.3.3 GPT-2

```
Step  Loss
1     {{G_S1_LOSS}}
...
50    {{G_S50_LOSS}}
```

## C.4 Ablation: LoRA Rank

### C.4.1 Detailed Table

| Rank | Model | Accuracy | Trainable | Peak VRAM | Time |
|---|---|---|---|---|---|
| 4 | DistilBERT | {{D_R4_ACC}} | {{D_R4_TP}} | {{D_R4_VRAM}} | {{D_R4_T}} |
| 8 | DistilBERT | {{D_R8_ACC}} | {{D_R8_TP}} | {{D_R8_VRAM}} | {{D_R8_T}} |
| 16 | DistilBERT | {{D_R16_ACC}} | {{D_R16_TP}} | {{D_R16_VRAM}} | {{D_R16_T}} |
| 32 | DistilBERT | {{D_R32_ACC}} | {{D_R32_TP}} | {{D_R32_VRAM}} | {{D_R32_T}} |
| 64 | DistilBERT | {{D_R64_ACC}} | {{D_R64_TP}} | {{D_R64_VRAM}} | {{D_R64_T}} |

(Repeat for RoBERTa and GPT-2)

## C.5 Ablation: Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |
| 200 | {{D_S200}} | {{R_S200}} | {{G_S200}} |

## C.6 Per-Batch Statistics

For the last epoch of training:

| Batch | Loss | Accuracy |
|---|---|---|
| 1 | {{B1_LOSS}} | {{B1_ACC}} |
| 2 | {{B2_LOSS}} | {{B2_ACC}} |
| ... | ... | ... |
| 50 | {{B50_LOSS}} | {{B50_ACC}} |

## C.7 Hardware Utilization

### C.7.1 GPU Utilization

| Phase | GPU % | VRAM % | Power (W) |
|---|---|---|---|
| Idle | 0 | 5 | 15 |
| Loading | 30 | 20 | 60 |
| Injection | 45 | 35 | 80 |
| Recovery | 95 | 85 | 120 |
| Evaluation | 60 | 40 | 90 |

### C.7.2 Memory Breakdown

| Component | Baseline | {{MODULE_NAME}} |
|---|---|---|
| Model weights | {{BASE_WEIGHTS}} MB | {{OUR_WEIGHTS}} MB |
| Activations | {{BASE_ACT}} MB | {{OUR_ACT}} MB |
| Optimizer state | {{BASE_OPT}} MB | {{OUR_OPT}} MB |
| Gradients | {{BASE_GRAD}} MB | {{OUR_GRAD}} MB |
| **Total** | **{{BASE_TOTAL}}** | **{{OUR_TOTAL}}** |

## C.8 Failure Case Examples

### C.8.1 Failed Predictions

| Input | Ground truth | Baseline | Ours |
|---|---|---|---|
| "{{FAIL_1}}" | {{FAIL_1_GT}} | {{FAIL_1_B}} | {{FAIL_1_O}} |
| "{{FAIL_2}}" | {{FAIL_2_GT}} | {{FAIL_2_B}} | {{FAIL_2_O}} |
| "{{FAIL_3}}" | {{FAIL_3_GT}} | {{FAIL_3_B}} | {{FAIL_3_O}} |

### C.8.2 Failure Categories

| Category | Count | Examples |
|---|---|---|
| Negation | {{NEG_COUNT}} | "not good" → positive |
| Sarcasm | {{SAR_COUNT}} | "oh great" → negative |
| Long input | {{LONG_COUNT}} | Truncated at 128 tokens |
| Rare words | {{RARE_COUNT}} | Tokenization errors |

## C.9 Full Result Logs

Raw logs are at:

```
benchmark_logs/{{module}}/
├── distilbert.log
├── roberta.log
└── gpt2.log
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_{{module}}.py --models distilbert,roberta,gpt2

# Individual runs
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,{{module}} \
    --recovery_steps 50 \
    --lora_rank 16 \
    --batch_size 8 \
    --precision fp16
```
```

---

## 📄 `D_negative_results.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What we tried that didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

We document our failed experiments to help future work avoid dead ends.

## D.1 What We Tried That Failed

### D.1.1 Attempt 1: {{FAIL_1_NAME}}

**Goal.** {{FAIL_1_GOAL}}

**Method.** {{FAIL_1_METHOD}}

**Result.** {{FAIL_1_RESULT}}

**Why it failed.** {{FAIL_1_WHY}}

### D.1.2 Attempt 2: {{FAIL_2_NAME}}

**Goal.** {{FAIL_2_GOAL}}

**Method.** {{FAIL_2_METHOD}}

**Result.** {{FAIL_2_RESULT}}

**Why it failed.** {{FAIL_2_WHY}}

### D.1.3 Attempt 3: {{FAIL_3_NAME}}

**Goal.** {{FAIL_3_GOAL}}

**Method.** {{FAIL_3_METHOD}}

**Result.** {{FAIL_3_RESULT}}

**Why it failed.** {{FAIL_3_WHY}}

## D.2 Abandoned Variants

### D.2.1 {{ABANDONED_1}}

We tested {{ABANDONED_1_DESC}} but abandoned it because {{ABANDONED_1_WHY}}.

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| {{AB1_CFG1}} | {{AB1_D1}} | {{AB1_R1}} | {{AB1_G1}} |
| {{AB1_CFG2}} | {{AB1_D2}} | {{AB1_R2}} | {{AB1_G2}} |

### D.2.2 {{ABANDONED_2}}

{{ABANDONED_2_DETAIL}}

## D.3 Failed Fixes for Known Bugs

### D.3.1 Fix for RoBERTa OOM

**Symptom.** CUDA OOM on RoBERTa (12 layers) but not DistilBERT (6 layers).

**Attempted fixes:**

| Fix | Result |
|---|---|
| Reduce batch size to 4 | Still OOM |
| Reduce batch size to 2 | Still OOM |
| Reduce `lora_rank` to 8 | Still OOM |
| Reduce `recovery_steps` to 25 | Still OOM |
| Gradient checkpointing | {{GC_RESULT}} |
| CPU offload | {{OFFLOAD_RESULT}} |

**Conclusion.** {{OOM_CONCLUSION}}

### D.3.2 Fix for Slow Inference

**Symptom.** 5× slowdown on all models.

**Attempted fixes:**

| Fix | Speedup |
|---|---|
| Remove FP32 cast | {{FIX_CAST}} |
| Vectorize local attention loop | {{FIX_VEC}} |
| Fused kernel for RFF | {{FIX_KERNEL}} |
| Batched matmuls | {{FIX_BATCHED}} |

**Conclusion.** {{SLOW_CONCLUSION}}

## D.4 Failed Hyperparameter Sweeps

### D.4.1 Learning Rate

| LR | DistilBERT acc | Loss stability |
|---|---|---|
| 1e-5 | {{D_LR1e5}} | Stable |
| 5e-5 | {{D_LR5e5}} | Stable |
| 1e-4 | {{D_LR1e4}} | Stable |
| 2e-4 | {{D_LR2e4}} | Stable |
| 5e-4 | {{D_LR5e4}} | Unstable |
| 1e-3 | {{D_LR1e3}} | NaN |

### D.4.2 Batch Size

| BS | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_BS1}} | {{R_BS1}} | {{G_BS1}} |
| 2 | {{D_BS2}} | {{R_BS2}} | {{G_BS2}} |
| 4 | {{D_BS4}} | {{R_BS4}} | {{G_BS4}} |
| 8 | {{D_BS8}} | {{R_BS8}} | {{G_BS8}} |
| 16 | {{D_BS16}} | OOM | OOM |

## D.5 Sink Preservation Attempt

**Motivation.** Attention sinks are known to be critical (Xiao et al., 2024). 
We hypothesized that failing to preserve them causes RoBERTa's drop.

**Implementation.** We added a loss term:

$$
\mathcal{L}_{\text{sink}} = \left\| \sum_{j=1}^{k} \alpha_j - c \right\|^2
\tag{D.1}
$$

where $\alpha_j$ are attention weights to the first $k$ tokens and $c$ is a 
target (e.g., 0.5).

**Result.** {{SINK_RESULT}}

**Conclusion.** {{SINK_CONCLUSION}}

## D.6 Orthogonal Regularization

**Motivation.** We hypothesized that LoRA factors might collapse to low-rank 
subspaces.

**Implementation.** We added:

$$
\mathcal{L}_{\text{orth}} = \| A A^\top - I \|_F^2 + \| B^\top B - I \|_F^2
\tag{D.2}
$$

**Result.** {{ORTH_RESULT}}

**Conclusion.** {{ORTH_CONCLUSION}}

## D.7 Multi-Seed Runs

We ran 3 seeds on DistilBERT to estimate variance:

| Seed | Accuracy | Δ from mean |
|---|---|---|
| 42 | {{SEED42}} | {{SEED42_D}} |
| 123 | {{SEED123}} | {{SEED123_D}} |
| 456 | {{SEED456}} | {{SEED456_D}} |
| **Mean** | **{{MEAN_ACC}}** | — |
| **Std** | **{{STD_ACC}}** | — |

**Conclusion.** {{SEED_CONCLUSION}}

## D.8 Things We Didn't Try (And Why)

| Idea | Why not tried |
|---|---|
| Performer-style features | Out of scope |
| FlashAttention-3 | Needs Hopper GPU |
| Full fine-tuning | Compute budget |
| Larger models (>1B) | VRAM limits |
| Multi-GPU training | Single GPU setup |

## D.9 Lessons Learned

1. **{{LESSON_1}}**
2. **{{LESSON_2}}**
3. **{{LESSON_3}}**

## D.10 Recommendations for Future Work

Based on our negative results, we recommend:

1. **{{REC_1}}**
2. **{{REC_2}}**
3. **{{REC_3}}**
```

---

## 📄 `E_ethics.md`

```markdown
<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical and societal considerations
     DEPENDENCIES: 11_conclusion.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations

## E.1 Broader Impact

### E.1.1 Positive Impacts

{{POSITIVE_IMPACT_1}}

{{POSITIVE_IMPACT_2}}

{{POSITIVE_IMPACT_3}}

### E.1.2 Negative Impacts

{{NEGATIVE_IMPACT_1}}

{{NEGATIVE_IMPACT_2}}

### E.1.3 Dual-Use Concerns

{{DUAL_USE}}

## E.2 Environmental Impact

### E.2.1 Training Cost

Our experiments used:

- **GPU:** NVIDIA RTX 4070 Laptop
- **Total training time:** ~{{TOTAL_HOURS}} hours
- **Estimated CO₂:** {{CO2_KG}} kg CO₂eq

Estimated using the [ML CO2 Impact Calculator](https://mlco2.github.io/impact/).

### E.2.2 Comparison

| Method | Training time | CO₂ (kg) |
|---|---|---|
| Full fine-tuning (est.) | 40 GPU-hours | 3.2 |
| LoRA (ours) | {{TRAIN_HOURS}} GPU-hours | {{CO2_KG}} |
| **Reduction** | **{{CO2_REDUCTION}}%** | **{{CO2_REDUCTION}}%** |

### E.2.3 Recommendations

We recommend:

1. Reusing pretrained checkpoints
2. Using LoRA instead of full fine-tuning
3. Running experiments during low-carbon hours
4. Reporting CO₂ in papers

## E.3 Data Ethics

### E.3.1 Dataset

We use **SST-2**, a public sentiment dataset. It contains:

- Movie reviews from Rotten Tomatoes
- Binary sentiment labels (positive / negative)
- 67K training examples

**Concerns:**

- **English only:** Method not validated on other languages
- **Western-centric:** Movie reviews may not represent global sentiment
- **Historical bias:** Reviews from 2005-2013 may reflect outdated views

### E.3.2 Privacy

No personally identifiable information (PII) is used. All reviews are 
public.

## E.4 Model Ethics

### E.4.1 Bias

We did not perform bias evaluation. Sentiment classifiers can exhibit:

- **Gender bias:** Different predictions on gendered pronouns
- **Race bias:** Different predictions on dialectal text
- **Age bias:** Different predictions on slang

Future work should audit {{MODULE_NAME}} for these biases.

### E.4.2 Misuse

{{MODULE_NAME}} could be misused for:

- Mass surveillance
- Manipulation of public opinion
- Automated censorship

We do not endorse these uses.

## E.5 Authorship and Credit

### E.5.1 Author Contributions

See Section 11.7.

### E.5.2 Data and Code

All code and checkpoints are released under **{{LICENSE}}**. This permits:

- ✅ Academic use
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution

With:

- ⚠️ Attribution required
- ⚠️ No warranty

## E.6 Conflicts of Interest

{{COI}}

## E.7 Funding

{{FUNDING}}

## E.8 Acknowledgements

{{ACKNOWLEDGMENTS}}

## E.9 Institutional Review

This work does not involve human subjects. No IRB approval was required.

## E.10 LLM Usage Disclosure

<!-- 
INSTRUCTIONS: Be transparent about LLM usage in writing.
-->

Parts of this paper were drafted with the assistance of large language 
models ({{LLM_NAMES}}). All technical content, experimental design, and 
final text were reviewed and edited by the authors.

## E.11 Reproducibility Ethics

We commit to:

1. Releasing all code and checkpoints
2. Reporting all results (including failures)
3. Documenting all hyperparameters
4. Responding to replication attempts

## E.12 Long-Term Considerations

Efficient attention mechanisms could:

- **Democratize AI:** Enable deployment on consumer hardware
- **Enable edge AI:** Allow offline, private inference
- **Reduce carbon footprint:** Lower energy per inference

However, efficiency improvements can also lead to:

- **Jevons paradox:** More usage at lower cost
- **Increased surveillance:** Cheaper to deploy everywhere
- **Consolidation:** Larger models on cheaper hardware

We encourage the community to consider these trade-offs.
```

---

## 📄 `build.py`

```python
"""
Build script: concatenate all sections into a single paper.md.

Usage:
    python build.py --module nha
    python build.py --module nha --output paper_nha.md
    python build.py --module nha --pdf  # also generate PDF if pandoc installed
"""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    "00_meta.md",
    "01_intro.md",
    "02_related.md",
    "03_background.md",
    "04_method.md",
    "05_architecture.md",
    "06_setup.md",
    "07_results.md",
    "08_analysis.md",
    "09_discussion.md",
    "10_limitations.md",
    "11_conclusion.md",
    "12_reproducibility.md",
    "13_references.md",
    "A_math.md",
    "B_implementation.md",
    "C_data.md",
    "D_negative_results.md",
    "E_ethics.md",
]


def build(module_dir: Path, output: Path, to_pdf: bool = False):
    """Concatenate all section files into one paper."""
    parts = []
    missing = []
    
    for section in SECTIONS:
        path = module_dir / section
        if not path.exists():
            missing.append(section)
            continue
        content = path.read_text(encoding="utf-8")
        parts.append(content)
        parts.append("\n\n---\n\n")
    
    if missing:
        print(f"Warning: {len(missing)} missing sections:")
        for m in missing:
            print(f"  - {m}")
    
    full = "".join(parts)
    output.write_text(full, encoding="utf-8")
    print(f"Wrote {len(full):,} chars to {output}")
    
    if to_pdf:
        try:
            subprocess.run([
                "pandoc", str(output),
                "-o", str(output.with_suffix(".pdf")),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
            ], check=True)
            print(f"Wrote PDF to {output.with_suffix('.pdf')}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("pandoc not available. Skipping PDF generation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Module name (e.g., nha)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    
    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Module directory not found: {module_dir}")
        return
    
    output = Path(args.output) if args.output else module_dir / "paper_full.md"
    build(module_dir, output, to_pdf=args.pdf)


if __name__ == "__main__":
    main()
```

---

## 📄 `variables.json`

```json
{
  "_comment": "Fill this file, then run `python apply_vars.py` to substitute all {{PLACEHOLDERS}} in every .md file of this module.",

  "MODULE_NAME": "NHA",
  "FULL_NAME": "Native Hybrid Attention",
  "SUBTITLE": "A Hybrid Local–Global Attention Mechanism",
  "CATEGORY": "hybrid attention",

  "AUTHOR_1_NAME": "Your Name",
  "AUTHOR_1_EMAIL": "you@university.edu",
  "AUTHOR_2_NAME": "Advisor Name",
  "AUTHOR_2_EMAIL": "advisor@university.edu",
  "UNIVERSITY_1": "University Name",
  "DEPARTMENT_1": "Department of Computer Science",
  "CORRESPONDING_AUTHOR": "Your Name",
  "EMAIL": "you@university.edu",
  "DATE": "2026-09-15",
  "VERSION": "1",
  "CHANGELOG": "initial draft",
  "ORG": "your-org",
  "HF_ORG": "Hooshaai",
  "LICENSE": "MIT",

  "ACC_D": "90.25",
  "BASE_D": "90.71",
  "DELTA_D": "-0.46",
  "ACC_R": "0.00",
  "BASE_R": "78.56",
  "DELTA_R": "FAIL",
  "ACC_G": "78.78",
  "BASE_G": "79.36",
  "DELTA_G": "-0.58",

  "PARAM_CHANGE_PCT": "+0.89",
  "PARAM_CHANGE_NOTE": "due to LoRA adapters",
  "VRAM_CHANGE_PCT": "-28",
  "LATENCY_CHANGE": "5",
  "LATENCY_REASON_SHORT": "kernel launch overhead",
  "LATENCY_REASON": "Python loops and missing fused kernels",

  "RECOVERY_STEPS": "50",
  "LORA_RANK": "16",
  "LORA_ALPHA": "32",
  "BATCH_SIZE": "8",
  "LR": "2e-4",
  "WD": "1e-2",
  "SEED": "42",

  "R_VERIFIED": "❌ OOM",
  "G_VERIFIED": "✅",
  "Q_VERIFIED": "not tested",

  "_note": "Add any other placeholders your paper needs"
}
```

---

## 📄 `apply_vars.py`

```python
"""
Apply variables.json to all .md files in the module directory.

Usage:
    python apply_vars.py --module nha
    python apply_vars.py --module nha --dry-run
"""

import argparse
import json
import re
from pathlib import Path


PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def apply_vars(module_dir: Path, dry_run: bool = False):
    vars_file = module_dir / "variables.json"
    if not vars_file.exists():
        print(f"variables.json not found in {module_dir}")
        return
    
    variables = json.loads(vars_file.read_text(encoding="utf-8"))
    # Remove comment keys
    variables = {k: v for k, v in variables.items() if not k.startswith("_")}
    
    print(f"Loaded {len(variables)} variables from {vars_file}")
    print()
    
    for md_file in sorted(module_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        
        missing = set()
        
        def replace(match):
            key = match.group(1)
            if key in variables:
                return str(variables[key])
            missing.add(key)
            return match.group(0)  # leave as-is
        
        new_content = PLACEHOLDER_PATTERN.sub(replace, content)
        
        if new_content != content:
            if dry_run:
                print(f"[DRY] Would update {md_file.name}")
            else:
                md_file.write_text(new_content, encoding="utf-8")
                print(f"Updated {md_file.name}")
        
        if missing:
            print(f"  Missing vars in {md_file.name}: {sorted(missing)}")
    
    print()
    if dry_run:
        print("Dry run complete. Run without --dry-run to apply.")
    else:
        print("Done. Review the files and rebuild with `python build.py --module <name>`.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    module_dir = Path(__file__).parent / args.module
    if not module_dir.exists():
        print(f"Module directory not found: {module_dir}")
        return
    
    apply_vars(module_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

---

## 📄 `checklist.md`

```markdown
# Pre-Submission Checklist — {{MODULE_NAME}}

## Before You Start

- [ ] Copied `_TEMPLATE/` to `{{MODULE_NAME}}/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module {{MODULE_NAME}}`
- [ ] Verified no `{{PLACEHOLDER}}` remains: `grep -r "{{" {{MODULE_NAME}}/`

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Every number matches Section 7
- [ ] If Δ Acc is negative, stated in abstract
- [ ] If params increased, stated in abstract
- [ ] If latency increased, stated in abstract

### Introduction
- [ ] Motivation is concrete (not "transformers are popular")
- [ ] Problem statement is formal
- [ ] Contributions are verifiable bullets
- [ ] Paper organization is complete

### Related Work
- [ ] Covers sparse, low-rank, kernel, hybrid attention
- [ ] Covers KV-cache compression
- [ ] Covers LoRA / PEFT
- [ ] Includes a "position of our work" section
- [ ] All citations are real (not hallucinated)

### Method
- [ ] Every formula is numbered
- [ ] Every formula is referenced in text
- [ ] Complexity analysis is complete
- [ ] Parameter analysis is honest
- [ ] Causality proof (if applicable)

### Results
- [ ] Table 7.1 (main results) is complete
- [ ] Table 7.2 (params) is honest
- [ ] Table 7.3 (efficiency) is complete
- [ ] All ablations are present
- [ ] Training curves are plotted

### Analysis
- [ ] Explains why it works on some models
- [ ] Explains why it fails on others
- [ ] Inference cost breakdown is detailed
- [ ] "Compression illusion" section is honest

### Limitations
- [ ] Task coverage limitation stated
- [ ] Model coverage limitation stated
- [ ] Hardware coverage limitation stated
- [ ] Inference overhead stated
- [ ] Single-seed limitation stated

## Reproducibility

- [ ] Code is on GitHub
- [ ] Checkpoints are on HuggingFace
- [ ] Hardware is specified
- [ ] Seeds are documented
- [ ] Dependencies are pinned
- [ ] Runtime estimates provided
- [ ] Reproduction commands work

## Formatting

- [ ] Tables have captions above
- [ ] Figures have captions below
- [ ] All equations are numbered
- [ ] All equations are referenced
- [ ] Consistent citation style
- [ ] No typos (run Grammarly)
- [ ] No broken links

## Honesty

- [ ] No overclaiming in abstract
- [ ] Failure cases reported
- [ ] Negative results in Appendix D
- [ ] Single-seed transparency
- [ ] "Compression" only used when real
- [ ] Parameter increase admitted

## Ethics

- [ ] Broader impact discussed
- [ ] Environmental cost estimated
- [ ] Bias concerns stated
- [ ] Misuse risks stated
- [ ] Funding disclosed
- [ ] COI disclosed

## Final Pass

- [ ] Read entire paper aloud
- [ ] Check every `{{PLACEHOLDER}}` is gone
- [ ] Verify all tables render
- [ ] Verify all equations render
- [ ] Verify PDF compiles: `python build.py --module {{MODULE}} --pdf`
- [ ] Get feedback from at least 2 readers
- [ ] Address all feedback

## Submission

- [ ] Choose venue (arXiv, workshop, conference)
- [ ] Format according to venue guidelines
- [ ] Prepare supplementary materials
- [ ] Prepare rebuttal plan
- [ ] Upload to arXiv / conference
- [ ] Announce on social media
- [ ] Update GitHub README with link

---

## Quick Start

```bash
# 1. Copy template
cp -r papers/_TEMPLATE papers/{{MODULE}}

# 2. Fill variables.json
nano papers/{{MODULE}}/variables.json

# 3. Apply variables
cd papers/{{MODULE}}
python apply_vars.py --module {{MODULE}}

# 4. Verify no placeholders remain
grep -rn "{{" .

# 5. Fill in the remaining manual sections (Method, Results, Analysis)

# 6. Build full paper
python build.py --module {{MODULE}} --pdf

# 7. Review and submit
```

**Done.** Your paper is ready.
```