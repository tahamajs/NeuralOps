---
name: paper-multi-seed-statistical-rigor
description: "Author mathematically rigorous empirical methodology sections on multi-seed statistical significance, Welch's t-tests, confidence intervals, and cross-seed variance reporting for AI papers"
version: "1.0.0"
tags: ["neuralops", "paper", "statistics", "reproducibility", "significance", "t-test", "confidence-intervals", "seeds"]
---

# Skill: Paper Multi-Seed Statistical Rigor & Uncertainty Quantification

## Overview & Scope
Use this skill when authoring, formalizing, auditing, or revising mathematical methodology sections, empirical statistical reporting, confidence intervals, and hypothesis testing for machine learning papers targeting top-tier AI and systems venues (NeurIPS, ICLR, ICML, MLSys, ACL, JMLR) and doctoral dissertations.

In empirical deep learning, claiming superiority based on single-seed evaluations or unquantified variance is scientifically unacceptable. Stochasticity originates from random weight initializations, non-deterministic CUDA kernel scheduling, mini-batch shuffling permutations, data augmentation pipelines, and distributed gradient reductions. This skill guarantees that every reported accuracy improvement, compression retention rate, and latency speedup is fortified by:
1. **A Structured Dual-Tier Multi-Seed Randomization Protocol**: Combining standard pseudo-random seeds with adversarial stress seeds to probe loss landscape stability.
2. **Welch's Two-Sample $t$-Tests**: Unequal-variance parametric hypothesis testing with exact Welch-Satterthwaite degrees of freedom.
3. **Rigorous Significance Thresholding**: Empirical proofs of non-triviality ($p < 0.05$, $p < 0.01$, and $p < 0.001$) under multiple-testing family-wise error rate corrections.
4. **Dual Confidence Intervals**: Both exact Student's $t$ parametric intervals and non-parametric percentile/BCa bootstrap intervals ($B = 10{,}000$).
5. **Standardized Effect Size Quantification**: Cohen's $d$ calculations across all comparative baselines.
6. **Strict Mathematical Syntax Consistency**: Mandatory math-mode English standard deviation notation (`$86.40 \pm 0.32\%$`, `$1.24 \pm 0.03\times$`).
7. **Production LaTeX Tables & High-DPI Seaborn/Matplotlib Visualizations**: Ready-to-compile camera-ready assets with fully self-contained captions.

---

## 1. Multi-Seed Randomization Protocol

Single-seed benchmarks are prone to "seed lottery" effects—where an optimization trajectory accidentally finds a sharp, unrepresentative local minimum. NeuralOps enforces a standardized multi-seed protocol across all downstream evaluations.

### 1.1 Dual-Tier Seed Taxonomy: Standard vs. Stress Seeds
Downstream evaluations (GLUE benchmark, WikiText-2 perplexity, LRA sequence tasks, and post-surgery recovery) must be evaluated across two distinct tiers of pseudo-random seeds:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   NEURALOPS MULTI-SEED RANDOMIZATION PROTOCOL               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tier 1: Standard Canonical Seeds                                          │
│  • Seed 42   : Standard baseline initialization (canonical benchmark seed)  │
│  • Seed 123  : Sequential permutation offset                                │
│  • Seed 456  : Mid-range pseudo-random state                                │
│  → Purpose   : Evaluates expected-case performance under typical seeds      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Tier 2: Adversarial Stress Seeds                                           │
│  • Seed 404  : Edge-case subspace initialization                            │
│  • Seed 777  : Prime-modulo harmonic sequence seed                          │
│  • Seed 999  : High-entropy state probing extreme mini-batch orderings       │
│  → Purpose   : Stresses loss landscape conditioning, tests for NaN traps,   │
│                singular value collapse, and gradient explosion              │
└─────────────────────────────────────────────────────────────────────────────┘
```

For extended multi-seed audits and formal statistical certificates, the evaluation suite expands to either the canonical 7-seed matrix $\mathcal{S}_7 = \{42, 100, 200, 300, 404, 777, 999\}$ or the 5-seed enterprise verification set $\mathcal{S}_5 = \{42, 1337, 2026, 31415, 99999\}$.

### 1.2 Deterministic Environment Conditioning Protocol
To ensure reproducible execution across different experimental runs, all random state generators across Python, NumPy, PyTorch CPU, and PyTorch CUDA must be explicitly seeded:

```python
import os
import random
import numpy as np
import torch

def set_deterministic_seed(seed: int) -> None:
    """Sets deterministic execution state across all runtime generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantee deterministic convolution and kernel algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 1.3 Mandatory Standard Deviation Reporting Syntax
In all manuscripts, tables, text snippets, and captions, standard deviations and uncertainties must adhere strictly to math-mode English notation:

| Metric Type | Mandatory Syntax | Prohibited Syntax (Do Not Use) |
| :--- | :--- | :--- |
| **Percentage Accuracy** | `$86.40 \pm 0.32\%$` or `$(86.40 \pm 0.32)\%$` | `86.40% +- 0.32%`, `86.40 (0.32)`, `86.40 \pm 0.32%` |
| **Speedup Factor** | `$1.24 \pm 0.03\times$` | `1.24x +- 0.03`, `1.24 +/- 0.03x` |
| **Perplexity (PPL)** | `$27.33 \pm 0.07$` | `27.33 (std 0.07)`, `27.33 +- 0.07` |
| **Latency (ms)** | `$4.12 \pm 0.04\text{ ms}$` | `4.12ms \pm 0.04ms`, `4.12 +- 0.04 ms` |
| **Compression Ratio** | `$2.18 \pm 0.02\times$` | `2.18x (0.02)` |

---

## 2. Parametric Hypothesis Testing: Welch's Two-Sample $t$-Test

When comparing the performance of a proposed architecture (e.g., $\text{HNC-SVD}$ or $\text{SVD-Linear Attention}$) against a baseline (e.g., $\text{Direct SVD}$ or $\text{Standard Softmax}$), standard Student's two-sample $t$-tests assume equal population variances ($\sigma_1^2 = \sigma_2^2$). In deep learning model surgery, this assumption is **frequently violated**: surgical interventions typically exhibit significantly different variance structures than uncompressed baselines (e.g., unconstrained fine-tuning exhibits up to $50\times$ higher variance than structured Riemannian-regularized recovery).

### 2.1 The Behrens-Fisher Problem and Welch's Formulation
To solve the Behrens-Fisher problem of comparing two normal populations with potentially unequal and unknown variances, we employ **Welch's two-sample $t$-test**.

Let $X_1 = (x_{1,1}, \dots, x_{1,N_1})$ and $X_2 = (x_{2,1}, \dots, x_{2,N_2})$ represent evaluation metrics across independent random seeds for systems 1 and 2, with sample sizes $N_1$ and $N_2$.

#### Sample Mean and Unbiased Sample Variance:
$$\bar{X}_1 = \frac{1}{N_1} \sum_{i=1}^{N_1} x_{1,i}, \quad s_1^2 = \frac{1}{N_1 - 1} \sum_{i=1}^{N_1} (x_{1,i} - \bar{X}_1)^2$$
$$\bar{X}_2 = \frac{1}{N_2} \sum_{i=1}^{N_2} x_{2,i}, \quad s_2^2 = \frac{1}{N_2 - 1} \sum_{i=1}^{N_2} (x_{2,i} - \bar{X}_2)^2$$

#### Welch's Test Statistic:
Under the null hypothesis $H_0: \mu_1 - \mu_2 = \Delta_0$ (typically $\Delta_0 = 0$ for equality of means), the test statistic is:
\begin{equation}
t = \frac{\bar{X}_1 - \bar{X}_2 - \Delta_0}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}
\label{eq:welch_t}
\end{equation}

#### Welch-Satterthwaite Effective Degrees of Freedom:
Because the denominator is a linear combination of two independent chi-square random variables with unequal coefficients, the exact distribution is not Student's $t$. The Welch-Satterthwaite approximation provides the effective degrees of freedom $\nu$:
\begin{equation}
\nu \approx \frac{\left( \frac{s_1^2}{N_1} + \frac{s_2^2}{N_2} \right)^2}{\frac{\left(s_1^2 / N_1\right)^2}{N_1 - 1} + \frac{\left(s_2^2 / N_2\right)^2}{N_2 - 1}}
\label{eq:welch_nu}
\end{equation}

```latex
% Formal LaTeX definition for paper methodology
\begin{equation}
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}, \quad 
\nu \approx \frac{\left(\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}\right)^2}{\frac{(s_1^2/N_1)^2}{N_1 - 1} + \frac{(s_2^2/N_2)^2}{N_2 - 1}}
\label{eq:welchs_t_test}
\end{equation}
```

### 2.2 Rejection Region and Two-Tailed $p$-Value Calculation
For a two-tailed test with significance level $\alpha$:
1. Calculate the cumulative Student's $t$ distribution function $F_t(t; \nu)$.
2. The two-tailed $p$-value is:
   $$p = 2 \cdot \left(1 - F_t(|t|; \nu)\right)$$
3. Reject $H_0$ if $p < \alpha$, concluding that the performance difference is statistically significant and not an artifact of random seed selection.

---

## 3. Significance Thresholding & Multi-Testing Corrections

### 3.1 Standard Significance Tiers
To prevent scientific overclaiming and misleading conclusions, p-values must be mapped to explicit significance tiers:

| Threshold | Significance Descriptor | LaTeX Marker | Empirical Interpretation |
| :--- | :--- | :---: | :--- |
| **$p < 0.001$** | Extremely Significant | $^{***}$ | Decisive proof; probability of observed difference under $H_0 < 0.1\%$. |
| **$p < 0.01$** | Highly Significant | $^{**}$ | Strong scientific proof; standard benchmark for top-tier conference claims. |
| **$p < 0.05$** | Statistically Significant | $^{*}$ | Acceptable threshold; indicates demonstrable effect over baseline. |
| **$p \ge 0.05$** | Not Statistically Significant | $\text{n.s.}$ | Inconclusive; differences cannot be distinguished from random seed noise. |

### 3.2 Family-Wise Error Rate (FWER) and FDR Control
When evaluating models across multiple benchmark tasks (e.g., 8 GLUE tasks or 6 attention architectures), computing multiple pairwise $t$-tests inflates the probability of Type I errors (false positives).

#### 1. Bonferroni Correction (Conservative FWER):
For $m$ simultaneous hypothesis tests with global significance level $\alpha$:
$$\alpha_{\text{Bonferroni}} = \frac{\alpha}{m}$$
Reject $H_{0,i}$ if $p_i < \frac{\alpha}{m}$.

#### 2. Holm-Bonferroni Step-Down Procedure (Uniformly More Powerful):
Sort unadjusted $p$-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$ corresponding to null hypotheses $H_{(1)}, \dots, H_{(m)}$.
For step $k = 1, 2, \dots, m$:
- If $p_{(k)} \le \frac{\alpha}{m - k + 1}$, reject $H_{(k)}$ and continue to step $k+1$.
- If $p_{(k)} > \frac{\alpha}{m - k + 1}$, stop and retain all remaining null hypotheses $H_{(k)}, \dots, H_{(m)}$.

#### 3. Benjamini-Hochberg False Discovery Rate (FDR):
To control the expected proportion of false discoveries at rate $q = 0.05$:
Find the largest $k$ such that:
$$p_{(k)} \le \frac{k}{m} \cdot q$$
Reject all hypotheses $H_{(1)}, \dots, H_{(k)}$.

---

## 4. Confidence Intervals: Parametric Student's $t$ and Bootstrapped BCa

Reporting only sample means or standard deviations obscures the true uncertainty of the population parameter. Every flagship accuracy metric must be accompanied by its **$95\%$ Confidence Interval ($95\%\text{ CI}$)**.

### 4.1 Parametric Student's $t$ Confidence Interval
When sample size $N$ is small (e.g., $N=6$ or $N=7$) and the underlying metric across seeds is approximately normally distributed:
\begin{equation}
\text{CI}_{1-\alpha} = \left[ \bar{X} - t_{\nu, 1-\alpha/2} \cdot \frac{s}{\sqrt{N}}, \quad \bar{X} + t_{\nu, 1-\alpha/2} \cdot \frac{s}{\sqrt{N}} \right]
\label{eq:param_ci}
\end{equation}
where $\nu = N - 1$ degrees of freedom and $t_{\nu, 1-\alpha/2}$ is the critical value of Student's $t$ distribution.
For $N=7$ seeds ($\nu = 6$) at $95\%$ confidence ($\alpha = 0.05$), the critical value is $t_{6, 0.975} \approx 2.4469$.

### 4.2 Non-Parametric Bootstrap Confidence Intervals ($B = 10{,}000$)
When sample distributions exhibit skewness or bounded ceiling effects (e.g., accuracy bounded at $100\%$), non-parametric bootstrapping generates distribution-free confidence intervals without asymptotic normality assumptions.

#### Algorithm: Non-Parametric Bootstrap
1. Let sample data be $\mathbf{x} = (x_1, \dots, x_N)$.
2. For $b = 1, \dots, B$ (where $B = 10{,}000$):
   - Draw a random resample $\mathbf{x}^{*(b)} = (x_1^*, \dots, x_N^*)$ with replacement from $\mathbf{x}$.
   - Compute the bootstrap replicate estimate: $\hat{\theta}^{*(b)} = g(\mathbf{x}^{*(b)})$.
3. Sort bootstrap replicates in ascending order: $\hat{\theta}^{*(1)} \le \hat{\theta}^{*(2)} \le \dots \le \hat{\theta}^{*(B)}$.

#### 1. Percentile Bootstrap Interval:
\begin{equation}
\text{CI}_{\text{percentile}} = \left[ \hat{\theta}^*_{\lfloor B \cdot \alpha/2 \rfloor}, \quad \hat{\theta}^*_{\lceil B \cdot (1 - \alpha/2) \rceil} \right]
\label{eq:percentile_boot}
\end{equation}
For $B = 10{,}000$ at $\alpha = 0.05$, the interval endpoints are the $250$-th and $9750$-th ordered replicates: $\left[ \hat{\theta}^{*(250)}, \hat{\theta}^{*(9750)} \right]$.

#### 2. Bias-Corrected and Accelerated (BCa) Bootstrap:
Adjusts for both median bias $z_0$ and distribution skewness (acceleration $a$):
\begin{equation}
\alpha_1 = \Phi\left( z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})} \right), \quad 
\alpha_2 = \Phi\left( z_0 + \frac{z_0 + z_{1-\alpha/2}}{1 - a(z_0 + z_{1-\alpha/2})} \right)
\label{eq:bca_formula}
\end{equation}
where:
$$z_0 = \Phi^{-1}\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}\{\hat{\theta}^{*(b)} < \hat{\theta}\} \right)$$
$$a = \frac{\sum_{i=1}^N (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^3}{6 \left[ \sum_{i=1}^N (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^2 \right]^{3/2}}$$
and $\hat{\theta}_{(i)}$ is the leave-one-out jackknife estimate omitting sample $i$.

---

## 5. Non-Parametric Testing & Standardized Effect Sizes

### 5.1 Wilcoxon Signed-Rank Test (Paired Seeds)
When comparing two architectures evaluated on identical paired seeds ($s \in \mathcal{S}$), the Wilcoxon signed-rank test provides a distribution-free non-parametric alternative:
1. Compute pairwise differences $d_i = x_{1,i} - x_{2,i}$ for each seed $i = 1, \dots, N$.
2. Discard differences where $d_i = 0$, leaving $N_r$ non-zero pairs.
3. Order absolute differences $|d_1|, \dots, |d_{N_r}|$ and assign ranks $R_i \in \{1, \dots, N_r\}$.
4. Compute the test statistic:
   \begin{equation}
   W = \sum_{i=1}^{N_r} \operatorname{sgn}(d_i) \cdot R_i
   \label{eq:wilcoxon_w}
   \end{equation}

### 5.2 Standardized Effect Size: Cohen's $d$
A statistically significant $p$-value does not necessarily imply a practically significant difference; conversely, small sample sizes might yield $p \approx 0.04$ despite massive empirical shifts. Standardized effect sizes quantify the magnitude of the divergence in units of pooled standard deviation.

\begin{equation}
d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad 
s_{\text{pooled}} = \sqrt{\frac{(N_1 - 1)s_1^2 + (N_2 - 1)s_2^2}{N_1 + N_2 - 2}}
\label{eq:cohens_d}
\end{equation}

#### Effect Size Magnitude Interpretation:
- $|d| < 0.20$: **Negligible** effect (noise regime).
- $0.20 \le |d| < 0.50$: **Small** effect.
- $0.50 \le |d| < 0.80$: **Medium** effect (detectable across seeds).
- $0.80 \le |d| < 1.20$: **Large** effect.
- $|d| \ge 1.20$: **Very Large / Transformative** effect (e.g., $\text{HNC-SVD}$ recovery over $\text{Direct SVD}$ typically achieves $|d| > 20.0$).

---

## 6. Numerical Stability Auditing & Zero-Divergence Law

Statistical validity requires that no evaluated seed encountered numerical instability, floating-point divergence, or silent gradient corruption.

### 6.1 The Zero-Divergence Audit Protocol
Across all seeds $s \in \mathcal{S}$, the runtime verification pipeline must confirm:
1. **Zero NaN / Inf Invariant**:
   $$\forall s \in \mathcal{S}, \quad \sum \mathbb{I}\{\operatorname{isnan}(\mathbf{W}_s)\} = 0 \quad \text{and} \quad \sum \mathbb{I}\{\operatorname{isinf}(\mathbf{W}_s)\} = 0$$
2. **Gradient Norm Boundedness**:
   $$\max_{t \in [1, T]} \|\mathbf{g}_t\|_2 \le G_{\text{clip}} < \infty$$
3. **Loss Monotonicity**: No run exhibits catastrophic divergence ($\mathcal{L}_{t+1} > 10 \cdot \mathcal{L}_t$).

### 6.2 Hardware Latency Jitter Protocol
In addition to accuracy metrics, latency and throughput measurements across seeds must report hardware jitter. The Coefficient of Variation ($\mathrm{CV}$) must be reported:
$$\mathrm{CV} = \frac{s}{\bar{X}} \times 100\%$$
For physical GPU evaluations on dedicated hardware, $\mathrm{CV} \le 1.5\%$ is required to certify that measurements reflect algorithmic throughput rather than operating system background contention or thermal throttling.

---

## 7. Production LaTeX Statistical Tables

### 7.1 Multi-Seed Accuracy & Variance Table (Individual Seeds + Welch's Test)

```latex
\begin{table*}[t]
\centering
\caption{\textbf{Cross-Seed Statistical Significance and Robustness Matrix Across 6 Random Seeds.}
Evaluations conducted on \textbf{RoBERTa-base} fine-tuned on SST-2 ($d=768$, $N=6$ seeds: 3 standard $\{42, 123, 456\}$ and 3 adversarial stress $\{404, 777, 999\}$). 
Welch's two-sample $t$-test and Cohen's $d$ effect sizes are computed relative to the Uncalibrated Direct SVD baseline. 
Significance levels: $^{***} p < 0.001$, $^{**} p < 0.01$, $^{*} p < 0.05$.}
\label{tab:multiseed_significance}
\small
\setlength{\tabcolsep}{4.2pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{} l cccccc ccc cc @{}}
\toprule
\textbf{Architecture / Method} & 
\textbf{Seed 42} & \textbf{Seed 123} & \textbf{Seed 456} & 
\textbf{Seed 404} & \textbf{Seed 777} & \textbf{Seed 999} & 
\textbf{Mean (\%)} & \textbf{Std ($\pm$)} & \textbf{95\% CI} & 
\textbf{Welch's $t$} & \textbf{Cohen's $d$} \\
\midrule
Standard Softmax Baseline     & 94.50 & 94.42 & 94.55 & 94.38 & 94.48 & 94.42 & \textbf{94.46} & 0.06 & $[94.40, 94.52]$ & --- & --- \\
Direct SVD (Rank 32, 50\%)    & 84.82 & 84.60 & 85.10 & 84.45 & 84.75 & 84.50 & 84.70 & 0.23 & $[84.46, 84.94]$ & \textit{ref} & \textit{ref} \\
Whitened SVD (No Recovery)    & 88.15 & 88.02 & 88.30 & 87.90 & 88.22 & 88.05 & 88.11 & 0.14 & $[87.96, 88.26]$ & $30.82^{***}$ & 17.79 \\
HNC Nullspace-Only            & 87.90 & 87.75 & 88.05 & 87.60 & 87.95 & 87.80 & 87.84 & 0.16 & $[87.67, 88.01]$ & $27.35^{***}$ & 15.79 \\
\textbf{Full HNC-SVD (Ours)}  & 93.12 & 93.05 & 93.20 & 92.95 & 93.15 & 93.08 & \textbf{93.09} & \textbf{0.09} & $[93.00, 93.18]$ & $\mathbf{83.31^{***}}$ & $\mathbf{48.10}$ \\
\midrule
\textbf{6-Pillar Healed Model} & 94.18 & 94.10 & 94.25 & 94.02 & 94.20 & 94.12 & \textbf{94.15} & \textbf{0.08} & $[94.07, 94.23]$ & $\mathbf{94.62^{***}}$ & $\mathbf{54.63}$ \\
\bottomrule
\end{tabular}
\end{table*}
```

### 7.2 Multi-Backbone Multi-Task Generalization Matrix

```latex
\begin{table}[t]
\centering
\caption{\textbf{Multi-Seed Robustness Evaluation Across Diverse Model Families and Tasks.}
Reported as $\text{Mean} \pm \text{Std Dev}$ across $N=6$ independent seeds. 
Evaluated on bidirectional encoder (\textbf{DistilBERT-base}, \textbf{RoBERTa-base}) and autoregressive causal decoder (\textbf{GPT-2}). 
All improvements over direct truncation pass Welch's $t$-test at $p < 0.001$.}
\label{tab:multitask_multiseed}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{} l l ccc c @{}}
\toprule
\textbf{Backbone Family} & \textbf{Evaluation Task} & \textbf{Dense Baseline} & \textbf{Direct SVD} & \textbf{HNC-SVD (Ours)} & \textbf{Retention (\%)} \\
\midrule
\multirow{3}{*}{\textbf{DistilBERT-base}} 
 & SST-2 (Accuracy \%)       & $90.71 \pm 0.08$ & $51.30 \pm 0.42$ & $\mathbf{87.20 \pm 0.11}$ & $96.13\%$ \\
 & MNLI (Match Acc \%)      & $82.20 \pm 0.12$ & $34.15 \pm 0.38$ & $\mathbf{79.45 \pm 0.14}$ & $96.65\%$ \\
 & AG News (Accuracy \%)     & $93.80 \pm 0.07$ & $64.20 \pm 0.35$ & $\mathbf{91.90 \pm 0.09}$ & $97.97\%$ \\
\midrule
\multirow{3}{*}{\textbf{RoBERTa-base}} 
 & SST-2 (Accuracy \%)       & $94.50 \pm 0.06$ & $84.70 \pm 0.23$ & $\mathbf{93.09 \pm 0.09}$ & $98.51\%$ \\
 & MNLI (Match Acc \%)      & $87.60 \pm 0.10$ & $71.20 \pm 0.31$ & $\mathbf{85.80 \pm 0.12}$ & $97.95\%$ \\
 & AG News (Accuracy \%)     & $94.90 \pm 0.05$ & $86.50 \pm 0.28$ & $\mathbf{93.85 \pm 0.08}$ & $98.89\%$ \\
\midrule
\multirow{2}{*}{\textbf{GPT-2 (124M)}} 
 & WikiText-2 (PPL $\downarrow$) & $26.15 \pm 0.05$ & $54.20 \pm 0.37$ & $\mathbf{26.85 \pm 0.06}$ & $97.39\%$ \\
 & LAMBADA (Accuracy \%)     & $45.20 \pm 0.14$ & $18.40 \pm 0.45$ & $\mathbf{43.10 \pm 0.16}$ & $95.35\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 8. Publication-Grade Statistical Visualization Templates

### 8.1 Figure 1: Cross-Seed Boxplot with Jitter Overlay & Mean Markers

```python
"""
Publication-grade boxplot with individual seed jitter overlay.
Generates camera-ready PDF conforming to IEEE/ACM and NeurIPS visual guidelines.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure publication typography and style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "text.usetex": False,  # True if host LaTeX installation is available
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

seeds = [42, 123, 456, 404, 777, 999]

data_dict = {
    "Standard Softmax": [94.50, 94.42, 94.55, 94.38, 94.48, 94.42],
    "Direct SVD (50%)": [84.82, 84.60, 85.10, 84.45, 84.75, 84.50],
    "Whitened SVD":     [88.15, 88.02, 88.30, 87.90, 88.22, 88.05],
    "HNC Nullspace":    [87.90, 87.75, 88.05, 87.60, 87.95, 87.80],
    "Full HNC-SVD":     [93.12, 93.05, 93.20, 92.95, 93.15, 93.08]
}

rows = []
for method, vals in data_dict.items():
    for s, v in zip(seeds, vals):
        rows.append({"Method": method, "Seed": s, "Accuracy (%)": v})
df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

palette = ["#64748b", "#ef4444", "#0284c7", "#f59e0b", "#10b981"]

# 1. Boxplot showing median and IQR
sns.boxplot(
    data=df, x="Method", y="Accuracy (%)", ax=ax,
    palette=palette, width=0.45, boxprops=dict(alpha=0.65),
    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    medianprops=dict(color="black", linewidth=1.5), showmeans=True,
    meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white", markersize=6)
)

# 2. Stripplot showing individual seed jitter
sns.stripplot(
    data=df, x="Method", y="Accuracy (%)", ax=ax,
    color="black", size=6, jitter=0.15, alpha=0.85
)

ax.set_ylabel("Validation Accuracy (%)", fontweight="bold")
ax.set_xlabel("Factorization & Surgery Configuration", fontweight="bold")
ax.set_title("Cross-Seed Distribution & Variance Analysis across 6 Random Seeds", pad=12)

# Annotate significance comparison bracket
y_max = 93.30
ax.plot([1, 1, 4, 4], [y_max, y_max + 0.3, y_max + 0.3, y_max], color="black", lw=1.2)
ax.text(2.5, y_max + 0.35, r"Welch's $t = 83.31^{***}, \ p < 10^{-12}$", 
        ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig_cross_seed_boxplot.pdf", bbox_inches="tight")
plt.close()
```

### 8.2 Figure 2: Shaded 95% Confidence Trajectories during Recovery

```python
"""
Plot training convergence trajectories with shaded 95% confidence bands across seeds.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

steps = np.arange(1, 51)
n_seeds = 6
t_crit = stats.t.ppf(0.975, df=n_seeds - 1)

# Generate synthetic deterministic convergence matrix for demonstration
np.random.seed(42)
trajectories_6pillar = []
trajectories_naive = []

for _ in range(n_seeds):
    traj_6p = 94.15 - 10.0 * np.exp(-steps / 8.0) + np.random.normal(0, 0.08, len(steps))
    traj_naive = 88.50 - 15.0 * np.exp(-steps / 15.0) + np.random.normal(0, 0.45, len(steps))
    trajectories_6pillar.append(traj_6p)
    trajectories_naive.append(traj_naive)

mat_6p = np.array(trajectories_6pillar)
mat_naive = np.array(trajectories_naive)

mean_6p = np.mean(mat_6p, axis=0)
ci_6p = t_crit * (np.std(mat_6p, axis=0, ddof=1) / np.sqrt(n_seeds))

mean_naive = np.mean(mat_naive, axis=0)
ci_naive = t_crit * (np.std(mat_naive, axis=0, ddof=1) / np.sqrt(n_seeds))

fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

# Plot 6-Pillar Recovery
ax.plot(steps, mean_6p, color="#10b981", lw=2.2, label=r"6-Pillar Distillation (Ours, $95\%$ CI)")
ax.fill_between(steps, mean_6p - ci_6p, mean_6p + ci_6p, color="#10b981", alpha=0.25)

# Plot Naive CE Fine-Tuning
ax.plot(steps, mean_naive, color="#ef4444", lw=2.2, linestyle="--", label=r"Naive CE Fine-Tuning ($95\%$ CI)")
ax.fill_between(steps, mean_naive - ci_naive, mean_naive + ci_naive, color="#ef4444", alpha=0.20)

ax.axhline(94.46, color="#64748b", linestyle=":", lw=1.5, label="Dense Teacher Baseline (94.46%)")

ax.set_xlabel("Distillation Recovery Steps", fontweight="bold")
ax.set_ylabel("Validation Accuracy (%)", fontweight="bold")
ax.set_title("Post-Surgery Optimization Trajectories with 95% Confidence Envelopes", pad=12)
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("figures/fig_seed_trajectories_ci.pdf", bbox_inches="tight")
plt.close()
```

### 8.3 Figure 3: Pairwise Welch's $t$-Test $p$-Value Heatmap

```python
"""
Generate pairwise hypothesis testing matrix with p-value heatmap.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

methods = ["Standard", "Direct SVD", "Whitened", "Nullspace", "HNC-SVD"]
p_mat = np.ones((len(methods), len(methods)))

for i in range(len(methods)):
    for j in range(len(methods)):
        if i != j:
            vals_i = data_dict[list(data_dict.keys())[i]]
            vals_j = data_dict[list(data_dict.keys())[j]]
            _, p_val = stats.ttest_ind(vals_i, vals_j, equal_var=False)
            p_mat[i, j] = p_val

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
sns.heatmap(
    p_mat, annot=True, fmt=".2e", cmap="magma_r", ax=ax,
    xticklabels=methods, yticklabels=methods, cbar_kws={'label': 'Welch p-value'}
)
ax.set_title(r"Pairwise Welch's $t$-Test Matrix ($p$-values)", pad=12)
plt.tight_layout()
plt.savefig("figures/fig_pairwise_pvalues.pdf", bbox_inches="tight")
plt.close()
```

---

## 9. Scientific Caption Guidelines

Reviewers evaluate caption quality as a direct signal of experimental rigor. A caption must be **completely self-contained**, enabling a reader to understand the statistical evidence without searching the main text.

### 9.1 Mandatory Caption Elements Checklist
Every statistical figure or table caption must explicitly declare:
1. **Sample Size ($N$)**: Explicitly state the exact number of evaluation runs ($N=6$, $N=7$).
2. **Explicit Seed Set**: List all evaluation seeds (e.g., `seeds: 42, 123, 456, 404, 777, 999`).
3. **Definition of Dispersion**: Explicitly distinguish between Standard Deviation ($\pm \sigma$), Standard Error ($\pm \text{SEM}$), or $95\%$ Confidence Interval ($95\%\text{ CI}$). Never use `\pm` without defining what follows it.
4. **Statistical Test Specification**: State the exact test used (e.g., "Two-sided Welch's two-sample $t$-test with Welch-Satterthwaite degrees of freedom").
5. **Significance Level Mapping**: Explicitly define asterisks ($^{***} p < 0.001$, $^{**} p < 0.01$, $^{*} p < 0.05$).
6. **Effect Size**: Report Cohen's $d$ for major claims.

### 9.2 Exemplary Self-Contained Captions

#### Good Table Caption:
> **Table 4: Multi-Seed Robustness Evaluation and Welch's $t$-Test Significance Matrix.**  
> Performance reported as $\text{Mean} \pm \text{Std Dev}$ across $N=6$ independent random initializations (standard seeds $\{42, 123, 456\}$ and adversarial stress seeds $\{404, 777, 999\}$) on RoBERTa-base ($d=768$) evaluated on SST-2 validation set. $95\%$ Confidence Intervals are computed via non-parametric BCa bootstrapping with $B=10{,}000$ resamples. Pairwise comparisons against the uncalibrated Direct SVD baseline use two-sided Welch's two-sample $t$-tests ($\nu \approx 6.4$); $^{***} p < 0.001$, $^{**} p < 0.01$, $^{*} p < 0.05$. Cohen's $d$ quantifies standardized effect size. All evaluations executed on an NVIDIA GeForce RTX 4070 (Ada Lovelace, SM 8.9) under identical deterministic CUDA state constraints.

#### Good Figure Caption:
> **Figure 3: Cross-Seed Accuracy Distributions and Optimization Trajectories.**  
> **(a)** Box-and-whisker distribution plot across $N=6$ seeds ($\mathcal{S} = \{42, 123, 456, 404, 777, 999\}$). Box boundaries indicate the first and third quartiles (IQR); center lines represent medians; diamond markers denote empirical means; overlaid jittered points represent individual seed runs. HNC-SVD achieves a $2.5\times$ variance reduction over Direct SVD ($\sigma = 0.09\%$ vs. $0.23\%$). **(b)** 50-step post-surgery distillation recovery trajectories; shaded envelopes denote two-sided $95\%$ Student's $t$ confidence intervals ($t_{5, 0.975} = 2.571$). 6-pillar distillation recovers $98.5\%$ of dense teacher accuracy with zero seed-induced divergence.

---

## 10. Reviewer Defense & Rebuttal Playbook

When submitting to NeurIPS, ICLR, or MLSys, statistical methodology frequently faces specific reviewer objections. Use these formal defense arguments:

### Objection 1: *"The performance improvement is marginal; how do we know it isn't seed variance?"*
**Author Defense**:
> "We emphasize that all reported gains are verified across a standardized dual-tier protocol of $N=6$ independent seeds, encompassing both canonical seeds (42, 123, 456) and adversarial stress seeds (404, 777, 999). As detailed in Table 4, our proposed method achieves a $+8.39\%$ accuracy gain over Direct SVD with an empirical standard deviation of $\sigma = 0.09\%$. Welch's two-sample $t$-test for unequal variances yields $t = 83.31$ with $\nu \approx 6.4$ degrees of freedom, rejecting the null hypothesis of equal means at $p = 4.2 \times 10^{-13} \ll 0.001$. Furthermore, the standardized effect size is Cohen's $d = 48.10$, demonstrating that the observed performance recovery exceeds the stochastic seed variance by more than $48$ standard deviations."

### Objection 2: *"Why did you use Welch's t-test instead of standard Student's t-test?"*
**Author Defense**:
> "Standard Student's $t$-tests assume homoscedasticity ($\sigma_1^2 = \sigma_2^2$), which is violated in model compression evaluations: surgical interventions without manifold regularization exhibit substantially higher cross-seed variance than structured recovery ($\sigma = 0.23\%$ for Direct SVD vs. $\sigma = 0.09\%$ for HNC-SVD, Levene's test $p = 0.018$). Under heteroscedastic conditions, Student's $t$-test suffers from severe Type I error rate inflation. Welch's $t$-test with Welch-Satterthwaite degrees of freedom eliminates this vulnerability without requiring equal variances."

### Objection 3: *"Did you correct for multiple comparisons across the benchmark tasks?"*
**Author Defense**:
> "Yes. To prevent Type I error inflation across the evaluation benchmark suite ($m=8$ classification and perplexity tasks), all significance claims are adjusted using the Holm-Bonferroni step-down procedure controlling the family-wise error rate at $\alpha = 0.01$. Every task comparison retains statistical significance under the adjusted threshold ($p < \alpha_{\text{Holm}} = 0.00125$ for the smallest p-value)."

---

## 11. Pre-Submission Verification & Quality Gates

Before finalizing any empirical manuscript section or PR involving statistical claims, verify this quality gate checklist:

```
[ ] Dual-Tier Multi-Seed Randomization: Evaluation incorporates at least 3 standard seeds (42, 123, 456) and 3 stress seeds (404, 777, 999).
[ ] Standard Deviation Syntax: All text and tables use math-mode English notation ($86.40 \pm 0.32\%$, $1.24 \pm 0.03\times$).
[ ] Parametric Testing: Welch's two-sample t-test is used for unequal variances (not naive Student's t-test).
[ ] Degrees of Freedom: Welch-Satterthwaite degrees of freedom nu is reported alongside t-statistics.
[ ] Significance Thresholding: P-values mapped to explicit significance tiers (p < 0.001, p < 0.01, p < 0.05).
[ ] Multiple-Testing Correction: Multi-task tables apply Holm-Bonferroni or Benjamini-Hochberg corrections.
[ ] Confidence Intervals: Both mean and 95% CI are explicitly reported.
[ ] Standardized Effect Size: Cohen's d is calculated and reported for primary baseline comparisons.
[ ] Zero-Divergence Audit: Formally verified zero NaN, Inf, or loss explosion across all seeds.
[ ] Self-Contained Captions: Captions explicitly declare sample size N, seed list, dispersion definition, and test parameters.
[ ] Clean Infrastructure References: No internal remote IP addresses, hostnames, or private environment paths appear in paper assets.
```
