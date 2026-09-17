# Analysis Template

## 8.1 Why Does It Work?
- Hypothesis 1 with evidence
- Hypothesis 2 with evidence
- Ablation-style table

## 8.2 Why Does It Fail?
- List of failed configurations
- For each, describe what you tried and what happened
- Be honest — reviewers value transparency

## 8.3 Inference Cost Breakdown
- Kernel launch count
- Python overhead
- FP32 casts
- Sequential matmul count

## 8.4 The Compression Illusion
If parameters increased, say so explicitly:
> "Despite the name, this method does not reduce parameter count..."

## 8.5 Where the Gains Come From
Component ablation table

## 8.6 Failure Mode Taxonomy
Table: failure mode, symptom, root cause, fix attempted

## Rules

- No speculation presented as fact
- Every hypothesis needs a test and a result
- Negative results are first-class content
