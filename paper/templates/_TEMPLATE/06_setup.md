# 6. Experimental Setup

## 6.1 Models

We evaluate on {{MODEL_LIST}}.

## 6.2 Datasets

We use {{DATASET_NAME}} for {{TASK}}. The benchmark uses the standard train/validation/test split and reports accuracy as the primary metric.

## 6.3 Hyperparameters

Training uses {{RECOVERY_STEPS}} recovery steps at rank {{LORA_RANK}} and scale factor {{LORA_ALPHA}}. The optimizer is {{OPTIMIZER}} with learning rate {{LR}}.

## 6.4 Evaluation Protocol

We report baseline, recovered, and recovered-with-module results for each model family. We also report inference memory and latency under the same hardware conditions and sequence lengths.
