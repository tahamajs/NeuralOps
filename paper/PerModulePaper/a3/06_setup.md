# 6. Experimental Setup

## 6.1 Models

We evaluate A3 on DistilBERT, RoBERTa-base, and GPT-2.

## 6.2 Datasets

The main benchmark uses SST-2 for sentiment classification, with a fixed split and standard evaluation metrics.

## 6.3 Hyperparameters

The compression rank is fixed at $r = 64$ for the main result, and a lightweight LoRA recovery stage is used after compressed replacement. The recovery phase is intentionally small to emphasize the practicality of the method.

## 6.4 Evaluation Protocol

Each model is measured before and after compression under the same hardware and sequence-length configuration. We report accuracy, parameter reduction, and VRAM savings under matching conditions.
