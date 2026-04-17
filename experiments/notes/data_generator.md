
# 🧪 Advanced Synthetic Data Generator for NLP Research

A high-fidelity, multi-domain **Synthetic Data Engine** designed to generate robust, diverse, and realistic datasets for training and evaluating Natural Language Processing (NLP) models. This tool addresses the scarcity of labeled data by programmatically generating thousands of samples across 7 specialized domains with varying linguistic complexities and realistic noise patterns.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Supported Domains](#supported-domains)
4. [Data Generation Strategies](#data-generation-strategies)
5. [Project Structure](#project-structure)
6. [Prerequisites & Installation](#prerequisites--installation)
7. [Usage Guide](#usage-guide)
8. [Dataset Specifications](#dataset-specifications)
9. [Output Examples](#output-examples)
10. [Author & License](#author--license)

---

## ✨ Overview

This script (`data_generator_v2.py`) creates a **comprehensive sentiment analysis dataset** tailored for rigorous testing of model compression techniques (like SVD) and robustness against real-world noise. Unlike simple random text generators, this engine uses **context-aware templates**, **domain-specific vocabulary**, and **controlled noise injection** to simulate authentic user-generated content.

---

## 🔑 Key Features

- **🌍 Multi-Domain Coverage**: Generates data for 7 distinct sectors (Movie, Product, Social Media, Education, Tech, Health, Finance).
- **🧠 Linguistic Complexity Levels**:
  - `Simple`: Direct Subject-Verb-Object structures.
  - `Compound`: Sentences joined by conjunctions (e.g., "and", "but").
  - `Complex Nested`: Subordinate clauses and advanced grammatical structures.
- **🛡️ Realistic Noise Injection**: Simulates imperfect human typing with:
  - Typos (swapped letters, missing characters).
  - Case inconsistencies (random uppercase/lowercase).
  - Missing punctuation.
  - Word repetition.
- **⚖️ Balanced Distribution**: Strict 50/50 split between Positive (1) and Negative (0) labels across all domains and complexity levels.
- **🔄 Semantic Diversity**: Uses paraphrasing and contrastive logic to prevent overfitting on specific sentence structures.

---

## 🗺️ Supported Domains

| Domain | Description | Example Keywords |
| :--- | :--- | :--- |
| **Movie** | Film reviews and critiques | *cinematography, plot, acting, direction* |
| **Product** | E-commerce reviews (Amazon style) | *battery life, build quality, shipping, warranty* |
| **Tweet** | Short-form social media posts | *movement, viral, bias, community* |
| **Education** | Course and learning platform feedback | *curriculum, instructor, assignments, syllabus* |
| **Tech News** | Technology and software updates | *API, security, latency, cloud, algorithm* |
| **Health** | Medical treatments and wellness | *therapy, diagnosis, symptoms, medication* |
| **Finance** | Economic reports and investment advice | *portfolio, ROI, volatility, assets, inflation* |

---

## 🛠 Prerequisites & Installation

### Requirements
- Python 3.8+
- `pandas`
- `numpy`

### Install Dependencies
```bash
pip install pandas numpy
```

---

## 🚀 Usage Guide

### Step 1: Run the Generator
Execute the script from the command line. It will automatically create the `data/` directory and populate it with files.

```bash
python data_generator_v2.py
```

### Configuration Options
You can modify the `total_samples` parameter in the `if __name__ == "__main__":` block to adjust the dataset size.
- **Default**: 10,000 samples (approx. 1,400 per domain).
- **Custom**: Change `total_samples=20000` for larger datasets.

### Expected Output
The script generates:
1.  **`combined_master_dataset.csv`**: A single file containing all data (shuffled).
2.  **`data_{domain}.csv`**: Individual files for each domain for granular analysis.

---

## 📊 Dataset Specifications

| Metric | Value |
| :--- | :--- |
| **Total Samples** | 10,000 (configurable) |
| **Positive/Negative Ratio** | 50% / 50% |
| **Domains** | 7 (Movie, Product, Tweet, Education, Tech, Health, Finance) |
| **Complexity Types** | Simple, Compound, Complex Nested |
| **Noise Rate** | ~30% of samples contain intentional noise |
| **Label Encoding** | 1 = Positive, 0 = Negative |
| **Columns** | `text`, `label`, `domain`, `complexity`, `noise` |

---

## 💻 Code Snippet: How It Works

The core generation logic combines vocabulary with templates:

```python
def generate_complex_sentence(domain: str, label: int, complexity: str) -> str:
    voc = DOMAIN_VOC[domain]
    # Select appropriate adjectives/nouns based on sentiment
    adj_list = voc["adj_pos"] if label == 1 else voc["adj_neg"]
    
    # Choose template based on complexity level
    if complexity == "simple":
        template = f"The {random.choice(voc['noun_pos'])} was {random.choice(adj_list)}."
    elif complexity == "complex_nested":
        template = f"Despite the {random.choice(adj_list)} nature of the {voc['noun_pos'][0]}, ..."
    
    return clean_text(template.format(...))
```

---

## 📈 Output Examples

The generated data includes both clean and noisy examples to ensure model robustness.

### Clean Examples
> ✅ **POS | TECH_NEWS | COMPLEX_NESTED**: "It is rare to see a **security protocol** that **protects users** so **robustly**, yet the **latency** remained **minimal**."

### Noisy/Realistic Examples
> ❌ **NEG | PRODUCT | SIMPLE + NOISE**: "t is rare to find such **value-packed** **build quality**... but the **shipping time** was **terrible**."  
> *(Note: The typo "t is" simulates a user typing error)*

> ✅ **POS | MOVIE | CONTRASTIVE**: "The **cinematography** was **stunning**, yet the **plot holes** were **confusing**."

---

## 👤 Author & License

**Developed by**: Rayno  
**Organization**: Apadana Intelligent Solutions Development Company  
**Date**: April 2026  

**License**: Educational & Research Use Only.  
*All code and generated synthetic data are provided for academic research, benchmarking, and educational purposes.*

---

> **Tip**: When using this dataset for training or evaluation, always shuffle the data (as done automatically by this script) to prevent the model from learning the order of domains or complexity levels.


RESULTS:

(bert_svd_env) ➜  Project git:(main) ✗ python src/data_generator.py    

🚀 Starting COMPREHENSIVE Generation... Target: 10000 samples
------------------------------------------------------------
✅ Domain 'movie': Generated 4284 samples (Pos: 2142, Neg: 2142)
✅ Domain 'product': Generated 4284 samples (Pos: 2142, Neg: 2142)
✅ Domain 'tweet': Generated 4284 samples (Pos: 2142, Neg: 2142)
✅ Domain 'education': Generated 4284 samples (Pos: 2142, Neg: 2142)
✅ Domain 'tech_news': Generated 4284 samples (Pos: 2142, Neg: 2142)
✅ Domain 'health_medical': Generated 4284 samples (Pos: 2142, Neg: 2142)
✅ Domain 'finance_econ': Generated 4284 samples (Pos: 2142, Neg: 2142)

============================================================
📊 FINAL SUMMARY - ULTRA-COMPREHENSIVE SYNTHETIC DATASET
============================================================
Total Samples:     29988
Positive (1):      14994 (50.0%)
Negative (0):      14994 (50.0%)
Domains Covered:   movie, product, tweet, education, tech_news, health_medical, finance_econ
Complexities:      simple, compound, complex_nested
Noise Applied:     ~30% of samples
Master File:       /Users/tahamajs/Documents/uni/Project/data/combined_master_dataset.csv
------------------------------------------------------------
📝 Sample Texts (Mixed Domains, Complexities & Noise):
  [❌ NEG] FINANCE_ECON | compound        | 🔇 CLEAN    | This product/service failed to deliver the inflation was speculative..
  [✅ POS] PRODUCT      | complex_nested  | 🔇 NOISE    | The warranty was intuitive, yet the packaging was defective.
  [❌ NEG] TWEET        | complex_nested  | 🔇 CLEAN    | Despite the weak nature of the apathy, the scandal spreads lies in a w...
  [✅ POS] PRODUCT      | complex_nested  | 🔇 CLEAN    | I must say, say, i found the interface to be sturdy.
  [✅ POS] HEALTH_MEDICAL | complex_nested  | 🔇 CLEAN    | t is rare to see a treatment that relieves so therapeutic, yet the car...
  [✅ POS] EDUCATION    | compound        | 🔇 CLEAN    | Truly, i found the learning path to be engaging.
  [❌ NEG] MOVIE        | compound        | 🔇 NOISE    | While the writing alienated beautifully, the plot holes resonated terr...
  [✅ POS] MOVIE        | simple          | 🔇 CLEAN    | The artistry was flawless, yet the pacing was tedious.
  [❌ NEG] PRODUCT      | simple          | 🔇 CLEAN    | Regrettably, i found the return process to be flimsy.
  [❌ NEG] FINANCE_ECON | compound        | 🔇 CLEAN    | One of the worst experiences I've had regarding i found the bubble to ...
============================================================

✨ SUCCESS! Dataset ready at: /Users/tahamajs/Documents/uni/Project/data/combined_master_dataset.csv
This dataset is now robust enough for rigorous SVD optimization testing.
(bert_svd_env) ➜  Project git:(main) ✗ 