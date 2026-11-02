# Skill Evaluation Matrix: Scoring & Installation Decisions

## Evaluation Criteria and Weightings
- **Relevance to NeuralOps (30%)**: Utility for generating reports from audit files and master ledger data.
- **Quality of Instructions (25%)**: Actionable steps, precise prerequisites, verification scripts, and anti-patterns.
- **LaTeX Integration (20%)**: Compatibility with XeLaTeX, booktabs, longtable, and thesis manuscript structure.
- **Provenance Support (15%)**: Capacity to trace numbers to Git SHAs, timestamps, and record IDs.
- **Maintainability (10%)**: Clean modular structure adhering to project conventions.

## Decision Rule
- **Score $\ge$ 80%**: ADAPT AND INSTALL
- **Score 60% – 79%**: EXTRACT PARTS AND MERGE
- **Score < 60%**: REJECT

---

## Detailed Evaluation Table

| # | Discovered Skill | Rel (30) | Qual (25) | LaTeX (20) | Prov (15) | Maint (10) | Total Score | Final Decision |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `empirical-paper-writer` | 29 | 24 | 19 | 14 | 9 | **95%** | **ADAPT AND INSTALL** |
| 2 | `scientific-writing` | 28 | 24 | 18 | 14 | 9 | **93%** | **ADAPT AND INSTALL** |
| 3 | `latex-rhythm-refiner` | 26 | 23 | 19 | 13 | 8 | **89%** | **ADAPT AND INSTALL** |
| 4 | `academic-humanizer` | 28 | 23 | 18 | 13 | 9 | **91%** | **ADAPT AND INSTALL** |
| 5 | `experimental-design` | 27 | 24 | 17 | 14 | 9 | **91%** | **ADAPT AND INSTALL** |
| 6 | `scientific-visualization` | 29 | 24 | 19 | 14 | 9 | **95%** | **ADAPT AND INSTALL** |
| 7 | `latex-compilation-diagnostics` | 27 | 25 | 20 | 12 | 9 | **93%** | **ADAPT AND INSTALL** |
| 8 | `bibtex-citation-provenance` | 28 | 23 | 19 | 15 | 9 | **94%** | **ADAPT AND INSTALL** |
| 9 | `claim-evidence-auditor` | 30 | 24 | 18 | 15 | 9 | **96%** | **ADAPT AND INSTALL** |
| 10 | `roofline-visualizer-generator` | 29 | 23 | 18 | 13 | 9 | **92%** | **ADAPT AND INSTALL** |
| 11 | `abstract-refiner` | 20 | 18 | 15 | 10 | 7 | 70% | MERGE INTO #1 |
| 12 | `arxiv-uploader` | 10 | 15 | 12 | 8 | 5 | 50% | REJECT |
| 13 | `systematic-literature-review` | 22 | 19 | 14 | 10 | 7 | 72% | MERGE INTO #2 |
| 14 | `reviewer-response-generator` | 12 | 16 | 12 | 6 | 6 | 52% | REJECT |
| 15 | `latex-table-beautifier` | 23 | 19 | 18 | 8 | 6 | 74% | MERGE INTO #1 |
| 16 | `pseudocode-reviewer` | 20 | 18 | 16 | 9 | 7 | 70% | MERGE INTO #1 |
| 17 | `paper-from-zero` | 21 | 18 | 15 | 10 | 7 | 71% | MERGE INTO #1 |
| 18 | `research-presentation` | 10 | 14 | 10 | 6 | 5 | 45% | REJECT |
| 19 | `statistical-hypothesis-testing`| 21 | 19 | 13 | 11 | 7 | 71% | MERGE INTO #5 |
| 20 | `figure-caption-writer` | 22 | 20 | 16 | 10 | 7 | 75% | MERGE INTO #6 |
