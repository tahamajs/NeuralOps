# Skills Collection — Paper-Ready Machine Learning

A structured set of skills for producing NeurIPS/ICML/ICLR/ACL-quality papers
from transformer compression and attention research.

## What is a skill?

A skill is a self-contained markdown file that describes:
- **What** to do
- **Why** it matters
- **How** to do it (with code)
- **Where** to save the output
- **How** to verify success

## Organization

Each skill lives in its own `.md` file inside a category folder. Every folder
has a `SKILL.md` that lists its contents and prerequisites.

## Usage

Each skill file is designed to be read by a coding agent (e.g., Claude Code,
Cursor, Aider) and executed against a project. The skill tells the agent
exactly what to produce.

## Categories

| Category | Purpose |
|---|---|
| `paper-writing/` | Generate paper sections (abstract, intro, method, ...) |
| `experiments/` | Run ablations and sweeps |
| `metrics/` | Compute statistical and diagnostic metrics |
| `figures/` | Generate publication-quality plots |
| `tables/` | Generate paper-ready CSV/Markdown tables |
| `reproducibility/` | Capture environment, seeds, hashes |
| `analysis/` | Deep-dive analysis (failures, qualitative, robustness) |
| `publishing/` | Upload checkpoints and cite results |
