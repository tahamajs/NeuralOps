---
name: neuralops-notebook-architect
description: "Author, execute, and validate reproducible Jupyter demonstration tutorials on GPU."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "notebook", "jupyter", "tutorials"]
---

# Skill: Notebook Architect

## When to use
Trigger when writing, updating, or validating interactive Jupyter notebooks in `notebooks/`.

## Prerequisites

## Steps
1. **Adhere to Master Notebook Template**:
   Structure notebook cells following `notebooks/00_master_*.ipynb` (preamble, architecture visualizer, kernel dispatch, memory profile, summary).
2. **Execute In-Place on Remote GPU**:
   Run the notebook end-to-end to generate persistent cell outputs:
   ```bash
   ```
3. **Verify Output Integrity**:
   Ensure cells contain non-empty execution outputs, rendered figures, and zero uncaught exceptions.
4. **Synchronize Artifact to Host**:
   Transfer the executed notebook back to the local repository:
   ```bash
   ```

## Verification
- Inspect the notebook JSON to verify cell execution counts and image mime-types exist.

## Common pitfalls
- Committing empty or stripped notebooks without saved execution outputs.
- Executing notebooks locally on macOS where CUDA acceleration fails.
