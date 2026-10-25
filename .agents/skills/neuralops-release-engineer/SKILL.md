---
name: neuralops-release-engineer
description: "Prepare, validate, package, tag, and publish official NeuralOps distribution releases."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "release", "pypi", "packaging", "git"]
---

# Skill: Release Engineer

## When to use
Trigger when preparing, validating, and publishing an official package release to PyPI and GitHub.

## Prerequisites
- Clean git working tree on `main`
- Build tools: `build`, `twine`
- PyPI credentials configured in environment or `.pypirc`

## Steps
1. **Synchronize Package Version**:
   Update version in single source of truth `neuralops/__init__.py:__version__` and propagate:
   ```bash
   python3 scripts/utils/sync_version.py
   ```
2. **Build Distribution Packages**:
   Build sdist and wheel:
   ```bash
   python -m build
   ```
3. **Verify Package Metadata and Size**:
   Validate package distributions:
   ```bash
   twine check dist/*
   ```
   Verify wheel bundle size remains strictly `< 15 MB` (excluding raw weights and artifacts).
4. **Create Annotated Git Tag**:
   Create tag matching the release version:
   ```bash
   git tag -a v1.3.7 -m "Release v1.3.7: NeuralOps Canonical Release"
   ```
5. **Publish Release**:
   Push tag to remote repository and upload to PyPI:
   ```bash
   git push origin v1.3.7
   twine upload dist/*
   ```

## Verification
- Verify package appears on PyPI and can be installed via `pip install neuralops==<version>`.

## Common pitfalls
- Manually editing downstream version files instead of using `scripts/utils/sync_version.py`.
- Including large binary test artifacts or shared libraries causing wheel size to exceed 15 MB.
