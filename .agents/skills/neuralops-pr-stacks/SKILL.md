---
name: neuralops-pr-stacks
description: Architect, rebase, and maintain stacked pull requests with clean branching strategies, atomic review chunks, and native GitHub stack integration.
---

# NeuralOps Stacked Pull Request Engineering and Maintenance

Use this skill when decomposing large architectural changes, multi-layer kernel optimizations, or cross-cutting features into small, atomic, dependent pull requests. Stacking PRs accelerates code review velocity, isolates blast radius, preserves `git bisect` integrity, and ensures fast CI cycle times. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) for scope limits and [neuralops-pr-readiness](../neuralops-pr-readiness/SKILL.md) for individual PR readiness.

---

## 1. Branching Strategy & Stack Topology

A well-structured PR stack is a directed acyclic chain of Git branches rooted at `main`. Each child branch builds strictly upon the commits of its immediate parent:

```
main (Trunk)
  │
  └── feat/foundation (Layer 0: Types, Constants, Registry definitions)
        │
        └── feat/cuda-kernel (Layer 1: Native C++ & Triton JIT implementations)
              │
              └── feat/autograd-dispatch (Layer 2: 3-level dispatch & AutogradBridge)
                    │
                    └── feat/attention-module (Layer 3: User-facing Attention class & docs)
```

### 1.1 Atomic Review Chunks: The 5-Layer Stack Rule
Do not submit monolithic 2,000-line pull requests. Partition work into discrete architectural layers:

```
+-----------------------------------------------------------------------------------------------+
|                               NeuralOps 5-Layer PR Stack Taxonomy                             |
+-------+-------------------------+-------------------------------------------------------------+
| Layer | Scope                   | Typical Changes & Artifacts                                 |
+-------+-------------------------+-------------------------------------------------------------+
| 0     | Foundation & Interfaces | Dtype constants, registry keys, abstract base interfaces    |
+-------+-------------------------+-------------------------------------------------------------+
| 1     | Hardware Kernel Core    | CUDA C++ (.cu), Triton JIT kernels, C++ extension bindings  |
+-------+-------------------------+-------------------------------------------------------------+
| 2     | Dispatch & Autograd     | 3-level dispatch wrappers, verifiers, AutogradBridge, tests |
+-------+-------------------------+-------------------------------------------------------------+
| 3     | High-Level Modules      | nn.Module integration in neuralops.attentions or compression|
+-------+-------------------------+-------------------------------------------------------------+
| 4     | Model Surgery & E2E     | Pretrained weight bridging, HuggingFace config adapters, CI |
+-------+-------------------------+-------------------------------------------------------------+
```

Each PR in the stack must compile, pass its own unit tests, and maintain 100% repository stability independently—without relying on incomplete code in future stacked PRs.

---

## 2. Step-by-Step Stack Creation Workflow

### Step 1: Create the Foundation Branch
```bash
# Ensure local main is synchronized
git checkout main
git pull --ff-only origin main

# Branch Layer 0: Foundation
git checkout -b feat/sliding-window-foundation
# Make atomic changes: types, signatures, registry keys
git add neuralops/core/types.py neuralops/core/registry.py
git commit -m "core: define sliding window attention types and registry keys"
git push -u origin feat/sliding-window-foundation
```

### Step 2: Branch Layer 1 from Layer 0
```bash
# Branch Layer 1 off Layer 0
git checkout -b feat/sliding-window-kernel feat/sliding-window-foundation
# Implement native CUDA C++ and Triton JIT kernels
git add neuralops/core/cuda_kernels/sliding_window/
git commit -m "kernels: implement sliding window CUDA and Triton JIT kernels"
git push -u origin feat/sliding-window-kernel
```

### Step 3: Branch Layer 2 from Layer 1
```bash
# Branch Layer 2 off Layer 1
git checkout -b feat/sliding-window-dispatch feat/sliding-window-kernel
# Implement 3-level fallback dispatcher and autograd bridges
git add neuralops/core/cuda_kernels/sliding_window/sliding_autograd.py tests/
git commit -m "dispatch: wire 3-level fallback and autograd tests for sliding window"
git push -u origin feat/sliding-window-dispatch
```

---

## 3. Advanced Rebase Workflows & Stack Surgery

When an intermediate branch in your stack is modified (e.g., following code review feedback on Layer 0) or `main` moves ahead, upper branches must be rebased cleanly using `git rebase --onto`.

```
Before Rebase:
main ──── A ─── B
          │
          └── Layer 0 ─── C ─── D (Updated with D')
                            │
                            └── Layer 1 ─── E ─── F

Target: Move Layer 1 to sit squarely on top of the revised Layer 0 (D')
Command: git rebase --onto feat/sliding-window-foundation feat/sliding-window-foundation@{1} feat/sliding-window-kernel
```

### 3.1 Propagating Changes Up the Stack
When you amend or add commits to a lower layer:

```bash
# 1. Update Layer 0
git checkout feat/sliding-window-foundation
# Apply review suggestions
git commit -am "review: tighten verifier head dimension check"
git push --force-with-lease origin feat/sliding-window-foundation

# 2. Rebase Layer 1 onto the updated Layer 0
git checkout feat/sliding-window-kernel
git rebase feat/sliding-window-foundation
# Resolve conflicts if any, then:
git push --force-with-lease origin feat/sliding-window-kernel

# 3. Rebase Layer 2 onto the updated Layer 1
git checkout feat/sliding-window-dispatch
git rebase feat/sliding-window-kernel
git push --force-with-lease origin feat/sliding-window-dispatch
```

### 3.2 Rebasing the Entire Stack onto a New `main`
When upstream `main` incorporates new commits:
```bash
# Update local main
git checkout main
git pull --ff-only origin main

# Rebase Layer 0 onto new main
git checkout feat/sliding-window-foundation
git rebase main
git push --force-with-lease origin feat/sliding-window-foundation

# Rebase Layer 1 onto Layer 0
git checkout feat/sliding-window-kernel
git rebase feat/sliding-window-foundation
git push --force-with-lease origin feat/sliding-window-kernel

# Rebase Layer 2 onto Layer 1
git checkout feat/sliding-window-dispatch
git rebase feat/sliding-window-kernel
git push --force-with-lease origin feat/sliding-window-dispatch
```

### 3.3 Verifying Incremental Diffs Across Layers
Always inspect the pure incremental diff between adjacent branches to ensure no commits were accidentally duplicated or leaked:
```bash
# Verify Layer 0 contains only foundation changes against main
git diff main...feat/sliding-window-foundation

# Verify Layer 1 contains only kernel changes against Layer 0
git diff feat/sliding-window-foundation...feat/sliding-window-kernel

# Verify Layer 2 contains only dispatch/autograd changes against Layer 1
git diff feat/sliding-window-kernel...feat/sliding-window-dispatch
```

---

## 4. Managing GitHub Native PR Stacks

NeuralOps supports native GitHub Stacked Pull Requests. Each pull request in GitHub is configured so that its **base branch** points to the branch immediately preceding it in the stack.

```mermaid
graph LR
    Main["main (Trunk)"]
    PR1["PR #101: feat/foundation<br/>Base: main"]
    PR2["PR #102: feat/kernel<br/>Base: feat/foundation"]
    PR3["PR #103: feat/dispatch<br/>Base: feat/kernel"]

    Main <-- PR1
    PR1 <-- PR2
    PR2 <-- PR3
```

### 4.1 Submitting and Linking the Stack
Using the GitHub CLI (`gh`), create each PR targeting its corresponding base:

```bash
# 1. Create PR for Layer 0 against main
gh pr create \
  --base main \
  --head feat/sliding-window-foundation \
  --title "core(types): sliding window attention types and interfaces" \
  --body "Layer 0 of sliding window stack. Introduces type definitions."

# 2. Create PR for Layer 1 against Layer 0
gh pr create \
  --base feat/sliding-window-foundation \
  --head feat/sliding-window-kernel \
  --title "kernels(cuda): sliding window native CUDA and Triton kernels" \
  --body "Layer 1 of sliding window stack. Depends on #101."

# 3. Create PR for Layer 2 against Layer 1
gh pr create \
  --base feat/sliding-window-kernel \
  --head feat/sliding-window-dispatch \
  --title "dispatch(sliding): 3-level fallback and autograd bridge" \
  --body "Layer 2 of sliding window stack. Depends on #102."
```

### 4.2 Verifying Stack Membership via REST API
Verify that the stack links and base references are recognized by GitHub:
```bash
gh api "repos/tahamajs/NeuralOps/pulls/102" \
  --jq '{number, head_branch: .head.ref, base_branch: .base.ref, state: .state}'
```

### 4.3 Merging Stacked PRs (Bottom-to-Top Order)
When approved, stacked PRs must be merged strictly in **bottom-to-top order**:
1. **Merge PR #101** into `main`.
2. GitHub automatically retargets PR #102 to point its base to `main`.
3. Confirm PR #102 CI runs cleanly against `main`.
4. **Merge PR #102** into `main`.
5. Repeat for subsequent layers until the complete feature is merged.
