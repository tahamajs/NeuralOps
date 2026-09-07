---
name: doc-code-alignment
description: Merges duplicate/stub Markdown documentation files under Docs/content/attentions/ or Docs/content/compressions/ into one canonical, code-accurate, publication-ready file per module — without ever discarding unique content. Use this whenever the user asks to "clean up", "dedupe", "merge", "align", "audit", or "fix" the docs, wants stub .md files expanded, wants documentation cross-checked against the actual code, or asks to delete short/duplicate .md files (redirect that request through this skill's merge-first workflow instead of deleting outright). Also trigger for "create a doc stub for module X" or "does every module have documentation".
---

# Documentation ↔ Code Alignment

The prime directive: **never delete or truncate content without first proving it's preserved
somewhere else.** Line count, "looks like a duplicate", or "seems outdated" are signals to
investigate, never deletion criteria on their own.

## Step 1 — Build the module ↔ doc mapping

1. List every `.py` module under `Codes/framework/attentions/` and
   `Codes/framework/compression_engine/`.
2. For each, extract every registered name from its
   `@ATTENTION_REGISTRY.register("...")` / `@COMPRESSOR_REGISTRY.register("...")` decorators
   — a module often has multiple aliases (e.g. `wgqa`, `wgqa_attention`).
3. List every `.md` file under `Docs/content/attentions/` and `Docs/content/compressions/`.
4. Match `.md` files to modules by filename/alias substring (e.g. `wgqa.md` and
   `wgqa_attention.md` both map to `wgqa_attention.py`). Build the full
   `module -> [registry names] -> [candidate .md files]` table before editing anything.

## Step 2 — Classify every group of candidate files for one module

- **Exact/near-exact duplicate** (>95% identical text): keep the longer/more complete file as
  canonical. Diff the shorter one and fold in any sentence, parameter, or example present only
  there.
- **Stub vs. expanded doc** (one file is short/thin, the other thorough): extract every fact
  unique to the stub — a registry key, a hyperparameter default, a usage example, a caveat —
  and merge it into the long file before removing the stub.
- **Genuinely different content on the same module** (e.g. one covers the main attention path,
  another covers an internal sub-component like a router or gate): merge into one file with
  clearly separated subsections. Never discard either side just because they don't overlap.
- **Doc references a `.py` file/class/registry key that no longer exists**: fix the reference
  to point at the correct current module, or mark the section clearly as deprecated — never
  silently delete the file.
- **`.py` module has zero matching `.md`**: create a full stub (Step 4 structure), marking
  unfillable sections `TODO`.

## Step 3 — Cross-check the (now-merged) canonical doc against the real code

Open the module's `.py` file and verify/fix:
- Class name(s) and registry key(s) exactly match what's registered.
- Constructor signature and default hyperparameters (rank, lora_rank, lora_alpha, thresholds,
  window sizes, num_experts, etc.) match the real constructor defaults — not stale or invented
  values.
- Mathematical formulas reflect what the code actually computes (masking, RoPE application,
  GQA repeat, SVD/low-rank projection, gating, sparsity — whatever applies).
- Tensor shape traces match the real `.view`/`.transpose`/`.reshape` calls.
- Supported architectures and special handling (GPT-2 Conv1D fused c_attn/c_proj, RoBERTa/BERT
  nested self-attention with bypassed output projection, Qwen/LLaMA GQA + partial RoPE,
  DistilBERT flat q_lin/k_lin/v_lin) are described accurately — remove claims the code doesn't
  support, add support the code has but the doc omits.
- Any Hugging Face link, Triton-kernel claim, or benchmark number is checked against reality:
  link the real private repo (`Hooshaai/svd-linear-attention-<model>-<module>`) and its real
  metrics if a run exists; otherwise mark clearly as "Pending" — **never invent numbers.**
- Where doc and code disagree, code wins. Note in your summary what was corrected and why.

## Step 4 — Required section structure (every canonical .md, in this order)

1. **Title & Overview** — what the module does, key innovation, motivation.
2. **Mathematical Formulation** — exact equations, matching the real code.
3. **Architecture & Data Flow** — component breakdown + step-by-step tensor shape trace.
4. **Configuration & Hyperparameters** — full table matching the constructor exactly.
5. **Complete Source Code Implementation** — the exact, current, verbatim source.
6. **Usage Examples** — correct injection/fine-tuning/benchmarking snippets, real registry name.
7. **Benchmark Results** — real numbers + HF/git-branch links if available, else a clearly
   labeled "Pending actual benchmark results" placeholder table.
8. **Attention Maps & Activations** — what's actually saved and where, or "Planned / Future
   diagnostics" if not implemented.
9. **Hugging Face Model Link** — real private repo URL, or "Not yet uploaded."
10. **Limitations & Failure Modes** — genuine and module-specific — never copy-pasted from an
    unrelated module (a real bug seen in this codebase: SVD/Lyapunov content bleeding into a
    module that isn't SVD-based at all — always sanity-check merged math sections belong to
    *this* module).
11. **Future Work** — realistic next steps.

## Step 5 — Verify before deleting anything

- `python -m py_compile <module>.py` still compiles.
- Confirm every registry name mentioned in the doc actually resolves:
  `ATTENTION_REGISTRY.resolve("<name>")` / `COMPRESSOR_REGISTRY.resolve("<name>")`.
- Only now delete a source `.md` file, and only if: (a) it was a confirmed 100% subset
  (duplicate or stub) of the file you just merged into, and (b) you can point to exactly which
  canonical file now contains everything it had.

## Step 6 — Report

Markdown table: canonical doc path | files merged into it + what unique content came from each
| code/doc mismatches fixed | new sections added / marked TODO | deletions performed (with
justification) | remaining open issues.

Process one module's full file group at a time — attentions first, then compressions. If a
file is already excellent and matches the code, say so explicitly ("no changes required")
rather than editing for the sake of editing.
