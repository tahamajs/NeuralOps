---
name: codebase-health-audit
description: Runs a full, evidence-backed health audit of this research repository — broken imports/references, duplicated code across compressor/attention modules, hardcoded machine-specific paths, oversized git history, inconsistent module counts across README/paper/docs, and methodological risks in the benchmark pipeline — then produces a prioritized, phased remediation plan with verification steps for each fix. Use this whenever the user asks to "analyze the project", "find problems", "audit the codebase", "what's broken", "clean up before submission", or wants a report on repo/project health. Every claim in the report must be backed by a concrete command output or file:line citation — never assert a problem exists without having actually checked.
---

# Codebase Health Audit

The output of this skill is only as good as its evidence. **Every problem listed must be
verified with an actual command or file read before it goes in the report** — grep counts,
import attempts, file-existence checks. Do not report suspected issues as confirmed ones;
label anything unverified as "needs investigation" separately.

## Step 1 — Baseline facts

- `git log -1 --oneline`, `git status -s`, commit count and date range.
- `du -sh .git` (repo bloat check).
- `git ls-files | grep -c "__pycache__\|\.DS_Store\|db.sqlite3"` (tracked junk check).
- Largest tracked files: `git ls-files -z | xargs -0 -I{} du -k "{}" 2>/dev/null | sort -rn | head -20`.

## Step 2 — Broken references (things that will fail if run today)

- Every `import` in the most-relied-on entry scripts (main sweep runner, README-documented
  commands, Dockerfile `CMD`) — actually attempt them (`python -c "import ..."`) rather than
  eyeballing the path.
- Every command/script filename mentioned in `README.md` — confirm the file exists in the
  repo (`find . -name "<filename>"`).
- The `Dockerfile`'s `CMD`/`ENTRYPOINT` target file — confirm it's committed.
- Any file referencing a path that no longer exists after a recent refactor.

## Step 3 — Structural duplication

- Count how many files redefine the same helper function name across the module directories
  (e.g. `grep -rln "def _cfg_get" <dir> | wc -l`). A helper duplicated across dozens of files
  is a maintenance/drift risk — name the exact count, not "many".
- Count competing base-class definitions (`grep -rn "class BaseCompressor\|class BaseAttention"`),
  which signal divergent contracts.
- Count silent-failure patterns: `grep -rn "except Exception" <dir> | wc -l` and
  `grep -rn "except:" <dir> | wc -l`. Flag `except: continue`/`except Exception: pass` patterns
  specifically inside weight-copy or compression paths — those can make a broken run silently
  report as passing.
- Identify any monolithic file (very high line count, huge single function) that's a natural
  refactor target — cite the actual line count.

## Step 4 — Portability and reproducibility

- `grep -rn` for hardcoded absolute paths (drive letters, `/Users/<name>/`,
  `C:\Users\...`) across the codebase, especially in benchmark driver scripts.
- Compare `requirements.txt`/pinned dependency versions against what CI actually pins —
  flag any mismatch (loose floor vs. exact CI pin) as a drift risk.
- Confirm `.gitignore` actually excludes virtual envs, datasets, checkpoints, and large result
  artifacts; check whether any large result files (raw detailed results, PDFs) are tracked
  despite that.

## Step 5 — Documentation/reporting consistency

- Grep for module/architecture counts across README badges, the paper draft, and any
  server/setup guide docs. If they don't match each other or a canonical source-of-truth file,
  flag it explicitly with each conflicting number and its location.
- Check for stray audit/report files sitting loose at a docs root that should be organized
  into a dated reports folder.
- Check for authorship/venue inconsistencies across README, paper title block, and any
  submission-variant folders.

## Step 6 — Methodological risks (only if this is a research/benchmark repo)

- Does the reported "compression rate" or similar headline metric net out fallback/skipped
  layers, or does it overstate effectiveness? Check the aggregation code, not just the CSV
  column name.
- Is evaluation done on a single dataset/task across all methods? If so, note it as a
  limitation a reviewer would raise, and suggest a small representative multi-task subset
  rather than a full re-run of everything.

## Step 7 — Produce the report

Structure: **A. Confirmed defects (broken today)** → **B. Structural debt** → **C. Portability
debt** → **D. Repo hygiene** → **E. Documentation consistency** → **F. Methodological risks**.
For every item: cite exact evidence (file:line or command output), state impact, give a
concrete fix with an estimated effort, and give a verification command that proves the fix
worked.

End with a **prioritized, phased execution plan** (e.g. "Phase 1 — today: fix what's actually
broken" → "Phase 2 — this week: stop divergence" → "Phase 3 — next week: deduplicate" →
"Phase 4 — before submission: consistency + methodology"), and an explicit **out-of-scope**
section for things intentionally not worth fixing right now (e.g. a separate, working
sub-product that doesn't need refactoring while the core research is still open).

Do not silently skip a phase's verification step — if a fix can't be verified, say so rather
than marking it done.
