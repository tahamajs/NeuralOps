## ACD-01 — Fix the arXiv build: zero undefined cites/refs, clean aux, remove orphaned expanded sections — Phase P0 — Findings AC-C02, AC-H02, AC-M06

**Depends on:** — (no dependency; run first)
**Goal:** `academic/paper/arxiv/main.pdf` builds clean from scratch (pdflatex + bibtex + pdflatex ×2) with exit 0, zero undefined citations, zero undefined references, zero overfull >5pt — and the three orphaned expanded sections are merged or archived.
**Context (Persian, 2-4 lines):** پی‌دی‌اف ۱٬۴۱۶ صفحه‌ای با ۸۸ ارجاع تعریف‌نشده و ۵۱ مرجع تعریف‌نشده ساخته شده — یعنی در متن «؟» هست. این قدم اول build را تمیز می‌کند و سه بخش تکراری یتیم را هم جمع می‌کند.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
academic tree lives in /Users/tahamajs/dev/Project/academic). Only change the
files listed. Do not touch anything else.

## Role + scope
You are a LaTeX build engineer fixing the arXiv monograph build. Your scope
is `academic/paper/arxiv/main.tex`, `academic/paper/arxiv/sections/` (the
three orphaned `*_expanded.tex` files + any section with an undefined
`\label`/`\ref`), `academic/paper/arxiv/references.bib` (ADD missing entries
only — do not delete anything here; deletion is ACD-04), and the build
artifacts you regenerate. Run your own multi-sub-agents in parallel for the
independent sub-tasks: (a) one sub-agent diffs each orphaned expanded section
against its main section; (b) one sub-agent collects every undefined cite/ref
from a fresh build log; (c) one sub-agent cleans the build artifacts. You
synthesize their reports into ONE final report — do not let sub-agents write
separate final reports.

## Background
Exact problems (audit AC-C02, AC-H02, AC-M06, 2026-11-07):
- `academic/paper/arxiv/main.pdf` (10,799,219 B, 1,416 pp) was "built from
  stale aux: log shows 88 undefined cites + 51 undefined refs" (`main.log`
  434,277 B; 796 Overfull, 1,762 Underfull). `main.bbl` has 96 entries vs 97
  cited keys. The blueprint's own checklist demands "0 undefined citations
  (no `?`), 0 undefined references, 0 overfull \hbox (>5pt)".
- Three orphaned near-duplicate sections never `\input`:
  `sections/sec01_introduction_expanded.tex` (0.985 similar to sec01),
  `sections/sec02_related_taxonomy_expanded.tex` (0.526 similar),
  `sections/sec04_system_architecture_expanded.tex` (0.964 similar).
- Build junk in the repo: `main.xdv` (22,349,804 B), `test_*.tex/.pdf/.aux/.log`
  test-build artifacts.

## Tasks
1. Delete the stale aux/bbl (`main.aux`, `main.bbl`, `main.blg`, `main.log`,
   `main.out`, `main.toc`, `main.lof`, `main.lot`) and rebuild from scratch:
   `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex
   main.tex`. Record every command's exit code. If pdflatex/bibtex is not
   installed on this host — STOP and report which host has the LaTeX
   toolchain; do not fake a build.
2. From the fresh `main.log`, extract EVERY undefined citation and every
   undefined reference into a list (file:line + key/label). For each one:
   EITHER add the missing bib entry (with a real, verifiable reference —
   never invent a bib entry) OR remove the `\cite`/`\ref` from the text.
   Record the choice per item. Rebuild until the log shows ZERO undefined
   cites and ZERO undefined refs.
3. Fix overfull `\hbox` > 5pt (all 796 of them, or as many as exist after the
   rebuild — record the before/after counts). Underfull warnings are recorded
   but need not all be fixed.
4. For each orphaned expanded section: diff it against its main section.
   Merge any UNIQUE content into the main section (with a one-line note in
   your report of what was merged), then move the expanded file to
   `academic/paper/arxiv/sections/_archive/`. If a section has NO unique
   content, archive it without merging and say so.
5. Move build junk out of the repo: add `*.xdv` and `test_*` build outputs to
   the appropriate `.gitignore` (or delete the files if they are tracked —
   record which) so a fresh clone does not carry 22 MB of .xdv.

## Constraints / do-not-do
- Never invent a bib entry, a label, or a number. Never delete a `\cite`
  silently — every removal is listed in your report with the reason. No new
  dependencies. Do not change any section's scientific content beyond the
  merges in Task 4. Do not touch any file outside `academic/paper/arxiv/`.
- Do not commit anything.

## Tests to add/run
- No new unit tests in this prompt. The verification IS the test: the fresh
  build + the log greps below.
- Commands: the full rebuild (Task 1); `grep -c "undefined" main.log`
  (expected: 0); `grep -c "Overfull \\\\hbox" main.log` (before/after);
  `grep -rn "\\\\input.*expanded" academic/paper/arxiv/sections/`
  (expected: 0 — the files are archived, not input).

## Verification (agent MUST run and report)
- Rebuild exit codes, pasted (all must be 0).
- `grep -i "undefined" main.log` — expected: zero matches; paste the grep output.
- Overfull count before vs after, pasted.
- `ls academic/paper/arxiv/sections/_archive/` — the three files, listed.
- `git diff --stat` and `git status --short` — confirm only the listed files changed.
- Checklist: [ ] clean rebuild exit 0 [ ] 0 undefined cites [ ] 0 undefined refs [ ] overfull >5pt fixed or listed [ ] 3 expanded sections merged/archived [ ] build junk ignored/removed.

## Definition of done
- `main.pdf` rebuilds from scratch with a clean log (0 undefined cites/refs).
- No orphaned expanded section remains in `sections/`.
- No 22 MB .xdv or test-build junk in the repo.

## Report back
- Files changed, deleted, archived (exact paths).
- Per undefined cite/ref: fixed by adding (bib key) or removing (file:line + reason).
- Per expanded section: merged (what) or archived as-is.
- Build log evidence (grep outputs + overfull before/after).
- Any new finding.

## Stop conditions
- If pdflatex/bibtex is not installed on this host — STOP and report which host has it.
- If an undefined citation cannot be resolved to a real reference — STOP for that item and report it; do not invent a bib entry.
- If a test that passed before now fails — STOP, do not improvise, report exactly what is blocked.
```

