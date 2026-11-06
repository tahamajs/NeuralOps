## ACD-03 — Fix the thesis data: impossible F1 values, disagreeing ledgers, stale defense PDF + QA docs — Phase P0 — Findings AC-C01, AC-C03, AC-C04, AC-H05, AC-H06, AC-H07, AC-M03 (+ gap: M6, M19, L5)

**Depends on:** — (no dependency; run in P0 after ACD-01)
**Goal:** No impossible metric value remains in the thesis data; one reproducible results ledger exists; the defense PDF builds clean from the same source as main; and all four QA docs state true numbers with a source per number.
**Context (Persian, 2-4 lines):** داده‌های پایان‌نامه F1 بالای ۱ دارند، دو دفتر نتایج با هم نمی‌خوانند، PDF دفاع کهنه است و اسناد QA عددهای غلط (مدرک ارشد، ۸۱۲ صفحه، سرعت ۱۷۹ برابر) دارند. این پرامپت اول داده را درست می‌کند، بعد سند را.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root; the
thesis lives in /Users/tahamajs/dev/Project/academic/thesis). Only change the
files listed. Do not touch anything else.

## Role + scope
You are a data-forensics + LaTeX engineer fixing the thesis. Your scope is
`academic/thesis/aggregated_metrics.json` (+ the script that built it),
`academic/thesis/master_verified_results.json`,
`academic/thesis/RESULTS_PROVENANCE.md`,
`academic/thesis/810101504_defense.tex`,
`academic/thesis/810101504_main.bbl` + `academic/thesis/chapters/references.tex`,
and the four QA docs (`DEFENSE_READINESS_CERTIFICATE.md`, `QA_REPORT.md`,
`CORRECTION_LOG.md`, `DEFENSE_QA.md`). Run your own multi-sub-agents in
parallel for the independent sub-tasks: (a) one sub-agent traces the F1
computation (find the builder script, recompute); (b) one sub-agent
reconciles the ledgers (match every ledger row to a real result file);
(c) one sub-agent rebuilds the defense PDF and fixes the QA docs. You
synthesize their reports into ONE final report.

## Background
Exact problems (audit AC-C01, AC-C03, AC-C04, AC-H05, AC-H06, AC-H07, AC-M03):
- `aggregated_metrics.json` (256 entries): "**106 entries have F1 > 1.0**,
  e.g. aasvd 147.41, admm 144.52, compressed_attention_layer_energy 203.9".
  F1 lives in [0,1] — these values are impossible as F1.
- `RESULTS_PROVENANCE.md`: "release ledger 42/161/0 FAIL vs paper-run table
  333 PASS / 49 FAIL / 2 TIMEOUT (384 rows)"; "master_verified_results.json
  needs regeneration"; it "cites missing source
  `results/unified_adaptive_span/summary.csv`". The JSON itself says "331
  unique modules: 262 PASS, 7 PASS_LOW, 62 FAIL" vs the blueprint's "0
  failures" for 203 modules.
- Defense PDF: "stale: 10 undefined refs + 3 undefined cites in log; no
  .bbl/.blg"; a fork of main with different margins and the 203-module
  appendix commented out.
- QA docs stale with wrong numbers: MSc degree (it is a BSc), "179.1x" Triton
  speedup, Frobenius "28.68 → 0.00 (100%)", SHA-256 of files in nonexistent
  `thesis_final/`, "RTX 4070 Laptop 8 GB", thesis "812 pp" / "818 pp" (real:
  main.pdf 593 pp), ICLR "62 pp" / NeurIPS "61 pp" (real: 8 / 7 pp),
  "25 سوال" vs "20 سوالی".
- `810101504_main.bbl` (148 entries): "stale (Oct 25) and never `\input`"
  (manual thebibliography renders instead); 3 never-cited keys
  (blelloch1990prefix, gu2020hippo, smith2022simplified).
- `attentions_latex/modules/` has 202 subdirs, not 203; `tehran-thesis.cls`
  suspected unused; `figures_list.txt` is EMPTY (0 B); the appendix symlink
  target is unverified.
- Gap findings (coordinator report §1): M6 — `810101504_defense.tex` (270
  lines) duplicates `810101504_main.tex` with different margins (3/2.5/3 cm
  vs 1.25/1.0/1.1 cm), no pdfauthor, and the 203-module appendix commented
  out. Any fix to main must be repeated in defense.tex — H4 shows it already
  drifted. M19 — the abbreviation table defines `AI` as "Arithmetic
  Intensity" (in an ML thesis most readers read AI as Artificial
  Intelligence); the approval page lists the examiner as "تعیین توسط گروه
  آموزش" (TBD by the department) with blank signature lines, dated شهریورماه
  1405 — a pre-defense template, not a signed approval. L5 —
  `figures_list.txt` is 0 B (already in Task 6) AND CORRECTION_LOG's PPL
  anecdote "4,248,689.50 → 146.41" — 146 is still terrible for GPT-2/WikiText-2
  and the catastrophic figure suggests a bug, not a result.

## Tasks
1. F1 forensics FIRST: find the script that built `aggregated_metrics.json`,
   read it, and determine what each entry really is (metric? unit? percent?).
   Recompute the 106 impossible entries from the underlying results. Then
   EITHER correct the values (with the recomputation evidence) OR rename the
   column to its true meaning (e.g. if it is a percent, "F1 (%)" with range
   0–100). Never guess a value. Regenerate every table/figure derived from
   this file (`app_131_attentions_empirical.tex`,
   `app_72_compressors_empirical.tex`). Add a guard (a small script +
   CI check) that fails if any F1 leaves [0,1].
2. Ledger reconciliation: match every row of `master_verified_results.json`
   and of the paper-run table to a real result file on disk. Find
   `results/unified_adaptive_span/summary.csv` or record it as missing and
   remove the dependency. Build ONE reproducible ledger (script + inputs +
   outputs with hashes) and mark the others "stale". Write the PASS /
   PASS_LOW / PASS_SLOW / FAIL definitions in one place and label every
   number with its module set (203 census vs 331 verified vs 384 rows).
   Fix the blueprint's "0 failures" claim scope in the same change ONLY if
   it is a one-line correction pointing at the ledger (do not rewrite the
   blueprint here — that is ACD-02).
3. Defense PDF: rebuild it as a THIN WRAPPER around main (shared
   preamble/chapters + a flag) — that is the target outcome, not a fork.
   Document the exact differences ONLY if a wrapper is impossible, with the
   reason. Run bibtex so .bbl/.blg exist. Rebuild until the log shows ZERO
   undefined refs/cites. Restore or explicitly drop the commented-out
   203-module appendix (record which).
4. Bibliography: EITHER delete the stale .bbl and state explicitly that the
   bibliography is manual (with the reason) OR return bibtex to the build
   and delete the manual `references.tex`. Two parallel sources are
   forbidden. The 3 never-cited keys are cited or moved to
   `references_unused.bib`.
5. QA docs: rewrite all four with TRUE numbers, each number carrying its
   source (file + line, or the counting command). Degree is BSc. Pages are
   593 / 8 / 7. The "179.1x" speedup and "100%" Frobenius claims are removed
   until they come from a real manifest. All `thesis_final/` references are
   removed. The question count is reconciled (25 vs 20 — count the real
   questions and use that number everywhere).
6. Small items: find the missing 203rd module subdir (which of the 203 is not
   among the 202); verify the appendix symlink with `readlink` + existence
   check; if `tehran-thesis.cls` is unused, move it to `_archive/` (record
   the evidence); fill or delete the empty `figures_list.txt`.
7. M19 + L5: write "Arithmetic Intensity" in FULL on first use (and define
   the abbreviation there — do not leave bare `AI` in the thesis); follow up
   the signed approval page (record whether a signed page exists in the repo
   — if it does not, mark it "unsigned — obtain the signed page" in your
   report, do not fabricate a signature). Verify the PPL anecdote
   ("4,248,689.50 → 146.41"): find the run that produced it; if the
   catastrophic figure was a bug, say so and correct the anecdote; 146 is
   still terrible for GPT-2/WikiText-2, so do not present it as a good result.

## Constraints / do-not-do
- Never invent a metric value, a speedup, or a page count. Never delete a
  result row silently — every removal is listed with the reason. No new
  dependencies. Do not change any scientific conclusion beyond what the
  corrected data forces (if a conclusion changes, say so explicitly in your
  report). Do not touch any file outside `academic/thesis/`.
- Do not commit anything.

## Tests to add/run
- New guard script: `academic/thesis/scripts/check_metrics_range.py` (or the
  existing CI location) asserting every F1 in [0,1]; run it (expected: clean
  after the fix, failing before — record both).
- Commands: the ledger rebuild script (record its hash outputs);
  `pdflatex` + `bibtex` rebuild of main + defense (exit codes + `grep -ci
  undefined` on both logs — expected: 0); `ls attentions_latex/modules/ | wc
  -l` (expected: 203 after the fix).

## Verification (agent MUST run and report)
- The F1 builder script found (path) + the recomputation evidence for the 106 entries (before/after sample, pasted).
- The guard script output — expected: clean.
- The single ledger: script path + input/output hashes, pasted; the stale ledgers marked.
- Defense + main rebuild logs — `grep -ci undefined` on both, expected: 0.
- QA docs: every corrected number with its source, listed.
- `git diff --stat` and `git status --short` — confirm only the listed files changed.
- Checklist: [ ] no F1 outside [0,1] [ ] one ledger, reproducible [ ] missing source found or removed [ ] defense builds clean [ ] bibliography has one source [ ] QA docs true with sources [ ] 203 module subdirs [ ] empty file handled [ ] defense is a thin wrapper (not a fork) [ ] "Arithmetic Intensity" spelled out + jury page followed up [ ] PPL anecdote verified (bug or result, stated).

## Definition of done
- Every thesis number is either recomputed from real data or removed with a reason.
- One ledger, one bibliography source, one thesis source (main + defense).
- All four QA docs are true.

## Report back
- Files changed, deleted, archived (exact paths).
- Per F1 entry group: what the values really were and what you did.
- The ledger reconciliation table (old numbers vs new single ledger).
- Per QA doc: the numbers corrected, with sources.
- Any conclusion that changed because the data changed.
- Any new finding.

## Stop conditions
- If the F1 builder script cannot be found — STOP and report it; do not guess the values.
- If a ledger row cannot be matched to a real result file — STOP for that row and report it; do not delete it silently.
- If pdflatex/bibtex is not installed on this host — STOP and report which host has it.
- If a test that passed before now fails — STOP, do not improvise, report exactly what is blocked.
```

