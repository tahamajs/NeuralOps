## ACD-04 — Citation cleanup: prune the dead bib entries (147 arxiv, 226 ICLR, 229 NeurIPS/ICML, 232 MLSys) — Phase P0 — Findings AC-H01 (+ gap: CIT 12 citation fixes, L6/CIT-AUDIT)

**Depends on:** — (no dependency; run in P0, before the final build verification in ACD-FINAL)
**Goal:** Every `.bib` file contains only entries that are actually cited (plus a separate archived `references_unused.bib`); the two conflicting uncited-count runs are reconciled to one recount.
**Context (Persian, 2-4 lines):** از ۲۴۴ ورودی bib تک‌نگاشت فقط ۹۷ تا cite شده؛ در ICLR از ۲۳۸ تا فقط ۱۲ تا. bib مرده build را کند و بازبینی را سخت می‌کند. این پرامپت آن را هرس می‌کند — بدون حذف کامل، به آرشیو می‌رود.
**Prompt to paste (English):** ```
You are working in /Users/tahamajs/dev/Project (the NeuralOps repo root).
Only change the files listed. Do not touch anything else.

## Role + scope
You are a bibliography engineer. Your scope is
`academic/paper/arxiv/references.bib`,
`academic/paper/iclr2027-submission/iclr2027_conference.bib`, the NeurIPS /
ICML / MLSys bib files, the new `references_unused.bib` archive files, and
`academic/paper/audit/` (the two old audit files — annotation only, Task 6).
Run your own multi-sub-agents in parallel: one sub-agent per venue bib
(arxiv, ICLR, NeurIPS, ICML, MLSys) — each counts cited vs uncited for its
own file and produces its own prune list. You synthesize the five lists into
ONE final report and apply the prunes consistently.

## Background
Exact numbers (audit AC-H01; `uncited_bib_lists.txt` contains TWO counting
runs that disagree by 6 on every venue — the second run matches the 238-entry
ICLR bib and is the consistent one, but you must recount with ONE script):
- arXiv: bib=244, cited=97, uncited=147 (INVENTORY says 148 — recount).
- ICLR: cited=12, uncited=226 of 238 (only 5% used).
- NeurIPS: cited=9, uncited=229. ICML: cited=9, uncited=229. MLSys: cited=6,
  uncited=232.
- Suspicious keys to check individually: `audit01`–`audit05`, `a3_ours`,
  `salt_2024`, `neuralops_telemetry_2026`, `neuralops_results_hf` — internal
  placeholders or real references?
- Gap findings (coordinator report §4): CIT — 12 dead/wrong citations with
  CORRECT values already established: (1) key `sun2024simple` (Wanda) does
  not exist — correct key is `sun2024wanda` (`.bib:276`), arXiv:2306.11695.
  (2) key `ba2016layer` (LayerNorm) does not exist at all — add Ba, Kiros &
  Hinton, arXiv:1607.06450. (3) Duplicate LoRA entries `hu2022lora` /
  `hu2021lora` byte-identical — keep `hu2022lora` (arXiv:2106.09685, ICLR
  2022); retarget the `.tex` citation first (it cites `hu2021lora`). (4)
  `lan2020albert` author field is "Lan, Zhenzhong and others" — correct
  authors: Lan, Chen, Goodman, Gimpel, Sharma, Soricut; arXiv:1909.11942.
  (5) `jiao2020tinybert` author field "Jiao, Xiaoqi and others" — correct
  authors: Jiao, Yin, Shang, Jiang, Chen, Li, Wang, Liu; arXiv:1909.10351.
  (6) `sanh2019distilbert` venue "NeurIPS Workshop 2019" is wrong —
  DistilBERT was an arXiv preprint (arXiv:1910.01108). (7)
  `eckart1936approximation`: DOI 10.1007/BF02288367 is real and correct, but
  the `.bib` entry has **no `doi` field** — add it. (8) `lin2023awq`: key year
  vs MLSys 2024 entry — not wrong as an identifier; rename to `lin2024awq`
  only if normalising keys. (9) `radford2019language` (GPT-2): `@article,
  journal = {OpenAI Blog}` with no URL — add
  `url = {https://openai.com/index/better-language-models/}` or re-type as
  `@misc`. (10) The old audit's source paths all nonexistent — correct paths:
  `academic/paper/iclr2027-submission/iclr2027_conference.bib` (238 entries)
  and `.../iclr2027_conference.tex` (cites 12 keys). (11) README §27 BibTeX
  (`README.md:2417-2434`): `author = {Majlesi, Taha and contributors}` —
  "and contributors" is not a person and is invalid as a BibTeX author; title
  differs from `CITATION.cff`; no DOI, no URL field. Correct author to
  `Majlesi, Taha`, title to match CFF, add `url` and the DOI once minted.
  (12) README monograph titles L286-288: ICLR title does not match the
  manuscript — correct title per `iclr2027_conference.tex`: "NeuralOps: A
  Unified Framework for In-Place Attention Surgery, Riemannian SVD
  Compression, and Representation Stability". L6/CIT-AUDIT — the prior
  citation audits are vacuous, not verified:
  `_CITATION_AUDIT.md` (2026-09-20, "COMPLETE"): its cited source paths **do
  not exist** (`academic/paper/build_paper/iclr2027_conference.bib`,
  `.../13_references.tex`, `academic/paper/REFERENCES.bib`); its 13-row table
  is partly wrong (2 of 13 keys do not exist in the real `.bib`); LoRA
  appears as two duplicate entries; ALBERT/TinyBERT author lists truncated
  to "and others"; the Eckart DOI is not in the `.bib` entry; the DistilBERT
  venue is wrong. `_AUDIT_DEAD_CITATIONS.md` ("Total flagged papers: 0") is
  vacuous: no method, no list of papers checked, no URLs attempted, no date,
  no evidence. L6: the prior audits claim "0" only because they never
  checked uncited entries.

## Tasks
1. Write ONE counting script (`academic/paper/scripts/count_citations.py`,
   new file) that, for a given .tex + .bib pair, reports bib entries, cited
   keys, uncited keys, and undefined cites (cited but not in bib). Run it on
   all five venues and record the single final numbers (this settles the
   147-vs-148 and 226-vs-232 discrepancies — quote the script output).
2. For each venue: move the uncited entries to a sibling
   `references_unused.bib` (same directory) — DO NOT delete them outright.
   Keep the cited entries in place, byte-identical.
3. Check every suspicious key (listed above): if it is an internal
   placeholder, either cite it properly (add the real reference) or move it
   to the unused file with a note; if it is a real reference that SHOULD be
   cited, add the missing `\cite` (and say where) or move it to unused.
   Record the decision per key.
4. Rebuild the arXiv PDF (same commands as ACD-01) and confirm the log is
   still clean (0 undefined cites — the prune must not remove a cited entry).
5. CIT — Apply the 12 citation fixes from the Background as an explicit task
   list (one sub-agent): (1) retarget `sun2024simple` cites to `sun2024wanda`;
   (2) add the `ba2016layer` entry; (3) merge the duplicate LoRA entries
   (keep `hu2022lora`, retarget the `.tex` first); (4) fix the ALBERT author
   list; (5) fix the TinyBERT author list; (6) fix the DistilBERT venue to
   arXiv:1910.01108; (7) add the Eckart DOI field; (8) rename `lin2023awq`
   only if normalising keys (record the decision); (9) fix the GPT-2 entry
   (add the URL or re-type as `@misc`); (10) record the correct audit source
   paths in your report; (11) fix the README §27 BibTeX (author `Majlesi,
   Taha`, title to match CFF, add `url` + DOI once minted — the DOI itself is
   core-owned, AGT-61); (12) fix the README monograph titles to the
   manuscript title. Rebuild after the fixes and confirm 0 undefined cites.
6. L6/CIT-AUDIT — Annotate BOTH old audit files
   (`academic/paper/audit/_CITATION_AUDIT.md` and
   `_AUDIT_DEAD_CITATIONS.md`) with a `superseded` header stating: the
   source paths do not exist, the "0 flagged" result was vacuous (no method,
   no evidence), and the real numbers are the ACD-04 recount. Do not delete
   the files — annotate them.

## Constraints / do-not-do
- Never delete a bib entry outright — archive to `references_unused.bib`.
  Never remove a cited entry. Never invent a reference. Do not change any
  .tex content except adding a missing `\cite` where Task 3 requires it (and
  only there). Do not touch any file outside `academic/paper/`.
- Do not commit anything.

## Tests to add/run
- New script: `academic/paper/scripts/count_citations.py` (Task 1) + a CI
  check that fails if uncited entries exceed 5% of a bib (record the
  threshold in the script header).
- Commands: run the script on all five venues (paste all five outputs);
  rebuild arXiv (`grep -ci undefined main.log` — expected: 0).

## Verification (agent MUST run and report)
- The five script outputs (bib / cited / uncited per venue), pasted.
- `ls` of the five new `references_unused.bib` files with entry counts.
- The per-key decisions for the suspicious keys, listed.
- ArXiv rebuild log — 0 undefined cites after the prune.
- `git diff --stat` and `git status --short` — confirm only the listed files changed.
- Checklist: [ ] one recount, discrepancies settled [ ] uncited archived, none deleted [ ] no cited entry removed [ ] suspicious keys decided [ ] rebuild still clean [ ] all 12 CIT fixes applied + verified [ ] old audit files annotated superseded.

## Definition of done
- Every venue bib is ≥95% cited entries; the rest live in `references_unused.bib`.
- One counting script exists and is the single source for the numbers.

## Report back
- Files changed and created (exact paths).
- The final per-venue numbers (from the one script).
- How the two conflicting runs were settled.
- The suspicious-key decisions.
- Any new finding.

## Stop conditions
- If the recount shows a cited key missing from the bib — STOP for that key and report it; do not invent the entry.
- If a `.bib` file is shared between venues (one edit affects two venues) — STOP and report the sharing; do not prune it twice.
- If a test that passed before now fails — STOP, do not improvise, report exactly what is blocked.
```

