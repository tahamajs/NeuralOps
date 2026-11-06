# User decisions for the NeuralOps agent-prompt pack (Appendix B)

Decided by the user on 2026-11-07 (~01:51 Tehran). The user delegated these
decisions to Muse with the instruction: decide yourself what is OK and execute
it. These adopt the recommendations in the prompt pack's Appendix B
(NeuralOps-Agent-Prompts-1405-08-15.md, پیوست ب) and the roadmap's preferences.
Any decision here can be overridden by the user later — if overridden, update
this file and re-run the affected prompt.

| # | Decision | Chosen option | Evidence / note |
|---|---|---|---|
| 1 | One README or two? | DELETE neuralops/README.md → short stub linking to root README + docs/ | pyproject.toml `readme = {file = "README.md"}` — the ROOT README is the PyPI long description, NOT neuralops/README.md. AGT-53's stop rule (do not delete if it is the long description) does NOT apply. Applies to AGT-53 / AGT-62. |
| 2 | F-74: delete operations/ or correct CHANGELOG? | (b) CORRECT the CHANGELOG [2.0.6] entry; do NOT delete neuralops/operations/ | Lower risk until proven nobody depends on operations/. Applies to AGT-27 / AGT-41 / AGT-61. |
| 3 | Three CLIs or one? | ONE canonical CLI: neuralops/benchmarks/cli.py | Console scripts in pyproject.toml all point at neuralops.benchmarks.cli:main. Shadow CLIs are deleted or shimmed per AGT-54. Applies to AGT-14 / AGT-15 / AGT-22 / AGT-54 / AGT-59. |
| 4 | CUDA support policy | Support 12.1 / 12.4 / 12.8 AND KEEP CUDA 11.3 (legacy, Sharif HPC) | User override 2026-11-07 ~01:59 (reverses the delegated decision): CUDA 11.3 stays supported alongside 12.x. The ptxas path for CUDA 11.3 (F-42) is KEPT as a last-resort fallback; 12.x is preferred. Applies to AGT-48 / AGT-13 / AGT-32 / AGT-57. |
| 5 | Notebook execution scope | Only the 29 notebooks executed on HPC; the other 65 are catalogued as "not executed — reason" | Applies to AGT-42. |
| 6 | Example execution scope | Compile ALL 35 + run examples 01 and 33 only | Applies to AGT-53. |
| 7 | CPU-only job skip threshold | 50%, after the AGT-00 baseline skip measurement | Applies to AGT-17. |
| 8 | Coverage targets | 70–85% per subsystem (not one flat 70%) | Applies to AGT-17 / AGT-FINAL-1. |
| 9 | notebooks/extended/: symlink or delete? | SYMLINK, do not delete | Until proven nothing external links to extended/. Applies to AGT-42. |
| 10 | PyPI skip-existing: fail or silently skip? | FAIL | Roadmap preference. Applies to AGT-FINAL-2. |
| 11 | Python 3.13 / 3.14: support or cut? | Officially support 3.10–3.12 ONLY until CI exists for 3.13/3.14 | Classifiers stay as-is until CI is added; docs must not claim 3.13/3.14 support. Applies to AGT-09 / AGT-25. |
| 12 | Canonical Triton count source | The REGISTRY is the single source of truth (owner: F-30/F-38); no number is hardcoded anywhere | Applies to AGT-13 / AGT-53 / AGT-57. |

Related: the release version is 2.1.0 (user decision 2026-11-07). AGT-61 applies it.
