# Cross-Domain Bugs Logged by W4-U1

## 2026-11-15T02:47:00Z
File: neuralops/benchmarks/suites/niah.py
Line: 76
Symptom: NeedleInAHaystackBenchmark raises NotImplementedError("NIAH requires real model; not implemented") instead of delegating to the real NeedleInAHaystackEvaluator in neuralops.evaluation.niah_passkey.
Suspected bug ID: P0-B1-NIAH
Suggested fix: In run_iteration or run, import NeedleInAHaystackEvaluator from neuralops.evaluation.niah_passkey and invoke evaluate_grid with the model and tokenizer instead of raising NotImplementedError.
