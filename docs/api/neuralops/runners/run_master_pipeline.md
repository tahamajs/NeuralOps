Module neuralops.runners.run_master_pipeline
============================================
Master pipeline orchestrator.
Runs 2 workers total across all task types.
Priority order: zeroshot > rank_sweep > c4_ppl > latency > multiseed.
Idempotent: skips completed tasks.

Functions
---------

`main()`
:   

`run_task(task: dict) ‑> dict`
: