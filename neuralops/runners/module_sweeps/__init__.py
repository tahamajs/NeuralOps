"""
Module-level sweep scripts — one script per attention/compression module.

Each ``run_all_4_<module>.py`` script runs the full four-benchmark suite
(GPT-2, RoBERTa, DistilBERT, GPT-Neo) for a single NeuralOps module and
writes results to the project ``results/`` directory.

Usage::

    python -m neuralops.runners.module_sweeps.run_all_4_svd --dataset sst2
"""
