"""
NeuralOps Runners — pipeline orchestration, continuous sweeps, and monitoring.

Subpackages
-----------
module_sweeps
    Per-module sweep scripts (``run_all_4_<module>.py``) that run the full
    four-model benchmark suite for each attention / compression module.

Top-level modules
-----------------
unified_orchestrator
    Central orchestration backend managing multi-model sweep scheduling.
run_master_pipeline
    Master pipeline orchestrator with idempotent task queuing.
run_autonomous_48h_pipeline
    48-hour autonomous multi-stream GPU pipeline.
run_continuous_orchestrator
    Continuous GPU orchestrator for sustained high-utilisation training.
run_monitor_server
    Production telemetry + benchmark database web server.
"""

from .unified_orchestrator import UnifiedOrchestrator

__all__ = ["UnifiedOrchestrator"]
