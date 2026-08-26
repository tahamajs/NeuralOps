"""NeuralOps Automated Phase 2 Quality Sweep Runner & Report Orchestrator."""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Codes"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AutoPhase2Runner")


def wait_for_efficiency_and_run_quality(
    efficiency_log: Path = Path("D:/SVD_Project/adaptive_sweep_v3.log"),
    poll_interval: int = 15,
    max_steps: int = 20,
    epochs: int = 3,
    seeds: str = "42,123,777",
    models: str = "bert,roberta,distilbert,t5",
    datasets: str = "sst2,mrpc,cola,stsb",
    precision: str = "fp16",
):
    logger.info("================================================================================")
    logger.info("📡 AUTO PHASE 2 ORCHESTRATOR STARTED")
    logger.info("Monitoring Efficiency Log: %s", efficiency_log)
    logger.info("================================================================================")

    while True:
        is_done = False
        completed = 0
        total = 720

        if efficiency_log.exists():
            try:
                with open(efficiency_log, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                for line in reversed(lines[-50:]):
                    if "Adaptive Benchmark Sweep Completed" in line or "ALL EVALUATIONS COMPLETED" in line or "Sweep finished" in line:
                        is_done = True
                        break
                
                pass_count = sum(1 for line in lines if " -> PASS" in line or " -> FAIL" in line or " -> TIMEOUT" in line)
                if pass_count >= 720:
                    is_done = True
                completed = pass_count
            except Exception as e:
                logger.warning("Error reading log: %s", e)

        if is_done or completed >= total:
            logger.info("🎯 Efficiency sweep complete! (%d/%d combos). Launching GPU Retest Suite...", completed, total)
            break

        logger.info("⏳ Efficiency sweep in progress: %d/%d combos. Waiting %ds...", completed, total, poll_interval)
        time.sleep(poll_interval)

    # 1. Run GPU Retest Suite on Heavy Modules
    logger.info("================================================================================")
    logger.info("🔬 RUNNING TARGETED GPU RETEST SUITE ON HEAVY MODULES")
    logger.info("================================================================================")
    unsupported_modules = set()
    try:
        from framework.retest_heavy_modules_gpu import run_gpu_retest
        retest_results = run_gpu_retest(output_dir="results/retest_heavy_modules")
        for mod, rdata in retest_results.items():
            if rdata.get("verdict") == "UNSUPPORTED":
                unsupported_modules.add(mod)
                logger.warning("Module '%s' marked UNSUPPORTED and excluded from Phase 2 selection", mod)
            else:
                logger.info("Module '%s' verified PASS (%s)", mod, rdata.get("verdict"))
    except Exception as e:
        logger.error("GPU retest suite failed: %s", e)

    ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = PROJECT_DIR / "results" / f"run_quality_{ts_now}"
    target_dir.mkdir(parents=True, exist_ok=True)

    from framework.run_quality_sweep import DEFAULT_QUALITY_MODULES, run_quality_benchmark_sweep
    active_quality_modules = [m for m in DEFAULT_QUALITY_MODULES if m not in unsupported_modules]

    sds = [int(s.strip()) for s in seeds.split(",")]
    mdls = [m.strip() for m in models.split(",")]
    dts = [d.strip() for d in datasets.split(",")]

    logger.info("🚀 Launching Phase 2 Quality Sweep with %d modules in: %s", len(active_quality_modules), target_dir)
    res_dir = run_quality_benchmark_sweep(
        modules=active_quality_modules,
        models=mdls,
        datasets=dts,
        seeds=sds,
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=3e-5,
        weight_decay=0.01,
        precision=precision,
        output_dir=str(target_dir),
    )

    logger.info("📊 Generating publication report...")
    from framework.generate_publication_report import generate_full_publication_report
    generate_full_publication_report(str(target_dir))

    logger.info("================================================================================")
    logger.info("🎉 PHASE 2 QUALITY SWEEP & PUBLICATION REPORT COMPLETED SUCCESSFULLY!")
    logger.info("Output: %s", target_dir)
    logger.info("================================================================================")


if __name__ == "__main__":
    wait_for_efficiency_and_run_quality()
