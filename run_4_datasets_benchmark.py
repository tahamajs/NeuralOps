"""Multi-dataset KD benchmark runner across 4 distinct datasets using pretrained BERT checkpoints.

Datasets:
1. glue:mrpc (textattack/bert-base-uncased-MRPC)
2. glue:rte (textattack/bert-base-uncased-RTE)
3. glue:qnli (textattack/bert-base-uncased-QNLI)
4. rotten_tomatoes (textattack/bert-base-uncased-rotten-tomatoes)
"""

import sys
import logging
from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.runner import KDRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/tahoora/logs/kd_4_datasets.log"),
    ],
)

DATASET_SUITE = [
    {
        "name": "glue:mrpc",
        "teacher": "textattack/bert-base-uncased-MRPC",
        "pre": 100,
        "post": 408,
    },
    {
        "name": "glue:rte",
        "teacher": "textattack/bert-base-uncased-RTE",
        "pre": 100,
        "post": 277,
    },
    {
        "name": "glue:qnli",
        "teacher": "textattack/bert-base-uncased-QNLI",
        "pre": 200,
        "post": 1000,
    },
    {
        "name": "rotten_tomatoes",
        "teacher": "textattack/bert-base-uncased-rotten-tomatoes",
        "pre": 200,
        "post": 1000,
    },
]

for ds_info in DATASET_SUITE:
    task_name = ds_info["name"]
    safe_name = task_name.replace(":", "_")
    out_dir = f"/home/tahoora/Project/results/kd_{safe_name}"

    logging.info("=" * 75)
    logging.info(f"STARTING BENCHMARK FOR DATASET: {task_name}")
    logging.info(f"Teacher (BERT with trained classification head): {ds_info['teacher']}")
    logging.info(f"Output directory: {out_dir}")
    logging.info("=" * 75)

    cfg = BenchmarkConfig(
        task=task_name,
        teacher=ds_info["teacher"],
        pre_kd_samples=ds_info["pre"],
        post_kd_samples=ds_info["post"],
        batch_size=4,
        grad_accum_steps=4,
        max_length=64,
        kd_steps=200,
        kd_lr=1e-5,
        eval_every=20,
        output_dir=out_dir,
        resume=True,
    )

    runner = KDRunner(cfg)
    runner.run()

logging.info("ALL 4 DATASETS BENCHMARK COMPLETED SUCCESSFULLY!")
