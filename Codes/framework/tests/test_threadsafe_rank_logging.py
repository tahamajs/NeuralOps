"""
Unit tests for thread-safe QueueHandler logging and isolated rank-specific log files.
"""

import concurrent.futures
import json
import logging
import os
import shutil
import tempfile
import time
import unittest

from framework.experiment_logging.config import LoggingConfig
from framework.experiment_logging.logger import (get_logger, get_rank,
                                                 setup_logging,
                                                 shutdown_logging)


class TestThreadSafeRankLogging(unittest.TestCase):
    """Test suite verifying QueueHandler thread-safety and rank_{LOCAL_RANK}.log isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutdown_logging()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_rank_specific_file_creation(self):
        config = LoggingConfig(
            log_dir=self.temp_dir,
            rank=3,
            rank_specific_logs=True,
            use_queue_handler=True,
            log_to_file=True,
            log_to_console=False,
            log_format="structured",
        )
        logger = setup_logging(config)
        logger.info("Message from Rank 3")

        # Flush queue
        shutdown_logging()

        expected_log = os.path.join(self.temp_dir, "rank_3.log")
        self.assertTrue(os.path.exists(expected_log))
        
        with open(expected_log, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        self.assertTrue(any("Message from Rank 3" in item["message"] for item in lines))
        self.assertEqual(lines[0]["rank"], 3)

    def test_multithreaded_queue_logging_safety(self):
        config = LoggingConfig(
            log_dir=self.temp_dir,
            rank=0,
            rank_specific_logs=True,
            use_queue_handler=True,
            log_to_file=True,
            log_to_console=False,
            log_format="structured",
        )
        logger = setup_logging(config)

        def worker_log(thread_id: int):
            for i in range(25):
                logger.info(f"Worker {thread_id} log message {i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_log, tid) for tid in range(8)]
            concurrent.futures.wait(futures)

        shutdown_logging()

        expected_log = os.path.join(self.temp_dir, "rank_0.log")
        self.assertTrue(os.path.exists(expected_log))
        
        with open(expected_log, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        # 8 workers * 25 messages = 200 messages + 1 init message = 201 messages
        worker_messages = [item for item in lines if "Worker" in item.get("message", "")]
        self.assertEqual(len(worker_messages), 200)


if __name__ == "__main__":
    unittest.main()
