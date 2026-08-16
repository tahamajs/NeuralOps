"""
Comprehensive test suite for Docker Manager, Container Controller,
Job Executor (lifecycle, watchdogs, streaming logs, artifacts), and Auto Updater.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import docker
from docker.errors import DockerException, NotFound

from agent.auto_updater import AutoUpdater
from agent.config import AgentConfig
from agent.container_controller import ContainerController
from agent.docker_manager import DockerManager
from agent.job_executor import JobContext, JobExecutor, JobState


class TestDockerManager(unittest.TestCase):
    """Test suite for DockerManager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DockerManager(
            default_shm_size="2g",
            default_mem_limit="8g",
            default_cpus=4.0,
            default_pids_limit=2048,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("docker.from_env")
    def test_verify_runtime_gpu_success(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.containers.run.return_value = b"GPU test output"

        manager = DockerManager()
        verified = manager.verify_runtime()

        self.assertTrue(verified)
        self.assertTrue(manager.has_gpu_runtime)
        mock_client.ping.assert_called_once()
        mock_client.containers.run.assert_called_once()

    @patch("docker.from_env")
    def test_verify_runtime_gpu_failure_cpu_fallback(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_client.ping.return_value = True
        # GPU container throws exception, CPU container succeeds
        mock_client.containers.run.side_effect = [Exception("No GPU runtime"), b"CPU ok"]

        manager = DockerManager()
        verified = manager.verify_runtime()

        self.assertTrue(verified)
        self.assertFalse(manager.has_gpu_runtime)
        self.assertEqual(mock_client.containers.run.call_count, 2)

    @patch("docker.from_env")
    def test_verify_runtime_docker_down(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_client.ping.side_effect = DockerException("Daemon not reachable")

        manager = DockerManager()
        verified = manager.verify_runtime()

        self.assertFalse(verified)
        self.assertFalse(manager.has_gpu_runtime)

    def test_build_device_requests(self):
        # All GPUs
        reqs = self.manager.build_device_requests()
        self.assertIsNotNone(reqs)
        self.assertEqual(reqs[0].count, -1)
        self.assertEqual(reqs[0].capabilities, [["gpu"]])

        # Specific GPU devices string
        reqs_str = self.manager.build_device_requests(gpu_devices="0,1")
        self.assertEqual(reqs_str[0].device_ids, ["0", "1"])

        # Specific GPU devices list
        reqs_list = self.manager.build_device_requests(gpu_devices=["0", "2"])
        self.assertEqual(reqs_list[0].device_ids, ["0", "2"])

        # Specific GPU count
        reqs_count = self.manager.build_device_requests(gpu_count=2)
        self.assertEqual(reqs_count[0].count, 2)

        # Disabled
        reqs_none = self.manager.build_device_requests(gpu_devices="none")
        self.assertIsNone(reqs_none)

    @patch("docker.from_env")
    def test_pull_image(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        manager = DockerManager()
        manager.pull_image("pytorch/pytorch:latest")
        mock_client.images.pull.assert_called_once_with("pytorch/pytorch:latest")

    @patch("docker.from_env")
    def test_run_job_with_gpu_and_isolation(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "mock_container_123456789012"
        mock_client.containers.run.return_value = mock_container

        manager = DockerManager()
        manager.has_gpu_runtime = True

        job_id = "test-job-001"
        job_dir = os.path.join(self.temp_dir, job_id)

        container_id = manager.run_job(
            job_id=job_id,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            command="python train.py",
            env_vars={"EPOCHS": "10", "LR": "0.001"},
            input_data_url="https://dataset.example.com/data.tar.gz",
            gpu_devices="0",
            cpu_limit=4.0,
            memory_limit="16g",
            shm_size="4g",
            job_dir=job_dir,
            read_only=True,
            tmpfs_size="1G",
        )

        self.assertEqual(container_id, mock_container.id)
        mock_client.containers.run.assert_called_once()
        _, kwargs = mock_client.containers.run.call_args

        self.assertEqual(kwargs["image"], "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime")
        self.assertEqual(kwargs["command"], "python train.py")
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["mem_limit"], "16g")
        self.assertEqual(kwargs["shm_size"], "4g")
        self.assertEqual(kwargs["nano_cpus"], int(4.0 * 1e9))
        self.assertIn("NEURALOPS_JOB_ID", kwargs["environment"])
        self.assertEqual(kwargs["environment"]["NEURALOPS_JOB_ID"], job_id)
        self.assertEqual(kwargs["environment"]["EPOCHS"], "10")
        self.assertEqual(kwargs["environment"]["INPUT_DATA_URL"], "https://dataset.example.com/data.tar.gz")
        self.assertIn("device_requests", kwargs)
        self.assertEqual(kwargs["device_requests"][0].device_ids, ["0"])

        # Check volume mounts
        workspace_dir = os.path.join(job_dir, "workspace")
        self.assertIn(workspace_dir, kwargs["volumes"])
        self.assertEqual(kwargs["volumes"][workspace_dir]["bind"], "/workspace")

    @patch("docker.from_env")
    def test_run_job_with_pre_script(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.id = "script_container_123"
        mock_client.containers.run.return_value = mock_container

        manager = DockerManager()
        job_id = "test-script-job"
        job_dir = os.path.join(self.temp_dir, job_id)

        pre_script = "#!/bin/bash\necho 'pre-script running'"
        command = "python evaluate.py"

        cid = manager.run_job_with_pre_script(
            job_id=job_id,
            image="python:3.10-slim",
            command=command,
            pre_script=pre_script,
            job_dir=job_dir,
            use_gpu=False,
        )

        self.assertEqual(cid, "script_container_123")
        entrypoint_path = os.path.join(job_dir, "workspace", "entrypoint.sh")
        self.assertTrue(os.path.exists(entrypoint_path))
        with open(entrypoint_path) as f:
            content = f.read()
            self.assertIn("pre-script running", content)
            self.assertIn("python evaluate.py", content)

    @patch("docker.from_env")
    def test_get_container_info_and_status(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.name = "neuralops-job-test"
        mock_container.attrs = {
            "State": {
                "Running": True,
                "ExitCode": 0,
                "OOMKilled": False,
                "StartedAt": "2026-08-16T10:00:00Z",
                "FinishedAt": "",
            }
        }
        mock_client.containers.get.return_value = mock_container

        manager = DockerManager()
        status = manager.get_container_status("cid_123")
        self.assertEqual(status, "running")

        info = manager.get_container_info("cid_123")
        self.assertEqual(info["status"], "running")
        self.assertTrue(info["running"])
        self.assertEqual(info["exit_code"], 0)
        self.assertFalse(info["oom_killed"])

    @patch("docker.from_env")
    def test_get_container_info_not_found(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_client.containers.get.side_effect = NotFound("Container not found")

        manager = DockerManager()
        status = manager.get_container_status("cid_missing")
        self.assertEqual(status, "not_found")

        info = manager.get_container_info("cid_missing")
        self.assertEqual(info["status"], "not_found")
        self.assertEqual(info["exit_code"], -1)

    @patch("docker.from_env")
    def test_stream_logs(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client
        mock_container = MagicMock()
        mock_container.logs.return_value = [b"line 1\n", b"line 2\n", b"line 3\n"]
        mock_client.containers.get.return_value = mock_container

        manager = DockerManager()
        stream = list(manager.stream_logs("cid_123"))
        self.assertEqual(stream, ["line 1\n", "line 2\n", "line 3\n"])

    @patch("docker.from_env")
    def test_stop_and_cleanup_containers(self, mock_from_env):
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        mock_container = MagicMock()
        mock_client.containers.get.return_value = mock_container

        manager = DockerManager()
        res = manager.stop_container("cid_123", timeout=5, remove=True)
        self.assertTrue(res)
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(v=True, force=True)

        # Test cleanup
        exited_container = MagicMock()
        exited_container.status = "exited"
        running_container = MagicMock()
        running_container.status = "running"
        mock_client.containers.list.return_value = [exited_container, running_container]

        removed = manager.cleanup()
        self.assertEqual(removed, 1)
        exited_container.remove.assert_called_once_with(force=True, v=True)
        running_container.remove.assert_not_called()


class TestContainerController(unittest.TestCase):
    """Test suite for ContainerController."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.controller = ContainerController(client=self.mock_client)

    def test_pause_and_unpause(self):
        mock_container = MagicMock()
        self.mock_client.containers.get.return_value = mock_container

        # Pause
        res = self.controller.pause("cid_123")
        self.assertEqual(res["status"], "paused")
        mock_container.pause.assert_called_once()

        # Unpause
        res = self.controller.unpause("cid_123")
        self.assertEqual(res["status"], "running")
        mock_container.unpause.assert_called_once()

    def test_stop_restart_kill(self):
        mock_container = MagicMock()
        self.mock_client.containers.get.return_value = mock_container

        # Stop
        res = self.controller.stop("cid_123", timeout=5)
        self.assertEqual(res["status"], "stopped")
        mock_container.stop.assert_called_once_with(timeout=5)

        # Restart
        res = self.controller.restart("cid_123", timeout=5)
        self.assertEqual(res["status"], "restarted")
        mock_container.restart.assert_called_once_with(timeout=5)

        # Kill
        res = self.controller.kill("cid_123", signal="SIGKILL")
        self.assertEqual(res["status"], "killed")
        mock_container.kill.assert_called_once_with(signal="SIGKILL")

    def test_stats_parsing(self):
        mock_container = MagicMock()
        self.mock_client.containers.get.return_value = mock_container

        # Simulated Docker stats structure
        mock_container.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 500000000, "percpu_usage": [250000000, 250000000]},
                "system_cpu_usage": 1000000000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 300000000},
                "system_cpu_usage": 500000000,
            },
            "memory_stats": {
                "usage": 2048 * 1024 * 1024,
                "limit": 8192 * 1024 * 1024,
                "stats": {"cache": 512 * 1024 * 1024},
            },
            "networks": {
                "eth0": {"rx_bytes": 10240, "tx_bytes": 20480},
                "eth1": {"rx_bytes": 5120, "tx_bytes": 10240},
            },
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "Read", "value": 4096},
                    {"op": "Write", "value": 8192},
                ]
            },
            "pids_stats": {"current": 42},
        }

        stats = self.controller.stats("cid_123")
        self.assertEqual(stats["status"], "active")
        self.assertEqual(stats["cpu_percent"], 80.0)
        self.assertEqual(stats["memory_used_mb"], 1536.0)
        self.assertEqual(stats["memory_limit_mb"], 8192.0)
        self.assertAlmostEqual(stats["memory_pct"], 18.75, places=2)
        self.assertEqual(stats["network_rx_bytes"], 15360)
        self.assertEqual(stats["network_tx_bytes"], 30720)
        self.assertEqual(stats["block_read_bytes"], 4096)
        self.assertEqual(stats["block_write_bytes"], 8192)
        self.assertEqual(stats["pids_count"], 42)

    def test_exec_in_container(self):
        mock_container = MagicMock()
        self.mock_client.containers.get.return_value = mock_container
        exec_mock = MagicMock()
        exec_mock.exit_code = 0
        exec_mock.output = b"hello from container\n"
        mock_container.exec_run.return_value = exec_mock

        res = self.controller.exec_in_container("cid_123", ["echo", "hello"])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(res["output"], "hello from container\n")


class TestJobExecutor(unittest.TestCase):
    """Test suite for JobExecutor lifecycle, watchdogs, streaming, and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_docker = MagicMock()
        self.mock_api = MagicMock()
        self.mock_ws = MagicMock()
        self.executor = JobExecutor(
            docker_manager=self.mock_docker,
            api_client=self.mock_api,
            ws_client=self.mock_ws,
            max_concurrent_jobs=2,
            base_job_dir=self.temp_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_successful_full_lifecycle(self):
        job_id = "job-success-001"
        job_data = {
            "id": job_id,
            "docker_image": "python:3.10-slim",
            "command": "python train.py",
            "timeout_seconds": 60,
        }

        # Mock docker actions
        self.mock_docker.run_job_with_pre_script.return_value = "cid_success_123"
        self.mock_docker.stream_logs.return_value = ["Training epoch 1...\n", "Training epoch 2...\n"]
        self.mock_docker.get_container_info.return_value = {
            "status": "exited",
            "exit_code": 0,
            "oom_killed": False,
        }
        self.mock_docker.get_container_logs.return_value = "Training completed."

        # Create a mock artifact in the job directory before execution completes
        workspace_dir = os.path.join(self.temp_dir, job_id, "workspace")
        artifacts_dir = os.path.join(workspace_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, "model.pt"), "w") as f:
            f.write("mock weights")

        # Run pipeline directly
        ctx = JobContext(job_id=job_id, job_data=job_data, job_dir=os.path.join(self.temp_dir, job_id))
        self.executor._active_jobs[job_id] = ctx
        self.executor._run_job_pipeline(ctx)

        # Verify state transitions
        self.assertEqual(ctx.state, JobState.COMPLETED)
        self.assertEqual(ctx.exit_code, 0)
        self.assertTrue(len(ctx.output_hash) > 0)
        self.assertEqual(len(ctx.artifacts_manifest), 1)
        self.assertEqual(ctx.artifacts_manifest[0]["path"], "artifacts/model.pt")

        # Verify API calls
        self.mock_docker.pull_image.assert_called_once_with("python:3.10-slim")
        self.mock_api.report_job_completed.assert_called_once()
        self.mock_ws.send_job_update.assert_called()

        # Verify container cleanup
        self.mock_docker.stop_container.assert_called_with("cid_success_123", timeout=5, remove=True)

    def test_job_failure_on_non_zero_exit_code(self):
        job_id = "job-fail-exitcode"
        job_data = {
            "id": job_id,
            "docker_image": "python:3.10-slim",
            "command": "python crash.py",
        }

        self.mock_docker.run_job_with_pre_script.return_value = "cid_fail_123"
        self.mock_docker.stream_logs.return_value = ["Traceback: NameError..."]
        self.mock_docker.get_container_info.return_value = {
            "status": "exited",
            "exit_code": 1,
            "oom_killed": False,
        }
        self.mock_docker.get_container_logs.return_value = "Traceback (most recent call last):..."

        ctx = JobContext(job_id=job_id, job_data=job_data, job_dir=os.path.join(self.temp_dir, job_id))
        self.executor._active_jobs[job_id] = ctx
        self.executor._run_job_pipeline(ctx)

        self.assertEqual(ctx.state, JobState.FAILED)
        self.assertEqual(ctx.exit_code, 1)
        self.mock_api.report_job_failed.assert_called_once()
        args, _ = self.mock_api.report_job_failed.call_args
        self.assertIn("non-zero exit code: 1", args[1])

    def test_job_failure_on_oom_killed(self):
        job_id = "job-oom"
        job_data = {"id": job_id, "docker_image": "python:3.10-slim", "memory_limit": "2g"}

        self.mock_docker.run_job_with_pre_script.return_value = "cid_oom_123"
        self.mock_docker.stream_logs.return_value = ["Allocating tensor 10GB..."]
        self.mock_docker.get_container_info.return_value = {
            "status": "exited",
            "exit_code": 137,
            "oom_killed": True,
        }

        ctx = JobContext(job_id=job_id, job_data=job_data, job_dir=os.path.join(self.temp_dir, job_id))
        self.executor._active_jobs[job_id] = ctx
        self.executor._run_job_pipeline(ctx)

        self.assertEqual(ctx.state, JobState.FAILED)
        self.mock_api.report_job_failed.assert_called_once()
        args, _ = self.mock_api.report_job_failed.call_args
        self.assertIn("killed by Linux OOM killer", args[1])

    def test_job_cancellation(self):
        job_id = "job-cancel-test"
        job_data = {"id": job_id}

        ctx = JobContext(job_id=job_id, job_data=job_data, job_dir=os.path.join(self.temp_dir, job_id))
        ctx.container_id = "cid_to_cancel"
        ctx.state = JobState.RUNNING
        self.executor._active_jobs[job_id] = ctx

        cancelled = self.executor.cancel(job_id, reason="User cancelled from dashboard")
        self.assertTrue(cancelled)
        self.assertEqual(ctx.state, JobState.CANCELLED)
        self.mock_docker.stop_container.assert_called_once_with("cid_to_cancel", timeout=2, remove=True)
        self.mock_api.report_job_failed.assert_called_once()
        self.assertNotIn(job_id, self.executor._active_jobs)

    def test_concurrency_limit(self):
        # Fill up concurrency limit
        self.executor._active_jobs["job1"] = JobContext("job1", {}, "/tmp/j1")
        self.executor._active_jobs["job1"].state = JobState.RUNNING
        self.executor._active_jobs["job2"] = JobContext("job2", {}, "/tmp/j2")
        self.executor._active_jobs["job2"].state = JobState.RUNNING

        accepted = self.executor.execute({"id": "job3"})
        self.assertFalse(accepted)
        self.mock_api.report_job_failed.assert_called_once()

    def test_artifact_manifest_computation(self):
        job_id = "artifact-test"
        job_dir = os.path.join(self.temp_dir, job_id)
        workspace = os.path.join(job_dir, "workspace")
        artifacts = os.path.join(workspace, "artifacts")
        os.makedirs(artifacts, exist_ok=True)

        with open(os.path.join(artifacts, "metrics.json"), "w") as f:
            f.write('{"accuracy": 0.985}')
        with open(os.path.join(artifacts, "model.bin"), "wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05")

        output_hash, manifest = self.executor._compute_output_artifacts(job_dir)
        self.assertTrue(len(output_hash) == 64)
        self.assertEqual(len(manifest), 2)
        paths = [item["path"] for item in manifest]
        self.assertIn("artifacts/metrics.json", paths)
        self.assertIn("artifacts/model.bin", paths)

    def test_timeout_watchdog(self):
        job_id = "job-timeout-test"
        job_data = {
            "id": job_id,
            "docker_image": "python:3.10-slim",
            "timeout_seconds": 1,
        }

        self.mock_docker.run_job_with_pre_script.return_value = "cid_timeout_123"
        self.mock_docker.stream_logs.return_value = []
        # Simulate container running past timeout
        self.mock_docker.get_container_info.return_value = {
            "status": "running",
            "exit_code": None,
            "oom_killed": False,
        }

        ctx = JobContext(job_id=job_id, job_data=job_data, job_dir=os.path.join(self.temp_dir, job_id))
        # Backdate start_time to trigger watchdog immediately
        ctx.start_time = time.time() - 10.0
        self.executor._active_jobs[job_id] = ctx
        self.executor._run_job_pipeline(ctx)

        self.assertEqual(ctx.state, JobState.FAILED)
        self.mock_docker.stop_container.assert_any_call("cid_timeout_123", timeout=2, remove=True)
        self.mock_api.report_job_failed.assert_called_once()
        args, _ = self.mock_api.report_job_failed.call_args
        self.assertIn("timed out", args[1])

    def test_job_execute_and_get_status(self):
        job_id = "job-status-query"
        job_data = {"id": job_id, "docker_image": "python:3.10-slim"}

        with patch.object(threading.Thread, "start"):
            accepted = self.executor.execute(job_data)
            self.assertTrue(accepted)

            status = self.executor.get_job_status(job_id)
            self.assertIsNotNone(status)
            self.assertEqual(status["job_id"], job_id)
            self.assertEqual(status["state"], JobState.QUEUED.value)

            active_jobs = self.executor.get_active_jobs()
            self.assertIn(job_id, active_jobs)


class TestAutoUpdater(unittest.TestCase):
    """Test suite for AutoUpdater."""

    def setUp(self):
        self.config = AgentConfig()
        self.config.version = "1.0.0"
        self.config.auto_update = True
        self.mock_api = MagicMock()
        self.mock_api.check_for_updates.side_effect = lambda v: self.mock_api._post("/api/agent/check-update/", {"version": v})
        self.updater = AutoUpdater(self.config, self.mock_api)

    def test_version_newer(self):
        self.assertTrue(self.updater._version_newer("1.1.0", "1.0.0"))
        self.assertTrue(self.updater._version_newer("2.0.0", "1.9.9"))
        self.assertTrue(self.updater._version_newer("1.0.1", "1.0.0"))
        self.assertFalse(self.updater._version_newer("1.0.0", "1.0.0"))
        self.assertFalse(self.updater._version_newer("0.9.0", "1.0.0"))

    def test_auto_update_disabled(self):
        self.config.auto_update = False
        res = self.updater.check_and_update()
        self.assertFalse(res)
        self.mock_api._post.assert_not_called()

    def test_check_and_update_no_update_available(self):
        self.mock_api._post.return_value = {"version": "1.0.0"}
        res = self.updater.check_and_update()
        self.assertFalse(res)

    def test_check_and_update_source_mode(self):
        self.mock_api._post.return_value = {
            "version": "1.2.0",
            "download_url": "https://releases.neuralops.com/agent-1.2.0.bin",
            "sha256": "abcdef123456",
        }

        # In non-frozen source mode, should notify without downloading binary
        res = self.updater.check_and_update()
        self.assertTrue(res)

    @patch("requests.get")
    def test_download_and_apply_checksum_mismatch(self, mock_requests_get):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"binary data content"]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Correct sha is sha256("binary data content")
        actual_sha = hashlib.sha256(b"binary data content").hexdigest()
        wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"

        res = self.updater._download_and_apply("https://example.com/bin", expected_sha256=wrong_sha)
        self.assertFalse(res)

    @patch("os.execv")
    @patch("os.chmod")
    @patch("shutil.move")
    @patch("requests.get")
    def test_download_and_apply_success(self, mock_requests_get, mock_move, mock_chmod, mock_execv):
        binary_bytes = b"fake binary executable content"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [binary_bytes]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        correct_sha = hashlib.sha256(binary_bytes).hexdigest()
        temp_target = tempfile.NamedTemporaryFile(delete=False)
        temp_target.close()

        try:
            res = self.updater._download_and_apply(
                url="https://example.com/agent-bin",
                expected_sha256=correct_sha,
                target_path=temp_target.name,
            )
            self.assertTrue(res)
            mock_move.assert_called()
            mock_chmod.assert_called_with(temp_target.name, 0o755)
            mock_execv.assert_called_once()
        finally:
            if os.path.exists(temp_target.name):
                os.remove(temp_target.name)


if __name__ == "__main__":
    unittest.main()

