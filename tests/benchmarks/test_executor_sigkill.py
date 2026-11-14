import multiprocessing as mp
from unittest.mock import MagicMock
from neuralops.benchmarks.executor import execute_single


def test_executor_handles_sigkill(monkeypatch):
    # Simulate a worker killed with SIGKILL (exitcode -9 or 137)
    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = False
    mock_proc.exitcode = -9  # SIGKILL

    mock_conn = MagicMock()
    mock_conn.poll.return_value = False

    monkeypatch.setattr(mp, "get_context", lambda ctx: MagicMock(
        Process=lambda **kw: mock_proc,
        Pipe=lambda duplex: (mock_conn, MagicMock())
    ))

    result = execute_single(
        module="baseline",
        model="bert-base-uncased",
        dataset="sst2",
        output_dir="results",
    )
    assert result.status == "hardware_error"
    assert "terminated abnormally" in result.error or "terminated abruptly" in result.error
