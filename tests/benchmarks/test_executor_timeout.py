import multiprocessing as mp
from unittest.mock import MagicMock
from neuralops.benchmarks.executor import execute_single


def test_executor_handles_timeout(monkeypatch):
    mock_proc = MagicMock()
    # is_alive initially True, then killed
    mock_proc.is_alive.side_effect = [True, False, False]
    mock_proc.exitcode = -15

    monkeypatch.setattr(mp, "get_context", lambda ctx: MagicMock(
        Process=lambda **kw: mock_proc,
        Pipe=lambda duplex: (MagicMock(), MagicMock())
    ))

    result = execute_single(
        module="baseline",
        model="bert-base-uncased",
        dataset="sst2",
        timeout_s=1,
    )
    assert result.status == "timeout"
    assert result.timed_out is True
    assert "Execution timed out" in result.error
