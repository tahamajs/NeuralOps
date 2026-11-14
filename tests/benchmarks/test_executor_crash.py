import multiprocessing as mp
from unittest.mock import MagicMock
from neuralops.benchmarks.executor import execute_single


def test_executor_handles_eof_on_clean_exitcode(monkeypatch):
    mock_proc = MagicMock()
    mock_proc.is_alive.return_value = False
    mock_proc.exitcode = 0

    mock_conn = MagicMock()
    mock_conn.recv.side_effect = EOFError("Connection broken abruptly")

    monkeypatch.setattr(mp, "get_context", lambda ctx: MagicMock(
        Process=lambda **kw: mock_proc,
        Pipe=lambda duplex: (mock_conn, MagicMock())
    ))

    result = execute_single(
        module="baseline",
        model="bert-base-uncased",
        dataset="sst2",
    )
    assert result.status == "failed"
    assert "Worker process terminated abruptly" in result.error
