from unittest.mock import MagicMock
from neuralops.benchmarks.suites.cuda_ops import CudaOpsBenchmark


def test_cuda_ops_raises_when_no_backend(monkeypatch):
    mock_config = MagicMock()
    mock_config.hardware.device = "cpu"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    bm = CudaOpsBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.setup()

    mock_adapter = MagicMock()
    mock_adapter._cuda_fn = None
    mock_adapter._triton_fn = None
    mock_adapter._pytorch_ref_fn = None
    monkeypatch.setattr("neuralops.benchmarks.suites.cuda_ops.CUDAKernelAdapter", lambda *a, **k: mock_adapter)
    monkeypatch.setattr(bm, "_prepare_inputs", lambda k: ([], {}))

    out = bm.run_iteration(step=0, kernel_name="unsupported_kernel")
    assert out["cuda_latency_ms"] == 0.0
    assert len(bm.results_table) == 1
    assert "No backend available" in bm.results_table[0]["status"]
