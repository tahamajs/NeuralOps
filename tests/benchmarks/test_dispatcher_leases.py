from neuralops.benchmarks.storage.dispatcher import MultiGPUDispatcher, VRAMGovernor


def test_dispatcher_acquire_release():
    dispatcher = MultiGPUDispatcher(device_ids=[0, 1])
    lease = dispatcher.acquire_device(task_id="task_1", preferred_device=0)
    assert lease is not None
    assert lease.device_id == 0
    assert "task_1" in dispatcher._active_leases[0]

    dispatcher.release_device(lease)
    assert "task_1" not in dispatcher._active_leases[0]
    assert lease.active is False


def test_vram_governor_non_cuda():
    gov = VRAMGovernor(min_free_mb=1024)
    # On environments without CUDA, can_schedule allows scheduling
    import torch

    if not torch.cuda.is_available():
        assert gov.can_schedule(0) is True
