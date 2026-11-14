from neuralops.benchmarks.roofline import calculate_flops, calculate_bandwidth_bytes, roofline_metrics


def test_roofline_real_calculations():
    # Flops calculation for full attention
    flops = calculate_flops("full", B=1, H=4, S=128, D=64)
    # 4 * 1 * 4 * (128*128) * 64 = 16,777,216
    assert flops == 16777216

    bytes_moved = calculate_bandwidth_bytes("full", B=1, H=4, S=128, D=64, dtype="float16")
    # 4 * 1 * 4 * 128 * 64 * 2 = 262,144
    assert bytes_moved == 262144

    metrics = roofline_metrics("full", B=1, H=4, S=128, D=64, latency_ms=1.0)
    assert metrics["flops"] == flops
    assert metrics["bytes"] == bytes_moved
    assert metrics["arithmetic_intensity"] == flops / bytes_moved
    assert metrics["achieved_tflops"] > 0.0
