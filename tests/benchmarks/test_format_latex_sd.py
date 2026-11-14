import pytest

from neuralops.benchmarks.multi_seed import format_latex_sd


def test_format_latex_sd_basic_units():
    # Percentage
    assert format_latex_sd(86.40, 0.32, unit="%") == r"$86.40 \pm 0.32\%$"
    assert format_latex_sd(86.40, 0.32, unit=r"\%") == r"$86.40 \pm 0.32\%$"

    # Speedup / multiplier
    assert format_latex_sd(1.24, 0.03, unit="x") == r"$1.24 \pm 0.03\times$"
    assert format_latex_sd(1.24, 0.03, unit=r"\times") == r"$1.24 \pm 0.03\times$"
    assert format_latex_sd(1.24, 0.03, unit="\\times") == r"$1.24 \pm 0.03\times$"
    assert format_latex_sd(1.24, 0.03, unit="\times") == r"$1.24 \pm 0.03\times$"

    # Latency / memory / throughput
    assert format_latex_sd(4.12, 0.04, unit="ms") == r"$4.12 \pm 0.04\text{ ms}$"
    assert format_latex_sd(12543.20, 112.50, unit="tokens/s") == r"$12543.20 \pm 112.50\text{ tokens/s}$"
    assert format_latex_sd(1024.50, 4.20, unit="MB") == r"$1024.50 \pm 4.20\text{ MB}$"
    assert format_latex_sd(142.15, 1.20, unit="GB/s") == r"$142.15 \pm 1.20\text{ GB/s}$"

    # Unitless
    assert format_latex_sd(27.33, 0.07, unit="") == r"$27.33 \pm 0.07$"


def test_format_latex_sd_precision():
    assert format_latex_sd(12.3456, 0.1234, unit="ms", precision=3) == r"$12.346 \pm 0.123\text{ ms}$"
    assert format_latex_sd(12.3456, 0.1234, unit="x", precision=1) == r"$12.3 \pm 0.1\times$"
    assert format_latex_sd(100.0, 5.0, unit="", precision=0) == r"$100 \pm 5$"
