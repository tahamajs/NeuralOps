import pytest
from neuralops.benchmarks.kd.teacher_cache import select_strided_layer_indices


def test_strided_layers_algorithm():
    assert select_strided_layer_indices(22) == [3, 7, 11, 15, 19, 21]
    assert select_strided_layer_indices(24) == [3, 7, 11, 15, 19, 23]
    assert select_strided_layer_indices(0) == []
