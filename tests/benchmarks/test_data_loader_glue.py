import pytest
from neuralops.benchmarks.core.data_loader import GLUE_TASKS


def test_glue_tasks_registered():
    assert "sst2" in GLUE_TASKS
    assert "mrpc" in GLUE_TASKS
    assert GLUE_TASKS["sst2"]["num_labels"] == 2
    assert GLUE_TASKS["sst2"]["val_size"] == 872
