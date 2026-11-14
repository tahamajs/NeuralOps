import argparse
import pytest
import warnings
from neuralops.benchmarks.core.arguments import (
    add_common_args,
    add_compression_args,
    add_dataset_args,
    add_model_args,
)
from neuralops.benchmarks.core.stats import paired_ttest


def test_arguments_no_conflict():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_model_args(parser)
    add_dataset_args(parser)
    # Should not raise ArgumentError: option string '--rank' or others
    add_compression_args(parser)


def test_paired_ttest_zero_variance_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = paired_ttest([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert p == 1.0
        assert len(w) == 1
        assert "zero variance" in str(w[0].message)
