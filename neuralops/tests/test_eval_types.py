"""Tests for eval_types module."""

import math
from neuralops.evaluation.eval_types import (
    BenchmarkResult,
    TRACK_MODULES,
    MODEL_TOKENIZER_MAPPING,
    LM_DATASETS,
    GLUE_DATASETS,
    ACTIVATION_AWARE_MODULES,
    COMPUTE_OPTIMIZERS,
    KV_CACHE_OPTIMIZERS,
    DEPRECATED_MODULES,
)


def test_benchmark_result_initialization():
    res = BenchmarkResult(
        track="track1_svd",
        module_name="svd_linear_attention",
        module_type="attention",
        model_family="bert",
        dataset="sst2",
        status="PASS",
    )
    assert res.track == "track1_svd"
    assert res.status == "PASS"
    assert res.compression_ratio == 1.0
    assert isinstance(res.to_dict(), dict)


def test_benchmark_result_to_dict_nan_handling():
    res = BenchmarkResult(
        track="track1_svd",
        module_name="test",
        module_type="attention",
        model_family="bert",
        dataset="sst2",
        status="PASS",
        loss=float("nan"),
        perplexity=float("inf"),
    )
    d = res.to_dict()
    assert d["loss"] is None
    assert d["perplexity"] is None


def test_track_modules_structure():
    assert "track1_svd" in TRACK_MODULES
    assert "track2_ssm" in TRACK_MODULES
    assert "track3_moe" in TRACK_MODULES
    assert "track4_kernel_quant" in TRACK_MODULES
    assert "track5_pruning" in TRACK_MODULES

    for track, data in TRACK_MODULES.items():
        assert "attentions" in data
        assert "compressors" in data
        assert isinstance(data["attentions"], list)
        assert isinstance(data["compressors"], list)


def test_constants_sets():
    assert "sst2" in GLUE_DATASETS
    assert "wikitext2" in LM_DATASETS
    assert "bert" in MODEL_TOKENIZER_MAPPING
    assert len(ACTIVATION_AWARE_MODULES) > 0
    assert len(COMPUTE_OPTIMIZERS) > 0
    assert len(KV_CACHE_OPTIMIZERS) > 0
    assert len(DEPRECATED_MODULES) > 0
