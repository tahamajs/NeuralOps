"""Unit tests for neuralops.analysis.module_analysis package."""

import pytest
from pathlib import Path
from neuralops.analysis.module_analysis import (
    determine_sub_category,
    determine_fix_priority,
    get_best_dataset_result,
    resolve_code_path,
    ANALYSIS_FIELDNAMES,
)


def test_determine_sub_category():
    assert determine_sub_category("mamba", "Attention Replacement", "VALID") == "SSM"
    assert determine_sub_category("cosformer", "Attention Replacement", "VALID") == "Linear-Attention"
    assert determine_sub_category("mla", "Attention Replacement", "VALID") == "Sparse-Attention"
    assert determine_sub_category("gqa", "Attention Replacement", "VALID") == "GQA-MQA"
    assert determine_sub_category("svd", "Attention Replacement", "VALID") == "SVD-Attention"
    assert determine_sub_category("test_comp", "Compressor", "VALID") == "Compression-Working"
    assert determine_sub_category("test_comp", "Compressor", "BROKEN") == "Compression-Failed"
    assert determine_sub_category("pruning_utils", "Utility", "NO_DATA") == "Utility"


def test_determine_fix_priority():
    assert determine_fix_priority("BROKEN", "Attention Replacement", "Linear-Attention") == "P0"
    assert determine_fix_priority("FALLBACK", "Compressor", "Compression-Failed") == "P1"
    assert determine_fix_priority("FALLBACK", "Attention Replacement", "Linear-Attention") == "P2"
    assert determine_fix_priority("NO_DATA", "Utility", "Utility") == "P3"
    assert determine_fix_priority("VALID", "Compressor", "Compression-Working") == "NONE"


def test_get_best_dataset_result():
    rows = [
        {"_detected_ds": "sst2", "accuracy_pct": "85.2"},
        {"_detected_ds": "sst2", "accuracy_pct": "91.5"},
        {"_detected_ds": "wikitext2", "perplexity": "18.4"},
        {"_detected_ds": "wikitext2", "perplexity": "15.2"},
    ]
    assert get_best_dataset_result(rows, "sst2") == "91.50%"
    assert get_best_dataset_result(rows, "wikitext2") == "15.20"
    assert get_best_dataset_result(rows, "mnli") == "N/A"


def test_resolve_code_path():
    path = resolve_code_path("AdaptiveSVD")
    assert isinstance(path, Path)
    assert path.name == "adaptive_svd_attention.py"


def test_analysis_fieldnames_length():
    assert len(ANALYSIS_FIELDNAMES) == 25
