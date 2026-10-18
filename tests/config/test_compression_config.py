"""Tests for neuralops.config.compression_config."""

from dataclasses import asdict, fields
import pytest

from neuralops.config import CompressionConfig


class TestCompressionConfig:
    """Test CompressionConfig dataclass fields, ranks, and serialization."""

    def test_creation(self):
        cfg = CompressionConfig()
        assert cfg is not None

    def test_has_rank_field(self):
        cfg = CompressionConfig()
        field_names = {f.name for f in fields(cfg)}
        rank_fields = [n for n in field_names if "rank" in n.lower()]
        assert len(rank_fields) > 0

    def test_custom_rank(self):
        cfg = CompressionConfig(rank=128)
        assert cfg.rank == 128

    def test_serialization(self):
        cfg = CompressionConfig()
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert len(d) > 20

    def test_config_field_count(self):
        cfg = CompressionConfig()
        field_count = len(fields(cfg))
        assert field_count >= 50
