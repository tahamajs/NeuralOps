"""End-to-end integration tests for the full pipeline."""

import pytest
import torch
import warnings


@pytest.mark.integration
class TestCrossModuleCompat:
    """Test registry stability and cross-module registration compatibility."""

    def test_attentions_registry_stable(self):
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401
        assert len(ATTENTION_REGISTRY.keys()) >= 100

    def test_compressors_registry_stable(self):
        from neuralops.core.registry import COMPRESSOR_REGISTRY
        import neuralops.compression_engine  # noqa: F401
        assert len(COMPRESSOR_REGISTRY.keys()) >= 60

    def test_deprecated_aliases_work(self):
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls = ATTENTION_REGISTRY.get("aegis_v2")
        assert cls is not None


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end pipeline execution smoke tests."""

    @pytest.mark.slow
    def test_attention_injection_pipeline(self):
        """Zero-shot attention injection on small model architecture."""
        pytest.importorskip("transformers")
        from transformers import AutoModel, AutoConfig
        from neuralops.core.replace_attention import replace_attention_in_model
        from neuralops.core.registry import ATTENTION_REGISTRY
        import neuralops.attentions  # noqa: F401

        cfg = AutoConfig.from_pretrained("distilbert-base-uncased")
        cfg.n_layers = 1
        model = AutoModel.from_config(cfg)

        model = replace_attention_in_model(model, "svd_linear_attention")

        x = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            out = model(x)
        assert out is not None
