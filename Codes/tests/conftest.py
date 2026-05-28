"""
Pytest configuration for all tests.

Provides shared fixtures, markers, and testing utilities for the entire test suite.
"""

import pytest
import sys
from pathlib import Path

# Add the Codes directory to path
codes_dir = Path(__file__).parent
sys.path.insert(0, str(codes_dir))


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def sample_config():
    """Provide a sample Config object for testing."""
    try:
        from framework.config import Config
        cfg = Config()
        return cfg
    except ImportError:
        pytest.skip("torch not installed")


@pytest.fixture
def sample_paths(temp_dir):
    """Provide sample paths for testing."""
    return {
        "base_dir": temp_dir,
        "model_dir": temp_dir / "models",
        "data_dir": temp_dir / "data",
        "output_dir": temp_dir / "outputs",
    }


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location."""
    for item in items:
        # Mark tests in tests/ as unit tests by default
        if "tests/" in str(item.fspath) and "framework/tests" not in str(item.fspath):
            if "integration" not in str(item.fspath):
                item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seed before each test for reproducibility."""
    import random
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    
    random.seed(42)
    
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    except ImportError:
        pass
    
    yield
