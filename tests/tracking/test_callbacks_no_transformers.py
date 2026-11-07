#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying lazy importing and fail-closed errors when transformers is unavailable (F-65)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_tracking_importable_without_transformers() -> None:
    """Verify that importing neuralops.tracking succeeds when transformers is not installed."""
    # Hide transformers in sys.modules
    with patch.dict(sys.modules, {"transformers": None}):
        # Reload neuralops.tracking and callbacks with transformers hidden
        import neuralops.tracking.callbacks as cb_mod

        importlib.reload(cb_mod)

        import neuralops.tracking as tracking_mod

        importlib.reload(tracking_mod)

        # Assert importing succeeded
        assert hasattr(tracking_mod, "TrackerCallback")
        assert hasattr(tracking_mod, "ExperimentTracker")

        # Assert constructing TrackerCallback without transformers raises clear ImportError
        mock_tracker = MagicMock()
        with pytest.raises(ImportError) as exc_info:
            tracking_mod.TrackerCallback(mock_tracker)

        err_msg = str(exc_info.value)
        assert "TrackerCallback requires the 'transformers' package" in err_msg
        assert "pip install" in err_msg

        # Assert calling ExperimentTracker.get_callback() also raises clear ImportError
        tracker = tracking_mod.ExperimentTracker()
        with pytest.raises(ImportError) as exc_info2:
            tracker.get_callback()

        err_msg2 = str(exc_info2.value)
        assert "requires the 'transformers' package" in err_msg2
        assert "pip install" in err_msg2

    # Clean up reload with transformers restored
    import neuralops.tracking.callbacks as cb_mod_restore

    importlib.reload(cb_mod_restore)
    import neuralops.tracking as tracking_mod_restore

    importlib.reload(tracking_mod_restore)
