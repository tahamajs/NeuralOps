#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for run directory locking and atomic writes."""

import json
from pathlib import Path

import pytest

from neuralops.cli.core.errors import RunDirectoryLockedError
from neuralops.cli.core.locks import atomic_write_json, create_run_directory, run_directory_lock


def test_create_run_directory(tmp_path: Path) -> None:
    out = tmp_path / "run_1"
    res = create_run_directory(out)
    assert res == out.resolve()
    assert res.is_dir()

    # Non-empty directory without overwrite raises
    (res / "file.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(RunDirectoryLockedError):
        create_run_directory(out, overwrite=False)

    # With overwrite=True it succeeds
    res2 = create_run_directory(out, overwrite=True)
    assert res2 == out.resolve()


def test_run_directory_lock(tmp_path: Path) -> None:
    out = tmp_path / "run_lock"
    with run_directory_lock(out):
        assert (out / ".neuralops.lock").exists()
        # Nested acquisition on separate descriptor should fail or be locked
        with pytest.raises(RunDirectoryLockedError):
            with run_directory_lock(out):
                pass


def test_atomic_write_json(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "data.json"
    payload = {"status": "SUCCESS", "speedup": 2.5}
    atomic_write_json(target, payload)

    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload
