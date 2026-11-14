from unittest.mock import patch
import os
import pytest
from neuralops.benchmarks.cli_v2.main import main


def test_cli_v2_doctor_command():
    with patch("neuralops.benchmarks.doctor.cmd_doctor", return_value=0) as mock_doc:
        ret = main(["doctor", "--device", "cpu"])
        assert ret == 0
        mock_doc.assert_called_once()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_cli_v2_ui_command():
    with patch("neuralops.ui.cli.cmd_ui", return_value=0) as mock_ui:
        ret = main(["ui", "--device", "cpu", "--port", "9000"])
        assert ret == 0
        mock_ui.assert_called_once()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
