import pytest
from neuralops.benchmarks.core.data_loader import load_preprocess_script


def test_preprocess_script_requires_trust(tmp_path):
    script_path = tmp_path / "preprocess.py"
    script_path.write_text("def preprocess(batch): return batch\n", encoding="utf-8")

    # default trust=False must raise PermissionError
    with pytest.raises(PermissionError, match="requires explicit trust=True"):
        load_preprocess_script(script_path, trust=False)

    # trust=True succeeds
    fn = load_preprocess_script(script_path, trust=True)
    assert callable(fn)
    assert fn({"x": 1}) == {"x": 1}
