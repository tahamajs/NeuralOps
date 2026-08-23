import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pytest
ret = pytest.main(["Codes/framework/tests/test_full_framework_end_to_end.py", "-v"])
sys.exit(ret)
